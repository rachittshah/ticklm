"""Local, scored-choice LoRA experiment with pinned, validated checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import get_total_parameters

LABELS = "ABCD"
SYSTEM = "Choose the correct answer. Reply with only one letter: A, B, C, or D."
GENERATION_PROMPTS = ["Explain why the sky appears blue in one sentence.",
                      "What is 17 plus 25?", "Write a short polite greeting."]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def reserve_output(output):
    """Reserve an empty output directory so concurrent runs cannot reuse it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output}")
    try:
        with (output / ".run-reserved").open("x") as f:
            f.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing nonempty output directory: {output}") from exc


def training_labels(cases, seed, shuffle):
    require_split(cases, "train")
    answers = [case["answer"] for case in cases]
    if shuffle:
        random.Random(f"{seed}:shuffle-labels").shuffle(answers)
    trained = [{**case, "answer": label} for case, label in zip(cases, answers, strict=True)]
    labels = [{"id": case["id"], "true_answer": case["answer"], "training_answer": label}
              for case, label in zip(cases, answers, strict=True)]
    return trained, labels


def runtime_provenance():
    source_dir = Path(__file__).parent
    sources = ["surgery.py", "surgery_data.py"]
    if (source_dir / "surgery_features.py").exists():
        sources.append("surgery_features.py")
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("mlx", "mlx-lm", "numpy", "transformers")},
            "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                              for name in sources}}


def apply_input_view(cases, view):
    from .surgery_features import with_feature_view
    return with_feature_view(cases, view)


def save_dataset(output, name, cases):
    output = Path(output)
    path = output / "datasets" / f"{name}.json"
    atomic_json(path, cases)
    run_path = output / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    run.setdefault("datasets", {})[name] = {"path": str(path.relative_to(output)), "count": len(cases),
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(run_path, run)


def base_identity(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.name in {"config.json", "generation_config.json", "tokenizer.json",
                               "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                               "merges.txt", "model.safetensors.index.json"} or
                    p.suffix == ".safetensors"))
    if not (path / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("local base must contain config.json and safetensors weights")
    if not any(p.name in {"tokenizer.json", "vocab.json"} for p in files):
        raise ValueError("local base must contain tokenizer files")
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.name] = digest.hexdigest()
    return hashes


def checkpoint_parameter_count(path):
    """Count stored elements, including any duplicated or tied tensors."""
    total = 0
    for file in sorted(Path(path).glob("*.safetensors")):
        with file.open("rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
        total += sum(math.prod(value["shape"]) for key, value in header.items()
                     if key != "__metadata__")
    return total


def choice_ids(tokenizer):
    tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in LABELS]
    if any(len(token) != 1 for token in tokens):
        raise ValueError("answer letters must each encode as a single token")
    ids = [token[0] for token in tokens]
    if len(set(ids)) != 4:
        raise ValueError("answer letters must have distinct token IDs")
    return ids


def pad_tokens(sequences, pad_id, context_limit):
    if not sequences or any(not x or len(x) > context_limit for x in sequences):
        raise ValueError("empty prompt or prompt exceeds model context limit")
    lengths = [len(x) for x in sequences]
    width = max(lengths)
    return mx.array([list(x) + [pad_id] * (width - len(x)) for x in sequences]), lengths


def last_logits(logits, lengths):
    return logits[mx.arange(len(lengths)), mx.array(lengths) - 1, :]


def answer_loss(logits, targets, ids, mode):
    if mode == "choice":
        return nn.losses.cross_entropy(logits[:, mx.array(ids)], targets, reduction="mean")
    if mode == "full":
        return nn.losses.cross_entropy(logits, mx.array(ids)[targets], reduction="mean")
    raise ValueError("unknown loss mode")


def require_split(cases, split):
    if not cases or any(case.get("split") != split for case in cases):
        raise ValueError(f"expected nonempty {split} split only")


def group_pairs(prepared):
    groups = {}
    for item in prepared:
        case, _ = item
        if not case.get("pair_id"):
            raise ValueError("paired batches require a pair_id on every case")
        groups.setdefault(case["pair_id"], []).append(item)
    if not groups or any(len(pair) != 2 or pair[0][0]["id"] == pair[1][0]["id"]
                         for pair in groups.values()):
        raise ValueError("paired batches require exactly two distinct cases per pair")
    return list(groups.values())


def sample_paired_batch(groups, batch_size, rng):
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("paired batch size must be positive and even")
    return [item for _ in range(batch_size // 2) for item in rng.choice(groups)]


def validate_adapter_tensors(expected, actual):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(k for k in set(expected) & set(actual) if expected[k].shape != actual[k].shape)
    if missing or extra or wrong:
        raise ValueError(f"adapter tensors incompatible: missing={missing}, extra={extra}, shape={wrong}")


def configure_adapter(model, layers, rank):
    if layers <= 0 or layers > len(model.layers) or rank <= 0:
        raise ValueError("layers must be positive and no greater than model layers; rank must be positive")
    total = get_total_parameters(model)
    model.freeze()
    config = {"rank": rank, "scale": 1.0, "dropout": 0.0,
              "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    linear_to_lora_layers(model, layers, config)
    tensors = dict(tree_flatten(model.trainable_parameters()))
    trainable = sum(t.size for t in tensors.values())
    if not trainable or trainable / total > 0.01:
        raise ValueError(f"invalid trainable parameter fraction: {trainable}/{total}")
    return config, {"total": total, "trainable": trainable, "fraction": trainable / total,
                    "tensors": {k: {"shape": list(v.shape), "count": v.size}
                                for k, v in tensors.items()}}


def load_adapter(model, directory, identity, revision):
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["base_hashes"] != identity or manifest["revision"] != revision:
        raise ValueError("adapter base identity or revision does not match")
    params = config["lora_parameters"]
    expected_config, counts = configure_adapter(model, config["num_layers"], params["rank"])
    if params != expected_config or config.get("fine_tune_type") != "lora":
        raise ValueError("unsupported adapter configuration")
    weights = mx.load(str(directory / "adapters.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    validate_adapter_tensors(expected, weights)
    model.load_weights(list(weights.items()), strict=False)
    return counts


def tokens_for(tokenizer, prompt, system=SYSTEM):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)


def context_limit(path):
    config = json.loads((Path(path) / "config.json").read_text())
    limit = config.get("max_position_embeddings")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("model config requires max_position_embeddings")
    return limit


def prepare(cases, tokenizer, limit):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS token")
    result = []
    for case in cases:
        tokens = tokens_for(tokenizer, case["prompt"])
        if not tokens or len(tokens) > limit:
            raise ValueError(f"case {case['id']} exceeds context or is empty")
        if case["answer"] not in LABELS:
            raise ValueError("invalid answer")
        result.append((case, tokens))
    return result, pad_id


def score_batch(model, batch, pad_id, limit, ids):
    inputs, lengths = pad_tokens([tokens for _, tokens in batch], pad_id, limit)
    logits = last_logits(model(inputs), lengths).astype(mx.float32)
    selected = logits[:, mx.array(ids)]
    return selected, mx.argmax(logits, axis=-1)


def evaluate_cases(model, tokenizer, cases, limit, batch_size):
    model.eval()
    prepared, pad_id = prepare(cases, tokenizer, limit)
    ids = choice_ids(tokenizer)
    records = []
    timings = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        tick = perf_counter()
        scores, top = score_batch(model, batch, pad_id, limit, ids)
        mx.eval(scores, top)
        timings.append(perf_counter() - tick)
        values, tops = np.asarray(scores), np.asarray(top)
        for (case, tokens), logits, top_id in zip(batch, values, tops, strict=True):
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite candidate logits")
            weights = np.exp(logits - np.max(logits)); weights /= weights.sum()
            prediction = LABELS[int(np.argmax(logits))]
            records.append({**case, "prediction": prediction, "correct": prediction == case["answer"],
                            "choice_probabilities": dict(zip(LABELS, map(float, weights), strict=True)),
                            "unrestricted_top_token_id": int(top_id),
                            "unrestricted_top_token": tokenizer.decode([int(top_id)]),
                            "unrestricted_letter_valid": int(top_id) in ids,
                            "unrestricted_correct_letter": int(top_id) == ids[LABELS.index(case["answer"])],
                            "input_tokens": len(tokens)})
    elapsed = sum(timings)
    pairs = {}
    families = {}
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for record in records:
        if record.get("pair_id") is not None:
            pairs.setdefault(record["pair_id"], []).append(record["correct"])
        families.setdefault(record["family"], []).append(record["correct"])
        confusion[record["answer"]][record["prediction"]] += 1
    complete_pairs = [v for v in pairs.values() if len(v) == 2]
    summary = {"count": len(records), "accuracy": sum(r["correct"] for r in records) / len(records),
               "per_family_accuracy": {k: sum(v) / len(v) for k, v in families.items()},
               "confusion": confusion,
               "pair_both_correct": (sum(all(v) for v in complete_pairs) / len(complete_pairs)
                                     if complete_pairs else None),
               "complete_pairs": len(complete_pairs),
               "unrestricted_letter_validity": sum(r["unrestricted_letter_valid"] for r in records) / len(records),
               "unrestricted_correct_letter_accuracy": sum(r["unrestricted_correct_letter"] for r in records) / len(records),
               "batch_size": batch_size, "scoring_wall_s": elapsed,
               "examples_per_s": len(records) / elapsed, "batch_wall_s": timings,
               "sequential_latency_s": timings if batch_size == 1 else None,
               "peak_memory_bytes": mx.get_peak_memory(), "peak_memory_scope": "process since startup"}
    return records, summary


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            for item in records:
                f.write(json.dumps(item, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_adapter(model, directory, config, layers, identity, revision, counts, step, input_view="raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    atomic_json(directory / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": layers,
                                                     "lora_parameters": config})
    atomic_json(directory / "manifest.json", {"base_hashes": identity, "revision": revision,
                                               "parameters": counts, "step": step,
                                               "training_input_view": input_view})


def inspect_model(model, tokenizer, output):
    model.eval()
    results = []
    for prompt in GENERATION_PROMPTS:
        tokens = tokens_for(tokenizer, prompt, "Answer the user's question concisely and accurately.")
        start = perf_counter()
        response = generate(model, tokenizer, prompt=tokens, max_tokens=80, verbose=False)
        results.append({"prompt": prompt, "response": response, "elapsed_s": perf_counter() - start})
    atomic_json(Path(output) / "generation.json", results)


def run_train(args, model, tokenizer, identity):
    from .surgery_data import build_cases, build_retention_cases
    if args.curriculum:
        from .surgery_data import build_curriculum_cases
        train = build_curriculum_cases(pairs_per_family=args.pairs_per_family, seed=args.seed)
    else:
        train = build_cases("train", pairs_per_family=args.pairs_per_family, seed=args.seed)
    dev = build_cases("dev", pairs_per_family=args.dev_pairs_per_family, seed=args.seed)
    retention_dev = build_retention_cases("dev")
    require_split(train, "train"); require_split(dev, "dev")
    require_split(retention_dev, "dev")
    train = apply_input_view(train, args.input_view)
    dev = apply_input_view(dev, args.input_view)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    save_dataset(output, "temporal_train", train)
    save_dataset(output, "temporal_dev", dev)
    save_dataset(output, "retention_dev", retention_dev)
    training_cases, actual_labels = training_labels(train, args.seed, args.shuffle_labels)
    save_dataset(output, "training_labels", actual_labels)
    prepared, pad_id = prepare(training_cases, tokenizer, context_limit(args.model))
    pairs = group_pairs(prepared) if args.paired_batches else None
    if args.paired_batches and args.batch_size % 2:
        raise ValueError("paired batch size must be even")
    anchor_prepared = None
    if args.anchor_every:
        anchors = build_retention_cases("train")
        require_split(anchors, "train")
        save_dataset(output, "retention_train", anchors)
        anchor_prepared, anchor_pad = prepare(anchors, tokenizer, context_limit(args.model))
        if anchor_pad != pad_id:
            raise ValueError("inconsistent tokenizer padding")
    ids = choice_ids(tokenizer)
    mx.random.seed(args.seed)
    config, counts = configure_adapter(model, args.layers, args.rank)
    opt = optim.Adam(learning_rate=args.learning_rate)
    rng = random.Random(args.seed)
    def loss_fn(model, inputs, lengths, targets, mode):
        logits = last_logits(model(inputs), lengths).astype(mx.float32)
        return answer_loss(logits, targets, ids, mode)
    choice_grad = nn.value_and_grad(model, lambda m, x, lens, y: loss_fn(m, x, lens, y, "choice"))
    full_grad = nn.value_and_grad(model, lambda m, x, lens, y: loss_fn(m, x, lens, y, "full"))
    logs = []; best = None
    started = perf_counter()
    interval = max(1, args.steps // 5)
    for step in range(1, args.steps + 1):
        model.train()
        anchor_step = bool(args.anchor_every and step % args.anchor_every == 0)
        pool = anchor_prepared if anchor_step else prepared
        mode = "full" if anchor_step else args.loss
        batch = (sample_paired_batch(pairs, args.batch_size, rng) if args.paired_batches and not anchor_step
                 else [pool[rng.randrange(len(pool))] for _ in range(args.batch_size)])
        inputs, lengths = pad_tokens([tokens for _, tokens in batch], pad_id, context_limit(args.model))
        targets = mx.array([LABELS.index(case["answer"]) for case, _ in batch])
        loss, grads = (full_grad if mode == "full" else choice_grad)(model, inputs, lengths, targets)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        entry = {"step": step, "train_loss": float(loss), "elapsed_s": perf_counter() - started,
                 "data_source": "retention_train" if anchor_step else "temporal_train", "loss_mode": mode}
        if step % interval == 0 or step == args.steps:
            records, metrics = evaluate_cases(model, tokenizer, dev, context_limit(args.model), args.eval_batch_size)
            checkpoint = output / f"checkpoint-{step:06d}"
            save_adapter(model, checkpoint, config, args.layers, identity, args.revision, counts, step,
                         input_view=args.input_view)
            write_records(checkpoint / "dev.jsonl", records)
            atomic_json(checkpoint / "dev_metrics.json", metrics)
            retention_records, retention_metrics = evaluate_cases(
                model, tokenizer, retention_dev, context_limit(args.model), args.eval_batch_size)
            write_records(checkpoint / "retention_dev.jsonl", retention_records)
            atomic_json(checkpoint / "retention_dev_metrics.json", retention_metrics)
            entry["dev_accuracy"] = metrics["accuracy"]
            entry["retention_dev_accuracy"] = retention_metrics["accuracy"]
            entry["checkpoint"] = checkpoint.name
            if best is None or metrics["accuracy"] > best["dev_accuracy"]:
                best = dict(entry)
            print(f"step {step}/{args.steps} loss={float(loss):.4f} "
                  f"dev={metrics['accuracy']:.3f} retention={retention_metrics['accuracy']:.3f} "
                  f"checkpoint={checkpoint.name}", file=sys.stderr, flush=True)
        logs.append(entry)
        atomic_json(output / "training_log.json", logs)
    atomic_json(output / "selection.json", {"criterion": "highest dev accuracy; earliest on tie", "selected": best,
                                            "status": "suggestion; review retention and generation separately",
                                            "training_wall_s": perf_counter() - started, "parameters": counts,
                                            "seed": args.seed, "args": vars(args)})
    # A fresh model verifies both strict checkpoint loading and score parity.
    selected = output / best["checkpoint"]
    selected_weights = mx.load(str(selected / "adapters.safetensors"))
    model.load_weights(list(selected_weights.items()), strict=False)
    model.eval()
    sample, sample_pad = prepare(dev[:min(4, len(dev))], tokenizer, context_limit(args.model))
    original, _ = score_batch(model, sample, sample_pad, context_limit(args.model), ids)
    mx.eval(original)
    fresh, _ = load(args.model)
    load_adapter(fresh, selected, identity, args.revision)
    fresh.eval()
    restored, _ = score_batch(fresh, sample, sample_pad, context_limit(args.model), ids)
    mx.eval(restored)
    diff = float(np.max(np.abs(np.asarray(original) - np.asarray(restored))))
    atomic_json(output / "reload_parity.json", {"max_absolute_logit_difference": diff, "tolerance": 1e-5,
                                                  "passed": diff <= 1e-5})
    if diff > 1e-5:
        raise ValueError("adapter reload parity failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "evaluate"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Resolved upstream revision, supplied by the caller")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--pairs-per-family", type=int, default=32)
    parser.add_argument("--dev-pairs-per-family", type=int, default=32)
    parser.add_argument("--loss", choices=["choice", "full"], default="choice")
    parser.add_argument("--anchor-every", type=int, default=0,
                        help="Use a training-retention full-vocabulary update every N steps; zero disables")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Shuffle temporal training labels once, using the seed; dev and test remain true")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use train-only time-scale and offset augmentation")
    parser.add_argument("--paired-batches", action="store_true",
                        help="Sample whole counterfactual pairs; requires an even batch size")
    parser.add_argument("--input-view", choices=["raw", "clock-features"], default="raw")
    args = parser.parse_args()
    if (min(args.steps, args.batch_size, args.eval_batch_size, args.pairs_per_family,
            args.dev_pairs_per_family) <= 0 or args.learning_rate <= 0 or args.anchor_every < 0):
        parser.error("steps, batch sizes, pairs and learning rate must be positive")
    if args.command == "train" and args.adapter:
        parser.error("training starts from the pinned base, not an existing adapter")
    if args.shuffle_labels and args.command != "train":
        parser.error("--shuffle-labels is only valid for training")
    if (args.curriculum or args.paired_batches) and args.command != "train":
        parser.error("--curriculum and --paired-batches are only valid for training")
    if args.paired_batches and args.batch_size % 2:
        parser.error("--paired-batches requires an even batch size")
    output = Path(args.output)
    reserve_output(output)
    identity = base_identity(args.model)
    start = perf_counter()
    model, tokenizer = load(args.model)
    counts = load_adapter(model, args.adapter, identity, args.revision) if args.adapter else None
    atomic_json(output / "run.json", {"args": vars(args), "base_hashes": identity,
                                      "revision_provenance": "caller supplied", "load_s": perf_counter() - start,
                                      "runtime": runtime_provenance(),
                                      "checkpoint_stored_tensor_elements": checkpoint_parameter_count(args.model),
                                      "loaded_model_parameters": get_total_parameters(model),
                                      "adapter_parameters": counts})
    if args.command == "train":
        run_train(args, model, tokenizer, identity)
    elif args.command == "inspect":
        inspect_model(model, tokenizer, output)
    else:
        from .surgery_data import build_cases, build_retention_cases, deterministic_answer
        cases = build_cases(args.split, pairs_per_family=args.pairs_per_family, seed=args.seed)
        require_split(cases, args.split)
        cases = apply_input_view(cases, args.input_view)
        save_dataset(output, f"temporal_{args.split}", cases)
        records, metrics = evaluate_cases(model, tokenizer, cases, context_limit(args.model), args.batch_size)
        metrics["deterministic_reference_accuracy"] = sum(
            deterministic_answer(case) == case["answer"] for case in cases) / len(cases)
        write_records(output / "cases.jsonl", records)
        atomic_json(output / "metrics.json", metrics)
        retention = build_retention_cases(split=args.split)
        require_split(retention, args.split)
        save_dataset(output, f"retention_{args.split}", retention)
        records, metrics = evaluate_cases(model, tokenizer, retention, context_limit(args.model), args.batch_size)
        write_records(output / "retention.jsonl", records)
        atomic_json(output / "retention_metrics.json", metrics)


if __name__ == "__main__":
    main()
