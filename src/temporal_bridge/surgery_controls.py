"""Robustness, retention, and frozen-feature probes for weight surgery.

Run as ``python -m temporal_bridge.surgery_controls robustness|retention ...``.
The retention passages are a small smoke test, not a general language benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .surgery import (
    atomic_json,
    base_identity,
    context_limit,
    evaluate_cases,
    last_logits,
    load_adapter,
    pad_tokens,
    prepare,
    require_split,
    reserve_output,
    runtime_provenance,
    save_dataset,
    write_records,
)
from .surgery_data import build_cases, build_robustness_cases, deterministic_answer
from .surgery_features import with_feature_view

_PASSAGES = {
    "dev": (
        "A baker mixes flour, water, yeast, and salt before putting the dough in an oven.",
        "The library has a quiet room where visitors can read books and study together.",
        "A bicycle has two wheels, pedals, a chain, and brakes to help the rider stop.",
        "Plants use sunlight, water, and carbon dioxide to produce sugars as they grow.",
    ),
    "test": (
        "A musician tunes her violin and plays a melody for the audience in the hall.",
        "A river flows through a valley, carrying water and sediment toward the sea.",
        "A chef slices tomatoes and cucumbers before adding them to a bowl of salad.",
        "The museum displays paintings, sculptures, and artifacts from many cultures.",
    ),
}


def retention_passages(split):
    if split not in _PASSAGES:
        raise ValueError("retention split must be dev or test")
    return _PASSAGES[split]


def distribution_sums(base_logits, adapted_logits, targets, mask):
    """Return masked token sums of base CE, adapted CE, KL(base || adapted), and count."""
    base_logits = base_logits.astype(mx.float32)
    adapted_logits = adapted_logits.astype(mx.float32)
    base = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
    adapted = adapted_logits - mx.logsumexp(adapted_logits, axis=-1, keepdims=True)
    rows = mx.arange(targets.shape[0])
    mask = mask.astype(mx.float32)
    base_ce = (-base[rows, targets] * mask).sum()
    adapted_ce = (-adapted[rows, targets] * mask).sum()
    kl = ((mx.exp(base) * (base - adapted)).sum(axis=-1) * mask).sum()
    return base_ce, adapted_ce, kl, mask.sum()


def robustness(model, tokenizer, args):
    limit = context_limit(args.model)
    primary = with_feature_view(
        build_cases(args.split, pairs_per_family=args.pairs_per_family, seed=args.seed), args.input_view)
    require_split(primary, args.split)
    save_dataset(args.output, "robustness_primary", primary)
    records, metrics = evaluate_cases(model, tokenizer, primary, limit, args.batch_size)
    write_records(Path(args.output) / "primary.jsonl", records)
    source = {r["id"]: r for r in records}
    summaries = {"primary": metrics}
    controls = build_robustness_cases(args.split, pairs_per_family=args.pairs_per_family, seed=args.seed)
    for name, cases in controls.items():
        cases = with_feature_view(cases, args.input_view)
        require_split(cases, args.split)
        save_dataset(args.output, f"robustness_{name}", cases)
        records, metrics = evaluate_cases(model, tokenizer, cases, limit, args.batch_size)
        metrics["deterministic_reference_accuracy"] = sum(
            deterministic_answer(c) == c["answer"] for c in cases) / len(cases)
        # Source IDs exist for rotation and translation; in-range examples use a separate seed.
        matched = [(r, source[r["parent_id"]]) for r in records if r["parent_id"] in source]
        if matched:
            metrics["same_action_as_primary"] = sum(
                r["choices"][r["prediction"]] == p["choices"][p["prediction"]]
                for r, p in matched) / len(matched)
            metrics["matched_primary_count"] = len(matched)
        write_records(Path(args.output) / f"{name}.jsonl", records)
        summaries[name] = metrics
    atomic_json(Path(args.output) / "metrics.json", summaries)


def retention(model, tokenizer, args, identity):
    limit = context_limit(args.model)
    texts = retention_passages(args.split)
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2 or len(ids) > limit:
            raise ValueError("retention passage must have 2 or more tokens and fit model context")
        sequences.append(ids)
    model.eval()
    # Evaluate each passage independently and keep its full baseline distributions on CPU.
    baseline = []
    for ids in sequences:
        logits = model(mx.array([ids[:-1]]))[0].astype(mx.float32)
        mx.eval(logits)
        baseline.append(np.array(logits, dtype=np.float32, copy=True))
    adapter_counts = None
    if args.adapter:
        adapter_counts = load_adapter(model, args.adapter, identity, args.revision)
    model.eval()
    sums = np.zeros(4, dtype=np.float64)
    records = []
    for text, ids, before in zip(texts, sequences, baseline, strict=True):
        after = model(mx.array([ids[:-1]]))[0].astype(mx.float32) if args.adapter else mx.array(before)
        values = distribution_sums(mx.array(before), after, mx.array(ids[1:]),
                                   mx.ones((len(ids) - 1,), dtype=mx.bool_))
        mx.eval(*values)
        case = np.array([float(v) for v in values], dtype=np.float64)
        if not np.isfinite(case).all() or case[3] <= 0:
            raise ValueError("nonfinite retention metrics or empty token mask")
        sums += case
        records.append({"text": text, "token_ids": ids, "scored_tokens": int(case[3]),
                        "base_cross_entropy": case[0] / case[3],
                        "adapted_cross_entropy": case[1] / case[3],
                        "mean_kl_base_to_adapted": case[2] / case[3]})
    count = int(sums[3])
    base_ce, adapted_ce, kl = (float(x / count) for x in sums[:3])
    summary = {"passages": len(texts), "scored_tokens": count,
               "base_cross_entropy": base_ce, "adapted_cross_entropy": adapted_ce,
               "base_perplexity": math.exp(base_ce), "adapted_perplexity": math.exp(adapted_ce),
               "mean_kl_base_to_adapted": kl,
               "scope": "Tiny fixed non-temporal smoke test; not a broad language benchmark.",
               "token_policy": "No special tokens or padding; predict each token after the first."}
    write_records(Path(args.output) / "passages.jsonl", records)
    atomic_json(Path(args.output) / "metrics.json", summary)
    return adapter_counts


def extract_probe_features(model, tokenizer, cases, limit, batch_size):
    """Cache final normalized hidden states at the same prompt position as scoring."""
    model.eval()
    prepared, pad_id = prepare(cases, tokenizer, limit)
    features = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        tokens, lengths = pad_tokens([tokens for _, tokens in batch], pad_id, limit)
        hidden = last_logits(model.model(tokens), lengths).astype(mx.float32)
        mx.eval(hidden)
        features.append(np.array(hidden, dtype=np.float32, copy=True))
    result = np.concatenate(features, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite hidden features")
    return result


def fit_probe(features, labels, steps=400, learning_rate=0.03, l2=0.01):
    """Fit a four-class linear head; normalize using training features only."""
    x = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(labels) or not len(labels) or np.any((labels < 0) | (labels > 3)):
        raise ValueError("invalid probe training features or labels")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    x = (x - mean) / scale
    weight = np.zeros((x.shape[1], 4), dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    mw, vw = np.zeros_like(weight), np.zeros_like(weight)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    for step in range(1, steps + 1):
        logits = x @ weight + bias
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(labels)), labels] -= 1
        probs /= len(labels)
        dw = x.T @ probs + l2 * weight
        db = probs.sum(axis=0)
        mw = .9 * mw + .1 * dw
        vw = .999 * vw + .001 * dw * dw
        mb = .9 * mb + .1 * db
        vb = .999 * vb + .001 * db * db
        weight -= learning_rate * (mw / (1 - .9 ** step)) / (np.sqrt(vw / (1 - .999 ** step)) + 1e-8)
        bias -= learning_rate * (mb / (1 - .9 ** step)) / (np.sqrt(vb / (1 - .999 ** step)) + 1e-8)
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias}


def probe_predict(head, features):
    return ((features - head["mean"]) / head["scale"]) @ head["weight"] + head["bias"]


def atomic_npz(path, values):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **values)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def probe(model, tokenizer, args, identity):
    output = Path(args.output)
    limit = context_limit(args.model)
    model.freeze()
    data_hash = hashlib.sha256(Path(__file__).with_name("surgery_data.py").read_bytes()).hexdigest()
    if args.probe_head:
        head_path = Path(args.probe_head)
        manifest = json.loads(head_path.with_name("head_manifest.json").read_text())
        if (manifest["base_hashes"] != identity or manifest["revision"] != args.revision
                or manifest["data_source_sha256"] != data_hash or manifest["seed"] != args.seed):
            raise ValueError("probe head base, revision, data code, or seed does not match")
        if hashlib.sha256(head_path.read_bytes()).hexdigest() != manifest["head_sha256"]:
            raise ValueError("probe head hash does not match manifest")
        with np.load(head_path, allow_pickle=False) as saved:
            head = {k: saved[k] for k in ("mean", "scale", "weight", "bias")}
    else:
        train = build_cases("train", pairs_per_family=args.train_pairs_per_family, seed=args.seed)
        require_split(train, "train")
        save_dataset(output, "probe_train", train)
        train_features = extract_probe_features(model, tokenizer, train, limit, args.batch_size)
        atomic_npz(output / "train_features.npz", {"features": train_features})
        labels = np.array(["ABCD".index(c["answer"]) for c in train])
        head = fit_probe(train_features, labels)
        atomic_npz(output / "head.npz", head)
        manifest = {"base_hashes": identity, "revision": args.revision, "seed": args.seed,
                    "data_source_sha256": data_hash, "training_split": "train",
                    "train_pairs_per_family": args.train_pairs_per_family,
                    "training_cases": len(train), "steps": 400, "learning_rate": .03,
                    "l2": .01, "selection": "one fixed setting, evaluated on dev",
                    "head_sha256": hashlib.sha256((output / "head.npz").read_bytes()).hexdigest()}
        atomic_json(output / "head_manifest.json", manifest)
    cases = build_cases(args.split, pairs_per_family=args.pairs_per_family, seed=args.seed)
    require_split(cases, args.split)
    save_dataset(output, f"probe_{args.split}", cases)
    features = extract_probe_features(model, tokenizer, cases, limit, args.batch_size)
    atomic_npz(output / f"{args.split}_features.npz", {"features": features})
    predictions = probe_predict(head, features).argmax(axis=1)
    records = []
    pairs = {}
    families = {}
    for case, pred in zip(cases, predictions, strict=True):
        prediction = "ABCD"[int(pred)]
        record = {**case, "prediction": prediction, "correct": prediction == case["answer"]}
        records.append(record)
        pairs.setdefault(case["pair_id"], []).append(record["correct"])
        families.setdefault(case["family"], []).append(record["correct"])
    complete = [p for p in pairs.values() if len(p) == 2]
    metrics = {"count": len(records), "accuracy": sum(r["correct"] for r in records) / len(records),
               "pair_both_correct": sum(all(p) for p in complete) / len(complete) if complete else None,
               "complete_pairs": len(complete),
               "per_family_accuracy": {k: sum(v) / len(v) for k, v in families.items()},
               "head": manifest,
               "interpretation": "Linear readout of frozen final prompt features; failure does not disprove latent capability."}
    write_records(output / "cases.jsonl", records)
    atomic_json(output / "metrics.json", metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["robustness", "retention", "probe"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Caller-supplied resolved base revision")
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pairs-per-family", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--input-view", choices=["raw", "clock-features"], default="raw")
    parser.add_argument("--train-pairs-per-family", type=int, default=128)
    parser.add_argument("--probe-head", help="Previously saved head.npz; required for probe test")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.pairs_per_family <= 0 or args.train_pairs_per_family <= 0:
        parser.error("batch size and pairs per family must be positive")
    if args.command == "probe" and args.adapter:
        parser.error("probe requires the frozen base without an adapter")
    if args.command == "probe" and ((args.split == "test") != bool(args.probe_head)):
        parser.error("probe test requires --probe-head; dev trains a new head")
    if args.command != "probe" and args.probe_head:
        parser.error("--probe-head applies only to probe")
    if args.command != "robustness" and args.input_view != "raw":
        parser.error("clock-features input is supported only for robustness")
    started = perf_counter()
    reserve_output(args.output)
    identity = base_identity(args.model)
    model, tokenizer = load(args.model)
    adapter_counts = None
    if args.command == "robustness" and args.adapter:
        adapter_counts = load_adapter(model, args.adapter, identity, args.revision)
    provenance = runtime_provenance()
    provenance["source_sha256"]["surgery_controls.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adapter_hashes = None
    if args.adapter:
        adapter_hashes = {}
        for name in ("manifest.json", "adapter_config.json", "adapters.safetensors"):
            digest = hashlib.sha256()
            with (Path(args.adapter) / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            adapter_hashes[name] = digest.hexdigest()
    atomic_json(Path(args.output) / "run.json", {"args": vars(args), "base_hashes": identity,
                "revision_provenance": "caller supplied", "runtime": provenance,
                "adapter_parameters": adapter_counts, "adapter_hashes": adapter_hashes})
    if args.command == "robustness":
        robustness(model, tokenizer, args)
    elif args.command == "retention":
        adapter_counts = retention(model, tokenizer, args, identity)
        atomic_json(Path(args.output) / "adapter_parameters.json", adapter_counts)
    else:
        probe(model, tokenizer, args, identity)
    atomic_json(Path(args.output) / "timing.json", {
        "total_wall_s": perf_counter() - started,
        "mlx_process_peak_memory_bytes": mx.get_peak_memory(),
        "memory_scope": "MLX process high-water mark, including model load and all controls",
    })


if __name__ == "__main__":
    main()
