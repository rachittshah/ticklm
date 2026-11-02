"""Reproducible development search, frozen held-out evaluation, and local demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from temporal_bridge.backends import TIMESFM_REVISION, MLXLanguageModel, TimesFMBackend
from temporal_bridge.benchmark import Case, build_cases, direct_answer
from temporal_bridge.core import (
    Evidence,
    baseline_forecast,
    compile_evidence,
    render_raw,
    render_stats,
)

DEFAULT_MODEL = "/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit"
CANDIDATES = [(codec, budget) for codec in ("uniform", "adaptive") for budget in (256, 384, 512)]
ARMS = ("question_only", "raw", "uniform", "stats", "cheap", "timesfm", "mismatched")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value):
    """Artifacts use explicit paths and exclusive creation to prevent silent reruns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=_json_value, allow_nan=False)
        stream.write("\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes():
    return {path.name: _hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def runtime_identity(model_path):
    model = Path(model_path)
    tsfm = Path(importlib.util.find_spec("mlx_tsfm").origin).parent
    return {
        "model_files": {p.name: _hash(p) for p in sorted(model.iterdir())
                        if p.suffix in {".safetensors", ".json", ".txt", ".model"}},
        "tsfm_source": {str(p.relative_to(tsfm)): _hash(p) for p in sorted(tsfm.rglob("*.py"))},
        "checkpoint": TIMESFM_REVISION,
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


def validate_frozen_identity(frozen, current):
    if frozen != current:
        raise ValueError("Runtime identity changed after development freeze; start a new dev run")


def _series_key(case: Case) -> str:
    return hashlib.sha256(case.context.astype("<f4").tobytes()).hexdigest()


def make_prompt(case: Case, evidence: str) -> str:
    """Only public task fields enter the LLM prompt; labels stay in the scorer."""
    public = case.public_view()
    options = "\n".join(f"{letter}) {value}" for letter, value in zip("ABCD", public["options"]))
    return f"Temporal evidence:\n{evidence}\n\nQuestion: {public['question']}\n{options}"


def _language(path: str):
    language = MLXLanguageModel(path)
    language._load()
    language.score("Which number is even? A) 1 B) 2 C) 3 D) 5")
    return language


def _forecast_cases(cases: list[Case], out: Path, expected_checkpoint_sha=None):
    forecaster = TimesFMBackend()
    by_key = {}
    # The first shape execution is retained separately from subsequent warm calls.
    first_call = None
    for case in cases:
        if case.task != "forecast" or _series_key(case) in by_key:
            continue
        key = _series_key(case)
        ts = forecaster.forecast(case.context[None, :], 24)
        cheap = baseline_forecast(case.context, 24)
        if first_call is None:
            first_call = ts.elapsed_s
        record = {
            "series_key": key, "case_id": case.case_id,
            "mean": float(np.mean(case.context)), "std": float(np.std(case.context)),
            "timesfm": {"point": ts.point[0], "q10": ts.q10[0], "q90": ts.q90[0],
                        "elapsed_s": ts.elapsed_s, "model_id": ts.model_id},
            "cheap": cheap,
        }
        by_key[key] = record
    checkpoint_sha = _hash(Path(forecaster.checkpoint_path))
    if expected_checkpoint_sha is not None and checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("TimesFM checkpoint weights changed after development freeze")
    timings = [r["timesfm"]["elapsed_s"] for r in by_key.values()][1:]
    metadata = {"load_s": forecaster.load_s, "checkpoint": forecaster.checkpoint_path,
                "checkpoint_sha256": checkpoint_sha, "revision": forecaster.revision,
                "first_call_s": first_call, "warm_calls": len(timings),
                "warm_p50_ms": float(np.median(timings) * 1000) if timings else None,
                "warm_p95_ms": float(np.quantile(timings, .95) * 1000) if timings else None}
    _write_json(out, {"metadata": metadata, "forecasts": list(by_key.values())})
    return by_key, metadata


def _forecast_matrix(record, arm):
    values = record[arm]
    return np.stack([values["q10"], values["point"], values["q90"]], axis=1)


def make_evidence(case, arm, codec, budget, count_tokens, forecasts):
    if arm == "question_only":
        return Evidence("No numeric observations supplied.", count_tokens("No numeric observations supplied."), None), 0.0, None
    if arm == "raw":
        return render_raw(case.context, case.step_hours, count_tokens), 0.0, None
    if arm == "stats":
        return render_stats(case.context, case.step_hours, budget=budget,
                            count_tokens=count_tokens), 0.0, None
    if arm == "uniform":
        return compile_evidence(case.context, case.step_hours, codec="uniform", budget=budget,
                                count_tokens=count_tokens), 0.0, None
    forecast, cost, donor = None, 0.0, None
    if case.task == "forecast":
        current = forecasts[_series_key(case)]
        source = current
        forecast_arm = "cheap" if arm == "cheap" else "timesfm"
        if arm == "mismatched":
            keys = sorted(forecasts)
            donor_key = keys[(keys.index(_series_key(case)) + 1) % len(keys)]
            if donor_key == _series_key(case):
                raise ValueError("Mismatched control requires two distinct histories")
            source = forecasts[donor_key]
            donor = source["case_id"]
        forecast = _forecast_matrix(source, forecast_arm)
        if donor is not None:
            # Match units and broad scale so the control does not fail only by being off-scale.
            forecast = ((forecast - source["mean"]) / max(source["std"], 1e-8)
                        * max(current["std"], 1e-8) + current["mean"])
        cost = current[forecast_arm]["elapsed_s"]
    provenance = ("empirical residual 10/90 bands" if arm == "cheap"
                  else "model marginal 10/50/90 quantiles")
    evidence = compile_evidence(case.context, case.step_hours, forecast, codec=codec,
                                budget=budget, count_tokens=count_tokens, forecast_source=provenance)
    return evidence, cost, donor


def _score_schedule(cases, configs, language, forecasts, output, seed):
    count_tokens = lambda text: len(language.tokenizer.encode(text, add_special_tokens=False))
    schedule = [(case, config) for case in cases for config in configs]
    np.random.default_rng(seed).shuffle(schedule)
    rows = []
    with output.open("x") as stream:
        for index, (case, (arm, codec, budget, config_id)) in enumerate(schedule):
            row_start = time.perf_counter()
            evidence, forecast_s, donor = make_evidence(
                case, arm, codec, budget, count_tokens, forecasts
            )
            prompt = make_prompt(case, evidence.text)
            result = language.score(prompt)
            online_without_forecast_s = time.perf_counter() - row_start
            row = {
                "case_id": case.case_id, "series_id": case.metadata["series_id"],
                "source": case.metadata["source"], "task": case.task, "family": case.family,
                "intervention": case.metadata.get("intervention"),
                "arm": arm, "config_id": config_id, "codec": codec, "budget": budget,
                "answer": "ABCD"[case.answer], "answer_text": case.options[case.answer],
                "choice_text": case.options["ABCD".index(result["choice"])],
                "correct": result["choice"] == "ABCD"[case.answer], **result,
                "evidence_tokens": evidence.evidence_tokens,
                "history_error": evidence.history_error, "forecast_error": evidence.forecast_error,
                "history_knots": evidence.history_knots, "forecast_knots": evidence.forecast_knots,
                "quantile_crossings": evidence.quantile_crossings,
                "codec_s": evidence.elapsed_s, "forecast_s": forecast_s,
                "measured_cached_forecast_pipeline_s": online_without_forecast_s,
                "estimated_uncached_pipeline_s": online_without_forecast_s + forecast_s,
                "mismatch_donor_id": donor, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            rows.append(row)
            stream.write(json.dumps(row, default=_json_value, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 24 == 0 or index + 1 == len(schedule):
                print(json.dumps({"scored": index + 1, "total": len(schedule),
                                  "last_config": config_id}), flush=True)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for task in ("all", row["task"]):
            groups[(row["config_id"], task)].append(row)
    summary = []
    for (config_id, task), group in sorted(groups.items()):
        summary.append({
            "config_id": config_id, "task": task, "n": len(group),
            "correct": sum(r["correct"] for r in group),
            "accuracy": float(np.mean([r["correct"] for r in group])),
            "mean_input_tokens": float(np.mean([r["input_tokens"] for r in group])),
            "mean_evidence_tokens": float(np.mean([r["evidence_tokens"] for r in group])),
            "llm_p50_ms": float(np.median([r["elapsed_s"] for r in group]) * 1000),
            "llm_p95_ms": float(np.quantile([r["elapsed_s"] for r in group], .95) * 1000),
            "codec_p50_ms": float(np.median([r["codec_s"] for r in group]) * 1000),
            "measured_cached_pipeline_p50_ms": float(np.median([r["measured_cached_forecast_pipeline_s"] for r in group]) * 1000),
            "estimated_uncached_pipeline_p50_ms": float(np.median([r["estimated_uncached_pipeline_s"] for r in group]) * 1000),
            "estimated_uncached_pipeline_p95_ms": float(np.quantile([r["estimated_uncached_pipeline_s"] for r in group], .95) * 1000),
            "unrestricted_letter_agreement": float(np.mean([
                r["unrestricted_top_token"].strip() == r["choice"] for r in group])),
        })
    return summary


def paired_difference(rows, left, right, task, seed=1102):
    lhs = {r["case_id"]: r for r in rows if r["config_id"] == left and r["task"] == task}
    rhs = {r["case_id"]: r for r in rows if r["config_id"] == right and r["task"] == task}
    if lhs.keys() != rhs.keys() or not lhs:
        raise ValueError("Paired comparison requires identical nonempty case IDs")
    groups = defaultdict(list)
    for key, row in lhs.items():
        groups[row["series_id"]].append(int(row["correct"]) - int(rhs[key]["correct"]))
    values = list(groups.values())
    sums = np.array([sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    sampled = np.random.default_rng(seed).integers(0, len(values), (4000, len(values)))
    bootstrap = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {"left": left, "right": right, "task": task, "n": len(lhs), "clusters": len(values),
            "difference": float(sums.sum() / counts.sum()),
            "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "left_only_correct": sum(lhs[k]["correct"] and not rhs[k]["correct"] for k in lhs),
            "right_only_correct": sum(rhs[k]["correct"] and not lhs[k]["correct"] for k in lhs)}


def _direct_metrics(cases, forecasts):
    rows = []
    for case in cases:
        if case.task != "forecast":
            continue
        for arm in ("cheap", "timesfm"):
            forecast = forecasts[_series_key(case)][arm]
            values = np.asarray(forecast["point"])
            future = case.future
            low = np.minimum(forecast["q10"], forecast["q90"])
            high = np.maximum(forecast["q10"], forecast["q90"])
            rows.append({"case_id": case.case_id, "arm": arm,
                         "correct": direct_answer(case, values) == case.answer,
                         "choice": "ABCD"[direct_answer(case, values)],
                         "mae": float(np.mean(np.abs(values - future))),
                         "scaled_mae": float(np.mean(np.abs(values - future)) / max(np.std(case.context), 1e-8)),
                         "pointwise_80pct_coverage": float(np.mean((future >= low) & (future <= high))),
                         "elapsed_s": forecast["elapsed_s"]})
    return rows


def _manifest(args, cases):
    model = Path(args.model)
    return {
        "created_unix_s": time.time(), "platform": platform.platform(),
        "hardware": subprocess.check_output(["sysctl", "-n", "hw.model", "hw.memsize"], text=True).splitlines(),
        "source_sha256": _code_hashes(), "dataset": str(Path(args.data).resolve()),
        "dataset_sha256": _hash(Path(args.data)),
        "dataset_url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "model_path": str(model),
        "model_file_sha256": {p.name: _hash(p) for p in sorted(model.glob("*.safetensors"))},
        "model_config_sha256": _hash(model / "config.json"),
        "case_ids": [c.case_id for c in cases],
        "answer_counts": dict(Counter("ABCD"[c.answer] for c in cases)),
        "candidate_grid": CANDIDATES,
        "selection_rule": "highest total development accuracy, tie: lowest mean input tokens, then config ID",
        "dependencies": subprocess.check_output(["uv", "pip", "freeze", "--python", str(Path(args.python))], text=True).splitlines(),
        "scoring": "conditional single-token A-D likelihood; no thinking, no model training",
    }


def run_dev(args):
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases("dev", args.data)
    if args.limit:
        cases = cases[:args.limit] + [c for c in cases if c.task == "forecast"][:args.limit]
    identity = runtime_identity(args.model)
    _write_json(out / "dev-manifest.json", {**_manifest(args, cases), "runtime_identity": identity})
    _write_json(out / "dev-cases.json", [asdict(c) for c in cases])
    forecasts, forecast_metadata = _forecast_cases(cases, out / "dev-forecasts.json")
    language = _language(args.model)
    configs = [("timesfm", codec, budget, f"{codec}-{budget}") for codec, budget in CANDIDATES]
    configs += [("raw", "none", 0, "raw"), ("stats", "none", 384, "stats"),
                ("question_only", "none", 0, "question_only")]
    rows = _score_schedule(cases, configs, language, forecasts, out / "dev-records.jsonl", 712)
    summary = summarize(rows)
    ranked = sorted([r for r in summary if r["task"] == "all" and r["config_id"] not in
                     {"raw", "stats", "question_only"}],
                    key=lambda r: (-r["accuracy"], r["mean_input_tokens"], r["config_id"]))
    chosen = ranked[0]
    codec, budget = chosen["config_id"].split("-")
    selection = {"codec": codec, "budget": int(budget), "selected_dev_metrics": chosen,
                 "created_unix_s": time.time(), "source_sha256": _code_hashes(),
                 "dataset_sha256": _hash(Path(args.data)), "model_path": args.model,
                 "runtime_identity": identity,
                 "timesfm_checkpoint_sha256": forecast_metadata["checkpoint_sha256"],
                 "dev_case_ids": [c.case_id for c in cases], "is_smoke_subset": bool(args.limit)}
    _write_json(out / "dev-summary.json", {"summary": summary, "forecast": forecast_metadata,
                                           "llm_load_s": language.load_s})
    _write_json(out / "selection.json", selection)
    print(json.dumps({"selected": selection, "summary": summary}, indent=2), flush=True)


def run_test(args):
    out = Path(args.run_dir)
    selection_path = out / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["is_smoke_subset"]:
        raise ValueError("A smoke subset cannot authorize held-out evaluation")
    if selection["source_sha256"] != _code_hashes():
        raise ValueError("Source changed after development freeze; retain old run and run new development")
    if selection["dataset_sha256"] != _hash(Path(args.data)) or selection["model_path"] != args.model:
        raise ValueError("Model or dataset differs from frozen development selection")
    validate_frozen_identity(selection["runtime_identity"], runtime_identity(args.model))
    cases = build_cases("heldout", args.data)
    _write_json(out / "heldout-manifest.json", {**_manifest(args, cases),
                                               "selection_sha256": _hash(selection_path)})
    _write_json(out / "heldout-cases.json", [asdict(c) for c in cases])
    forecasts, forecast_metadata = _forecast_cases(cases, out / "heldout-forecasts.json",
                                                   selection["timesfm_checkpoint_sha256"])
    language = _language(args.model)
    configs = [(arm, selection["codec"], selection["budget"], arm) for arm in ARMS]
    rows = _score_schedule(cases, configs, language, forecasts, out / "heldout-records.jsonl", 719)
    differences = [paired_difference(rows, "timesfm", other, task)
                   for task in ("observed", "forecast")
                   for other in ("raw", "uniform", "stats", "cheap", "mismatched", "question_only")]
    direct = _direct_metrics(cases, forecasts)
    pairs = []
    for arm in ARMS:
        observed = [r for r in rows if r["arm"] == arm and r["task"] == "observed"]
        grouped = defaultdict(list)
        for row in observed:
            grouped[row["series_id"]].append(row)
        pairs.append({"arm": arm, "pairs": len(grouped),
                      "both_correct": sum(all(r["correct"] for r in group) for group in grouped.values())})
    import mlx.core as mx

    result = {"summary": summarize(rows), "paired_differences": differences,
              "direct_forecaster": direct, "counterfactual_pairs": pairs,
              "forecast_metadata": forecast_metadata, "llm_load_s": language.load_s,
              "mlx_peak_memory_bytes": mx.get_peak_memory(),
              "selection_sha256": _hash(selection_path)}
    result["limits"] = [
        "Forecast intervals bootstrap guarded windows from one ETTh1-OT source; exploratory, not cross-dataset inference.",
        "Uncached pipeline latency is a sum using separately measured forecast calls, including a separately identified first call.",
        "Forecasts can change adaptive history knot allocation; TimesFM-cheap measures the whole evidence pipeline.",
        "Quantile coverage is pointwise and does not establish event/path probabilities.",
    ]
    _write_json(out / "heldout-summary.json", result)
    print(json.dumps(result, indent=2, default=_json_value), flush=True)


def run_demo(args):
    out = Path(args.run_dir)
    selection = json.loads((out / "selection.json").read_text())
    cases = build_cases("heldout", args.data)
    forecasts_json = json.loads((out / "heldout-forecasts.json").read_text())
    forecasts = {f["series_key"]: f for f in forecasts_json["forecasts"]}
    language = _language(args.model)
    count_tokens = lambda text: len(language.tokenizer.encode(text, add_special_tokens=False))
    chosen = [next(c for c in cases if c.task == task) for task in ("observed", "forecast")]
    outputs = []
    for case in chosen:
        evidence, _, _ = make_evidence(case, "timesfm", selection["codec"], selection["budget"],
                                       count_tokens, forecasts)
        prompt = make_prompt(case, evidence.text) + "\nGive the option and a short explanation. Distinguish observations from predictions."
        result = language.generate(prompt, max_tokens=160)
        outputs.append({"case_id": case.case_id, "prompt": prompt,
                        "correct_option": "ABCD"[case.answer], "correct_text": case.options[case.answer],
                        **result})
    _write_json(out / "demo.json", outputs)
    print(json.dumps(outputs, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "test", "demo"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="data/ETTh1.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--limit", type=int, help="Smoke-only dev subset; cannot freeze a final test")
    args = parser.parse_args()
    {"dev": run_dev, "test": run_test, "demo": run_demo}[args.stage](args)


if __name__ == "__main__":
    main()
