"""Ask a local frozen language model about your own regularly sampled measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temporal_bridge.backends import MLXLanguageModel, TimesFMBackend
from temporal_bridge.core import baseline_forecast, compile_evidence
from temporal_bridge.experiment import DEFAULT_MODEL


def read_series(path: str | Path) -> tuple[np.ndarray, float]:
    """Read explicit regular samples; reject timestamps rather than silently discard them."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "values" not in payload or "step_hours" not in payload:
        raise ValueError("Input JSON needs values and step_hours")
    if set(payload) != {"values", "step_hours"}:
        raise ValueError("Only values and step_hours are accepted; resample explicit timestamps first")
    values = np.asarray(payload["values"], dtype=np.float64)
    interval = float(payload["step_hours"])
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional series of at least two samples")
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("step_hours must be finite and positive")
    return values, interval


def answer(values, interval, question, *, language, forecaster=None, horizon=24,
           budget=384, codec="uniform", max_tokens=180, cheap=False):
    """Run the real request path; the forecaster and capsule never receive the question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional series with at least two samples")
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive")
    for name, value in (("horizon", horizon), ("budget", budget), ("max_tokens", max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if horizon < 2 or codec not in {"uniform", "adaptive"}:
        raise ValueError("horizon must be >=2 and codec must be uniform or adaptive")
    if cheap and forecaster is not None:
        raise ValueError("cheap and TimesFM forecasts are mutually exclusive")
    if cheap and len(values) < 48:
        raise ValueError("The cheap forecaster needs at least 48 observations")
    started = perf_counter()
    language._load()
    forecast = None
    forecast_s = 0.0
    source = "no forecast"
    provenance = None
    cast_error = None
    if forecaster is not None:
        result = forecaster.forecast(values[None, :], horizon)
        forecast = np.stack([result.q10[0], result.point[0], result.q90[0]], axis=1)
        forecast_s = result.elapsed_s
        source = "model marginal 10/50/90 quantiles"
        cast_error = float(np.max(np.abs(values - values.astype(np.float32).astype(np.float64))))
        provenance = {"repo": forecaster.repo, "revision": forecaster.revision,
                      "checkpoint": forecaster.checkpoint_path}
    elif cheap:
        result = baseline_forecast(values, horizon)
        forecast = np.stack([result["q10"], result["point"], result["q90"]], axis=1)
        forecast_s = result["elapsed_s"]
        source = "empirical residual 10/90 bands"
        provenance = {"model_id": result["model_id"]}
    evidence = compile_evidence(
        values, interval, forecast, codec=codec, budget=budget,
        count_tokens=lambda text: len(language.tokenizer.encode(text, add_special_tokens=False)),
        forecast_source=source,
    )
    prompt = (
        f"Temporal evidence:\n{evidence.text}\n\nQuestion: {question}\n"
        "Use the explicit clock. State whether your answer describes observations or a projection. "
        "If the evidence does not resolve the question, say so. Keep the answer brief."
    )
    generated = language.generate(prompt, max_tokens=max_tokens)
    return {
        "question": question, "answer": generated["text"], "evidence": evidence.text,
        "source": source, "observations": len(values), "step_hours": interval,
        "evidence_tokens": evidence.evidence_tokens, "input_tokens": generated["input_tokens"],
        "observed_interpolation_error": evidence.history_error,
        "projected_interpolation_error": evidence.forecast_error,
        "quantile_crossing_repairs": evidence.quantile_crossings,
        "forecaster": provenance, "timesfm_input_cast_max_error": cast_error,
        "forecast": None if forecast is None else forecast.tolist(),
        "forecast_array_format": "raw lower/point/upper; evidence expands crossed bands while retaining point",
        "timings_s": {"request_wall": perf_counter() - started,
                       "forecast_forward": forecast_s, "codec": evidence.elapsed_s,
                       "language_generation": generated["elapsed_s"]},
        "model_path": language.model_path,
        "timing_scope": "answer() wall clock includes first loads, excludes file IO, process startup and output serialization",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON with values and step_hours")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    forecasting = parser.add_mutually_exclusive_group()
    forecasting.add_argument("--forecast", action="store_true", help="Add a frozen TimesFM projection")
    forecasting.add_argument("--cheap-forecast", action="store_true", help="Use the classical control")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--codec", choices=("uniform", "adaptive"), default="uniform")
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat real requests to distinguish first and warm costs")
    args = parser.parse_args()
    if args.repeats < 1 or args.horizon < 2:
        parser.error("repeats must be positive and horizon must be at least two")
    values, interval = read_series(args.input)
    language = MLXLanguageModel(args.model)
    forecaster = TimesFMBackend() if args.forecast else None
    for i in range(args.repeats):
        result = answer(values, interval, args.question, language=language,
                        forecaster=forecaster, horizon=args.horizon, budget=args.budget,
                        codec=args.codec, max_tokens=args.max_tokens, cheap=args.cheap_forecast)
        print(json.dumps({"request_index": i, **result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
