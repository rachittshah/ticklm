"""Query-independent temporal representations and inexpensive forecast controls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _history(values, step_hours=1.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("History must contain at least two finite scalar observations")
    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("Sampling interval must be finite and positive; irregular data need resampling")
    return x


def _number(value: float) -> str:
    return f"{float(value):.4g}"


def _clock(value: float) -> str:
    # Round-trip the binary float: rounded timestamps invalidate an error certificate.
    return repr(float(value)).removesuffix(".0")


def knot_indices(values: np.ndarray, count: int, method: str = "adaptive") -> list[int]:
    """Choose joint time indices; adaptive error is vertical, not geometric distance."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or count < 2:
        raise ValueError("Knots require finite one/two-dimensional curves and count >= 2")
    count = min(count, len(x))
    if method == "uniform":
        return np.unique(np.rint(np.linspace(0, len(x) - 1, count)).astype(int)).tolist()
    if method != "adaptive":
        raise ValueError(f"Unknown codec: {method}")
    chosen = [0, len(x) - 1]
    axis = np.arange(len(x))
    while len(chosen) < count:
        reconstruction = np.stack(
            [np.interp(axis, chosen, x[chosen, j]) for j in range(x.shape[1])], axis=1
        )
        errors = np.max(np.abs(x - reconstruction), axis=1)
        errors[chosen] = -1
        next_index = int(np.argmax(errors))
        if errors[next_index] <= 1e-12:
            break
        chosen.append(next_index)
        chosen.sort()
    return chosen


def interpolation_error(values, indices, *, rounded: bool = False, times=None) -> float:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    selected = x[indices].copy()
    if rounded:
        selected = np.vectorize(lambda v: float(_number(v)))(selected)
    axis = np.arange(len(x)) if times is None else np.asarray(times, dtype=float)
    reconstruction = np.stack(
        [np.interp(axis, axis[indices], selected[:, j]) for j in range(x.shape[1])],
        axis=1,
    )
    return float(np.max(np.abs(x - reconstruction)))


@dataclass
class Evidence:
    text: str
    evidence_tokens: int
    history_error: float | None
    forecast_error: float | None = None
    history_knots: int = 0
    forecast_knots: int = 0
    quantile_crossings: int = 0
    elapsed_s: float = 0.0


def _error_text(value: float) -> str:
    # Round upward so the printed certificate remains an upper bound.
    if value == 0:
        return "0"
    exponent = np.floor(np.log10(value))
    unit = 10 ** (exponent - 2)
    return _number(np.ceil(value / unit) * unit)


def compile_evidence(
    history,
    step_hours: float,
    forecast: np.ndarray | None = None,
    *,
    codec: str = "adaptive",
    budget: int = 384,
    count_tokens: Callable[[str], int] = len,
    forecast_source: str = "model marginal quantiles",
) -> Evidence:
    """Compile only observations/projections; no question or answer enters this API.

    Forecast columns are q10, median, q90. The measured certificate bounds numeric
    interpolation distortion at discrete samples, not the forecast's accuracy.
    """
    start = time.perf_counter()
    x = _history(history, step_hours)
    f = None if forecast is None else np.asarray(forecast, dtype=float)
    crossings = 0
    if f is not None:
        if f.ndim != 2 or f.shape[1] != 3 or len(f) < 2 or not np.isfinite(f).all():
            raise ValueError("Forecast must have finite (horizon>=2, 3) q10/median/q90 values")
        crossings = int(np.any(np.diff(f, axis=1) < 0, axis=1).sum())
        # Repair envelopes while preserving the actual point/median prediction.
        f = np.stack([f.min(axis=1), f[:, 1], f.max(axis=1)], axis=1)
    history_times = (np.arange(len(x)) - len(x) + 1) * step_hours
    best = None
    for resolution in range(1, 33):
        hi = knot_indices(x, min(len(x), 2 * resolution + 2), codec)
        he = interpolation_error(x, hi, rounded=True, times=history_times)
        text = (
            f"Now=0 hours; sample_interval={_clock(step_hours)} hours. "
            "Negative time=past. Observed knots time:value; interpolate linearly.\n"
            + " ".join(f"{_clock(history_times[i])}:{_number(x[i])}" for i in hi)
            + f"\nHistory max interpolation error<={_error_text(he)}."
        )
        fi, fe = [], None
        if f is not None:
            fi = knot_indices(f, min(len(f), resolution + 2), codec)
            forecast_times = (np.arange(len(f)) + 1) * step_hours
            fe = interpolation_error(f, fi, rounded=True, times=forecast_times)
            text += (
                f"\nProjected knots hours:lower/point/upper ({forecast_source}); "
                "not path probabilities.\n"
                + " ".join(
                    f"{_clock(forecast_times[i])}:"
                    + "/".join(_number(v) for v in f[i])
                    for i in fi
                )
                + f"\nForecast max interpolation error<={_error_text(fe)}; projections may be wrong."
            )
        tokens = count_tokens(text)
        if tokens <= budget:
            candidate = Evidence(text, tokens, he, fe, len(hi), len(fi), crossings)
            # More knots can cost fewer tokens after rounding the error scalar.
            if best is None or (he + (fe or 0)) < (best.history_error + (best.forecast_error or 0)):
                best = candidate
    if best is None:
        raise ValueError(f"Evidence budget {budget} is too small for clock, endpoints, and provenance")
    best.elapsed_s = time.perf_counter() - start
    return best


def render_raw(history, step_hours: float, count_tokens: Callable[[str], int] = len) -> Evidence:
    start = time.perf_counter()
    x = _history(history, step_hours)
    text = (
        f"Observed values, oldest to newest; sample_interval={_clock(step_hours)} hours; "
        f"last sample is now=0 hours, first is {_clock(-(len(x)-1)*step_hours)} hours.\n"
        + ",".join(_number(v) for v in x)
    )
    error = float(np.max(np.abs(x - np.array([float(_number(v)) for v in x]))))
    return Evidence(text, count_tokens(text), error, history_knots=len(x),
                    elapsed_s=time.perf_counter() - start)


def render_stats(history, step_hours: float, *, budget: int,
                 count_tokens: Callable[[str], int] = len) -> Evidence:
    """Chronologically ordered equal-width aggregates, without answer-specific facts."""
    start = time.perf_counter()
    x = _history(history, step_hours)
    best = None
    for count in range(2, min(16, len(x)) + 1):
        bins = np.array_split(np.arange(len(x)), count)
        text = (f"Now=0 hours; sample_interval={_clock(step_hours)} hours. "
                "Observed time-bin summaries start..end:mean,min,max; negative=past.\n")
        for ix in bins:
            lo, hi = (ix[[0, -1]] - len(x) + 1) * step_hours
            values = x[ix]
            text += (f"{_clock(lo)}..{_clock(hi)}:"
                     f"{_number(values.mean())},{_number(values.min())},{_number(values.max())} ")
        tokens = count_tokens(text)
        if tokens <= budget:
            best = Evidence(text, tokens, None, history_knots=count)
    if best is None:
        raise ValueError("Statistics evidence budget too small")
    best.elapsed_s = time.perf_counter() - start
    return best


def baseline_forecast(history, horizon: int) -> dict:
    """Choose a classical model using a validation tail wholly inside observed history."""
    start = time.perf_counter()
    x = _history(history)
    if len(x) < 48 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Classical selection requires >=48 observations and a positive horizon")
    hold = min(24, len(x) // 4)
    train, validation = x[:-hold], x[-hold:]
    candidates = [("last", 1), ("seasonal", 12), ("seasonal", 24), ("seasonal", 48),
                  ("linear", 16), ("linear", 32), ("linear", 64)]
    candidates = [candidate for candidate in candidates if candidate[1] <= len(train)]

    def predict(series, count, candidate):
        kind, window = candidate
        if kind == "last":
            return np.full(count, series[-1])
        if kind == "seasonal":
            return np.resize(series[-window:], count)
        recent = series[-window:]
        slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)
        return intercept + slope * np.arange(len(recent), len(recent) + count)

    losses = [float(np.mean(np.abs(predict(train, hold, c) - validation))) for c in candidates]
    selected = candidates[int(np.argmin(losses))]
    residuals = validation - predict(train, hold, selected)
    point = predict(x, horizon, selected)
    low, high = np.quantile(residuals, [0.1, 0.9])
    return {"point": point, "q10": point + min(0.0, float(low)),
            "q90": point + max(0.0, float(high)), "model_id": f"{selected[0]}-{selected[1]}",
            "validation_mae": min(losses), "elapsed_s": time.perf_counter() - start}
