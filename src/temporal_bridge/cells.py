"""Compile exact, query-independent fixed-time-cell summaries."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .core import Evidence, _clock, _history


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _render_history(x: np.ndarray, step_hours: float, bins: int) -> str:
    chunks = np.array_split(np.arange(len(x)), bins)
    rows = []
    for indices in chunks:
        values = x[indices]
        times = (indices - len(x) + 1) * step_hours
        rows.append(
            f"past [{_clock(times[0])},{_clock(times[-1])}]h n={len(values)} "
            f"mean={_clock(np.mean(values, dtype=np.float64))} "
            f"min={_clock(np.min(values))} max={_clock(np.max(values))} "
            f"positive={int(np.count_nonzero(values > 0))}/{len(values)}"
        )
    return "\n".join(rows)


def _render_forecast(f: np.ndarray, step_hours: float, bins: int) -> str:
    chunks = np.array_split(np.arange(len(f)), bins)
    rows = []
    for indices in chunks:
        values = f[indices]
        times = (indices + 1) * step_hours
        row = (
            f"future [{_clock(times[0])},{_clock(times[-1])}]h n={len(values)} "
            f"point_mean={_clock(np.mean(values[:, 1], dtype=np.float64))} "
            f"point_min={_clock(np.min(values[:, 1]))} "
            f"point_max={_clock(np.max(values[:, 1]))}"
        )
        if f.shape[1] == 3:
            row += (
                f" q10_mean={_clock(np.mean(values[:, 0], dtype=np.float64))}"
                f" q90_mean={_clock(np.mean(values[:, 2], dtype=np.float64))}"
            )
        rows.append(row)
    return "\n".join(rows)


def compile_cells(
    history,
    step_hours: float,
    forecast: np.ndarray | None = None,
    *,
    history_bins: int = 4,
    forecast_bins: int = 4,
    include_history: bool = True,
    count_tokens: Callable[[str], int] = len,
    budget: int = 384,
    forecast_source: str = "model marginal quantiles",
) -> Evidence:
    """Summarize fixed sample intervals without access to a question or its label.

    Forecast columns are the forecaster's q10, point, and q90 values. Quantiles
    remain unchanged; only point values determine each cell's extrema. Marginal
    quantiles describe per-step values, not a probability band for complete paths.
    """
    started = time.perf_counter()
    history_bins = _positive_integer(history_bins, "history_bins")
    forecast_bins = _positive_integer(forecast_bins, "forecast_bins")
    budget = _positive_integer(budget, "budget")
    if not isinstance(include_history, bool):
        raise TypeError("include_history must be a boolean")
    try:
        step = float(step_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sampling interval must be finite and positive") from exc
    x = _history(history, step).astype(np.float64, copy=False)
    if not include_history and forecast is None:
        raise ValueError("At least one history summary or forecast is required")

    f = None if forecast is None else np.asarray(forecast, dtype=np.float64)
    crossings = 0
    if f is not None:
        if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1 or not np.isfinite(f).all():
            raise ValueError("Forecast must have finite (horizon>=1, 3) q10/point/q90 values")
        crossings = int(np.any(np.diff(f, axis=1) < 0, axis=1).sum())

    source = str(forecast_source).replace("\r", " ").replace("\n", " ").strip()
    if f is not None and not source:
        raise ValueError("forecast_source must be a nonempty description")
    used_forecast_bins = min(forecast_bins, len(f)) if f is not None else 0

    def render(used_history_bins: int) -> str:
        lines = [
            (
                f"Now=0h at the latest history sample; sample_interval={_clock(step)}h; "
                "negative time=past."
            ),
        ]
        if include_history:
            lines.append("History cells use inclusive sample-time endpoints and exact sample statistics.")
            lines.append(_render_history(x, step, used_history_bins))
        else:
            lines.append(f"Observed current sample: t=0h value={_clock(x[-1])}")
        if f is not None:
            lines.extend(
                [
                    (
                        f"Future cells use point forecasts from source={source}; "
                        "future sample times start one interval after now."
                    ),
                    (
                        "Marginal q10/q90 are per-step quantiles, not path probabilities; "
                        "point extrema use only point values."
                    ),
                    _render_forecast(f, step, used_forecast_bins),
                ]
            )
        return "\n".join(lines)

    candidate_bins = (
        range(min(history_bins, len(x)), 0, -1) if include_history else (0,)
    )
    best = None
    for used_history_bins in candidate_bins:
        text = render(used_history_bins)
        token_count = count_tokens(text)
        if isinstance(token_count, bool) or not isinstance(token_count, (int, np.integer)):
            raise TypeError("count_tokens must return an integer")
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("count_tokens must return a nonnegative integer")
        if token_count <= budget:
            best = (text, token_count, used_history_bins)
            break
    if best is None:
        raise ValueError(f"Evidence budget {budget} is too small for minimum cells and provenance")

    text, token_count, used_history_bins = best
    return Evidence(
        text=text,
        evidence_tokens=token_count,
        history_error=None,
        forecast_error=None,
        history_knots=used_history_bins,
        forecast_knots=used_forecast_bins,
        quantile_crossings=crossings,
        elapsed_s=time.perf_counter() - started,
    )
