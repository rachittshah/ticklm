"""Leakage-resistant local cases for observed-time and forecast QA."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

_CONTEXT_LENGTH = 128
_HORIZON = 24
_WINDOW_LENGTH = _CONTEXT_LENGTH + _HORIZON
_WINDOW_GUARD = 40
_WINDOW_STRIDE = _WINDOW_LENGTH + _WINDOW_GUARD
_FAMILIES = ("event_order", "trend", "turn_timing", "duration")
_FORECAST_OPTIONS = ("Q1", "Q2", "Q3", "Q4")
_OBSERVED_OPTIONS = {
    "event_order": ("highest sample first", "lowest sample first", "same time", "no clear extremes"),
    "trend": ("last-quarter mean higher", "last-quarter mean lower", "means equal", "direction unclear"),
    "turn_timing": ("before midpoint", "after midpoint", "at midpoint", "no clear turn"),
    "duration": ("above-zero state longer", "at-or-below-zero state longer", "equal duration", "no clear states"),
}


@dataclass(frozen=True)
class Case:
    """One scored item; `public_view` deliberately excludes every scoring field."""

    case_id: str
    split: str
    family: str
    task: str
    context: np.ndarray
    future: np.ndarray | None
    step_hours: float
    question: str
    options: list[str]
    answer: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        context = np.asarray(self.context, dtype=np.float32)
        if context.ndim != 1 or context.size == 0 or not np.isfinite(context).all():
            raise ValueError("context must be a nonempty finite one-dimensional array")
        context = context.copy()
        context.setflags(write=False)
        object.__setattr__(self, "context", context)

        if self.future is not None:
            future = np.asarray(self.future, dtype=np.float32)
            if future.ndim != 1 or future.size == 0 or not np.isfinite(future).all():
                raise ValueError("future must be None or a nonempty finite one-dimensional array")
            future = future.copy()
            future.setflags(write=False)
            object.__setattr__(self, "future", future)

        if not np.isfinite(self.step_hours) or self.step_hours <= 0:
            raise ValueError("step_hours must be finite and greater than zero")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a nonempty string")
        options = list(self.options)
        if len(options) != 4 or len(set(options)) != 4:
            raise ValueError("options must contain four distinct choices")
        if isinstance(self.answer, bool) or not isinstance(self.answer, int) or not 0 <= self.answer < 4:
            raise ValueError("answer must be an option index from 0 through 3")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def public_view(self) -> dict[str, Any]:
        """Return only the information an inference prompt may receive."""
        return {
            "task": self.task,
            "context": self.context.tolist(),
            "step_hours": float(self.step_hours),
            "question": self.question,
            "options": list(self.options),
        }


def build_cases(split: str, real_csv: str | Path | None = None) -> list[Case]:
    """Build 36 dev or 72 held-out items with balanced observed and forecast tasks.

    The observed half is generated from fresh seeded signals and includes exact
    reversal or time-dilation counterfactuals. The forecast half uses fixed,
    non-overlapping windows from ETTh1 when supplied; without a CSV it uses a
    deterministic synthetic continuation so the harness remains runnable.
    """
    if split not in {"dev", "heldout"}:
        raise ValueError("split must be 'dev' or 'heldout'")

    seed_count = 9 if split == "dev" else 18
    seed_offset = 31_000 if split == "dev" else 73_000
    cases: list[Case] = []
    option_slot = 0

    for index in range(seed_count):
        seed = seed_offset + index
        family = _FAMILIES[index % len(_FAMILIES)]
        context, latent = _make_observed_signal(family, seed, split == "heldout")
        series_id = f"{split}-synthetic-{index:03d}"
        original = _observed_case(
            case_id=f"{series_id}-original",
            split=split,
            family=family,
            context=context,
            step_hours=1.0,
            metadata={
                **latent,
                "source": "synthetic",
                "series_id": series_id,
                "pair_id": series_id,
                "seed": seed,
                "intervention": "original",
            },
            option_slot=option_slot % 4,
            option_seed=seed,
        )
        cases.append(original)
        option_slot += 1

        # Each family receives both interventions, including in the smaller dev split.
        intervention_kind = "reversed" if (index // len(_FAMILIES)) % 2 == 0 else "time_dilated"
        if intervention_kind == "reversed":
            transformed = reverse_case_time(original)
        else:
            transformed = dilate_case_time(original, 1.5)
        transformed = _move_correct_option(transformed, option_slot % 4, seed + 500_000)
        cases.append(transformed)
        option_slot += 1

    if real_csv is None:
        cases.extend(_synthetic_forecast_cases(split, seed_count, seed_offset + 100_000))
    else:
        cases.extend(_etth1_forecast_cases(split, Path(real_csv)))
    return cases


def reverse_case_time(case: Case) -> Case:
    """Reverse an observed synthetic case and recompute its exact relation label."""
    if case.task != "observed":
        raise ValueError("time reversal is defined here only for observed cases")
    metadata = dict(case.metadata)
    for key in ("peak_index", "trough_index", "turn_index"):
        if key in metadata:
            metadata[key] = len(case.context) - 1 - int(metadata[key])
    if "trend_direction" in metadata:
        metadata["trend_direction"] = -int(metadata["trend_direction"])
    metadata["intervention"] = "reversed"
    metadata["paired_case_id"] = case.case_id
    reversed_context = case.context[::-1]
    answer_label = _observed_label(case.family, reversed_context)
    return replace(
        case,
        case_id=f"{case.case_id}-reversed",
        context=reversed_context,
        answer=case.options.index(answer_label),
        metadata=metadata,
    )


def dilate_case_time(case: Case, factor: float) -> Case:
    """Change the elapsed time per sample while preserving an order question's label."""
    if case.task != "observed":
        raise ValueError("time dilation is defined here only for observed cases")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("time dilation factor must be finite and greater than zero")
    metadata = dict(case.metadata)
    metadata["intervention"] = "time_dilated"
    metadata["time_scale_factor"] = float(factor)
    metadata["paired_case_id"] = case.case_id
    return replace(
        case,
        case_id=f"{case.case_id}-time-{factor:g}",
        step_hours=case.step_hours * float(factor),
        metadata=metadata,
    )


def direct_answer(case: Case, values: np.ndarray) -> int:
    """Score the frozen forecaster's argmax-bin decision without consulting labels."""
    if case.task != "forecast" or case.metadata.get("numerical_answer_type") != "argmax_quarter":
        raise ValueError("direct_answer supports forecast cases with argmax_quarter labels")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (_HORIZON,) or not np.isfinite(values).all():
        raise ValueError(f"forecast values must be a finite vector of length {_HORIZON}")
    quarter = int(np.argmax(values) // (_HORIZON // 4)) + 1
    return case.options.index(f"Q{quarter}")


def _observed_case(
    *,
    case_id: str,
    split: str,
    family: str,
    context: np.ndarray,
    step_hours: float,
    metadata: dict[str, Any],
    option_slot: int,
    option_seed: int,
) -> Case:
    question_by_family = {
        "event_order": (
            "Which occurs first in the observed context: the sample with the highest value or the "
            "sample with the lowest value? If tied, use the earliest sample."
        ),
        "trend": (
            "Compare the mean of the first 32 samples with the mean of the last 32 samples in "
            "this 128-sample context. Is the last-quarter mean higher or lower?"
        ),
        "turn_timing": (
            "Find the sample centered on the largest absolute second difference (the sharpest "
            "local bend). Does it occur before or after the context midpoint? If tied, use the "
            "earliest sample."
        ),
        "duration": (
            "Which state occurs more often in this context: above zero or at-or-below zero?"
        ),
    }
    answer_label = _observed_label(family, context)
    options, answer = _options_with_answer(
        _OBSERVED_OPTIONS[family], answer_label, option_slot, option_seed
    )
    return Case(
        case_id=case_id,
        split=split,
        family=family,
        task="observed",
        context=context,
        future=None,
        step_hours=step_hours,
        question=question_by_family[family],
        options=options,
        answer=answer,
        metadata=metadata,
    )


def _observed_label(family: str, context: np.ndarray) -> str:
    """Derive synthetic observed-task labels from the visible history itself."""
    if family == "event_order":
        return "highest sample first" if np.argmax(context) < np.argmin(context) else "lowest sample first"
    if family == "trend":
        first_mean = float(np.mean(context[:32]))
        last_mean = float(np.mean(context[-32:]))
        if last_mean > first_mean:
            return "last-quarter mean higher"
        if last_mean < first_mean:
            return "last-quarter mean lower"
        return "means equal"
    if family == "turn_timing":
        second_difference = np.diff(np.asarray(context, dtype=np.float64), n=2)
        turn_index = int(np.argmax(np.abs(second_difference))) + 1
        if turn_index < len(context) / 2:
            return "before midpoint"
        if turn_index > len(context) / 2:
            return "after midpoint"
        return "at midpoint"
    if family == "duration":
        above = int(np.count_nonzero(context > 0))
        at_or_below = len(context) - above
        if above > at_or_below:
            return "above-zero state longer"
        if at_or_below > above:
            return "at-or-below-zero state longer"
        return "equal duration"
    raise ValueError(f"unsupported observed family: {family}")


def _make_observed_signal(family: str, seed: int, heldout: bool) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    time = np.arange(_CONTEXT_LENGTH, dtype=np.float64)
    noise_scale = 0.018 if not heldout else 0.025
    background = 0.08 * np.sin(2 * np.pi * time / 31.0) + rng.normal(0, noise_scale, size=time.size)

    if family == "event_order":
        peak, trough = int(rng.integers(20, 48)), int(rng.integers(78, 108))
        if rng.integers(0, 2):
            peak, trough = trough, peak
        width_peak = int(rng.integers(5, 9))
        width_trough = int(rng.integers(5, 9))
        if heldout:
            peak_shape = np.maximum(0.0, 1.0 - np.abs(time - peak) / width_peak)
            trough_shape = np.maximum(0.0, 1.0 - np.abs(time - trough) / width_trough)
            morphology = "triangular_pulses"
        else:
            peak_shape = np.exp(-0.5 * ((time - peak) / (width_peak / 2.0)) ** 2)
            trough_shape = np.exp(-0.5 * ((time - trough) / (width_trough / 2.0)) ** 2)
            morphology = "gaussian_pulses"
        values = background + 3.0 * peak_shape - 3.0 * trough_shape
        return values.astype(np.float32), {
            "peak_index": peak,
            "trough_index": trough,
            "morphology": morphology,
        }

    if family == "trend":
        direction = 1 if rng.integers(0, 2) else -1
        scaled_time = time / (_CONTEXT_LENGTH - 1)
        curvature = 0.30 * scaled_time**2 if heldout else 0.0
        values = background + direction * (2.2 * scaled_time + curvature)
        return values.astype(np.float32), {
            "trend_direction": direction,
            "morphology": "curved_drift" if heldout else "linear_drift",
        }

    if family == "turn_timing":
        turn = int(rng.choice([rng.integers(25, 49), rng.integers(80, 104)]))
        is_peak = bool(rng.integers(0, 2))
        distance = (time - turn) / 64.0
        turn_background = 0.005 * np.sin(2 * np.pi * time / 31.0) + rng.normal(
            0, 0.004, size=time.size
        )
        if heldout:
            rounded_distance = np.sqrt(distance**2 + 0.02**2)
            shape = -rounded_distance if is_peak else rounded_distance
            morphology = "rounded_turn"
        else:
            shape = -np.abs(distance) if is_peak else np.abs(distance)
            morphology = "piecewise_linear_turn"
        values = turn_background + 2.8 * shape
        return values.astype(np.float32), {
            "turn_index": turn,
            "turn_kind": "peak" if is_peak else "trough",
            "morphology": morphology,
        }

    if family == "duration":
        high_is_longer = bool(rng.integers(0, 2))
        high_duration = int(rng.integers(78, 100) if high_is_longer else rng.integers(28, 51))
        low_duration = _CONTEXT_LENGTH - high_duration
        high_at_start = bool(rng.integers(0, 2))
        if high_at_start:
            levels = np.where(time < high_duration, 1.0, -1.0)
            transition = high_duration
        else:
            levels = np.where(time < low_duration, -1.0, 1.0)
            transition = low_duration
        if heldout:
            lo, hi = max(0, transition - 2), min(_CONTEXT_LENGTH, transition + 3)
            if hi > lo:
                ramp = np.linspace(levels[max(0, lo - 1)], levels[min(_CONTEXT_LENGTH - 1, hi)], hi - lo)
                levels[lo:hi] = ramp
            morphology = "soft_state_transition"
        else:
            morphology = "step_state_transition"
        values = levels + rng.normal(0, noise_scale, size=levels.size)
        return values.astype(np.float32), {
            "high_duration": high_duration,
            "low_duration": low_duration,
            "morphology": morphology,
        }

    raise ValueError(f"unsupported observed family: {family}")


def _move_correct_option(case: Case, option_slot: int, option_seed: int) -> Case:
    answer_label = case.options[case.answer]
    options, answer = _options_with_answer(case.options, answer_label, option_slot, option_seed)
    return replace(case, options=options, answer=answer)


def _options_with_answer(
    canonical_options: tuple[str, ...] | list[str],
    answer_label: str,
    option_slot: int,
    shuffle_seed: int,
) -> tuple[list[str], int]:
    if len(canonical_options) != 4 or answer_label not in canonical_options:
        raise ValueError("the answer must be one of four canonical options")
    options: list[str | None] = [None, None, None, None]
    options[option_slot] = answer_label
    distractors = [option for option in canonical_options if option != answer_label]
    np.random.default_rng(shuffle_seed).shuffle(distractors)
    distractor_iterator = iter(distractors)
    for index, option in enumerate(options):
        if option is None:
            options[index] = next(distractor_iterator)
    return [str(option) for option in options], option_slot


def _etth1_forecast_cases(split: str, real_csv: Path) -> list[Case]:
    values = _read_target_csv(real_csv, "OT")
    starts = _etth1_starts(len(values), split, 18 if split == "dev" else 36)
    cases = []
    for index, start in enumerate(starts):
        boundary = start + _CONTEXT_LENGTH
        stop = boundary + _HORIZON
        context = values[start:boundary]
        future = values[boundary:stop]
        quarter = int(np.argmax(future) // (_HORIZON // 4)) + 1
        options, answer = _options_with_answer(
            _FORECAST_OPTIONS, f"Q{quarter}", index % 4, 83_000 + index
        )
        cases.append(
            Case(
                case_id=f"{split}-etth1-{start:06d}",
                split=split,
                family="argmax_quarter",
                task="forecast",
                context=context,
                future=future,
                step_hours=1.0,
                question=(
                    "Which six-hour block of the next 24 hourly measurements contains the highest "
                    "value? Q1 is hours 1–6; Q2 is hours 7–12; Q3 is hours 13–18; Q4 is hours "
                    "19–24. If values tie, use the earliest hour."
                ),
                options=options,
                answer=answer,
                metadata={
                    "source": "ETTh1",
                    "series_id": f"ETTh1-OT-window-{start:06d}",
                    "source_series_id": "ETTh1-OT",
                    "source_start": start,
                    "source_stop": stop,
                    "target_column": "OT",
                    "numerical_answer_type": "argmax_quarter",
                    "answer_value": f"Q{quarter}",
                    "intervention": "none",
                },
            )
        )
    return cases


def _read_target_csv(path: Path, target: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or target not in reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'date' and '{target}' columns")
        values = []
        previous_timestamp = None
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["date"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid date timestamp on CSV line {line_number}") from error
            if previous_timestamp is not None:
                try:
                    interval = timestamp - previous_timestamp
                except TypeError as error:
                    raise ValueError("date timestamps must use a consistent timezone") from error
                if interval != timedelta(hours=1):
                    raise ValueError(
                        f"date timestamps must increase at an exact 1-hour cadence "
                        f"(line {line_number})"
                    )
            try:
                value = float(row[target])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid {target} value on CSV line {line_number}") from error
            if not np.isfinite(value):
                raise ValueError(f"non-finite {target} value on CSV line {line_number}")
            values.append(value)
            previous_timestamp = timestamp
    if not values:
        raise ValueError("CSV has no data rows")
    return np.asarray(values, dtype=np.float32)


def _etth1_starts(length: int, split: str, count: int) -> list[int]:
    max_start = length - _WINDOW_LENGTH
    if max_start < 0:
        raise ValueError("ETTh1 CSV must contain enough rows for a 128+24 point window")
    dev_limit = int(max_start * 0.4)
    heldout_start = int(np.ceil(max_start * 0.6))
    if split == "dev":
        starts = np.rint(np.linspace(0, dev_limit, count)).astype(int).tolist()
    else:
        starts = np.rint(np.linspace(heldout_start, max_start, count)).astype(int).tolist()
    if any(right - left < _WINDOW_STRIDE for left, right in pairwise(starts)):
        raise ValueError("ETTh1 CSV needs at least 16,952 rows for disjoint guarded windows")
    return starts


def _synthetic_forecast_cases(split: str, count: int, seed_offset: int) -> list[Case]:
    cases = []
    for index in range(count * 2):
        seed = seed_offset + index
        rng = np.random.default_rng(seed)
        time = np.arange(_WINDOW_LENGTH, dtype=np.float64)
        phase = rng.uniform(0, 2 * np.pi)
        signal = (
            np.sin(2 * np.pi * time / _HORIZON + phase)
            + 0.18 * np.sin(2 * np.pi * time / 8.0 + phase / 2.0)
            + rng.normal(0, 0.025, size=time.size)
        ).astype(np.float32)
        context, future = signal[:_CONTEXT_LENGTH], signal[_CONTEXT_LENGTH:]
        quarter = int(np.argmax(future) // (_HORIZON // 4)) + 1
        options, answer = _options_with_answer(
            _FORECAST_OPTIONS, f"Q{quarter}", index % 4, 97_000 + index
        )
        cases.append(
            Case(
                case_id=f"{split}-synthetic-forecast-{index:03d}",
                split=split,
                family="argmax_quarter",
                task="forecast",
                context=context,
                future=future,
                step_hours=1.0,
                question=(
                    "Which six-hour block of the next 24 hourly measurements contains the highest "
                    "value? Q1 is hours 1–6; Q2 is hours 7–12; Q3 is hours 13–18; Q4 is hours "
                    "19–24. If values tie, use the earliest hour."
                ),
                options=options,
                answer=answer,
                metadata={
                    "source": "synthetic",
                    "series_id": f"{split}-synthetic-forecast-{index:03d}",
                    "seed": seed,
                    "numerical_answer_type": "argmax_quarter",
                    "answer_value": f"Q{quarter}",
                    "intervention": "none",
                },
            )
        )
    return cases
