import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from temporal_bridge.benchmark import (
    build_cases,
    dilate_case_time,
    direct_answer,
    reverse_case_time,
)


def _write_etth1_csv(path, rows=17_420, step=timedelta(hours=1)):
    first_timestamp = datetime(2016, 7, 1)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "OT"])
        for index in range(rows):
            value = np.sin(index / 18.0) + 0.15 * np.sin(index / 4.0)
            timestamp = first_timestamp + index * step
            writer.writerow([timestamp.isoformat(sep=" "), f"{value:.8f}"])
    return path


@pytest.fixture
def etth1_csv(tmp_path):
    return _write_etth1_csv(tmp_path / "ETTh1.csv")


@pytest.mark.parametrize(("split", "expected"), [("dev", 36), ("heldout", 72)])
def test_builds_balanced_observed_and_forecast_cases(split, expected, etth1_csv):
    cases = build_cases(split, etth1_csv)

    assert len(cases) == expected
    assert sum(case.task == "observed" for case in cases) == expected // 2
    assert sum(case.task == "forecast" for case in cases) == expected // 2
    assert all(case.context.shape == (128,) for case in cases)
    assert all(case.step_hours > 0 for case in cases)
    forecast_cases = [case for case in cases if case.task == "forecast"]
    assert all(case.future.shape == (24,) for case in forecast_cases)
    assert all(case.metadata["source"] == "ETTh1" for case in forecast_cases)
    assert all(case.metadata["source"] == "synthetic" for case in cases if case.task == "observed")
    assert all(case.options[case.answer] in {"Q1", "Q2", "Q3", "Q4"}
               for case in forecast_cases)


def test_public_view_is_json_serializable_and_hides_scoring_metadata(etth1_csv):
    case = next(case for case in build_cases("dev", etth1_csv) if case.task == "forecast")

    public_view = case.public_view()
    encoded = json.dumps(public_view)

    assert set(public_view) == {"task", "context", "step_hours", "question", "options"}
    assert "future" not in encoded
    assert "answer" not in encoded
    assert "metadata" not in encoded
    assert "source_start" not in encoded
    assert len(public_view["context"]) == 128


def test_options_are_deterministic_shuffled_and_balanced():
    first = build_cases("dev")
    second = build_cases("dev")

    assert [case.options for case in first] == [case.options for case in second]
    for task in ("observed", "forecast"):
        options = [case.options for case in first if case.task == task]
        answer_slots = np.bincount(
            [case.answer for case in first if case.task == task], minlength=4
        )
        assert int(answer_slots.max() - answer_slots.min()) <= 1
        assert len({tuple(item) for item in options}) > 1


def test_real_windows_do_not_overlap_between_splits_or_within_a_split(tmp_path):
    data_path = _write_etth1_csv(tmp_path / "ETTh1.csv")
    dev = [case for case in build_cases("dev", data_path) if case.task == "forecast"]
    heldout = [case for case in build_cases("heldout", data_path) if case.task == "forecast"]

    for cases in (dev, heldout):
        spans = sorted((case.metadata["source_start"], case.metadata["source_stop"])
                       for case in cases)
        assert all(left[1] < right[0] for left, right in pairwise(spans))
    assert max(case.metadata["source_stop"] for case in dev) < min(
        case.metadata["source_start"] for case in heldout
    )


def test_synthetic_counterfactuals_have_exact_reversal_and_dilation_relations():
    cases = [case for case in build_cases("heldout") if case.task == "observed"]
    pairs = defaultdict(list)
    for case in cases:
        pairs[case.metadata["series_id"]].append(case)

    assert len(pairs) == 18
    assert all(case.options[case.answer] == _visible_label(case) for case in cases)
    for pair in pairs.values():
        assert len(pair) == 2
        original = next(case for case in pair if case.metadata["intervention"] == "original")
        intervention = next(case for case in pair if case.metadata["intervention"] != "original")
        assert original.family == intervention.family
        if intervention.metadata["intervention"] == "reversed":
            assert np.array_equal(intervention.context, original.context[::-1])
            assert intervention.step_hours == original.step_hours
            assert intervention.options[intervention.answer] == _reversed_label(original)
        else:
            assert intervention.metadata["intervention"] == "time_dilated"
            assert np.array_equal(intervention.context, original.context)
            assert intervention.step_hours == pytest.approx(original.step_hours * 1.5)
            assert intervention.options[intervention.answer] == original.options[original.answer]


def _reversed_label(case):
    answer = case.options[case.answer]
    return {
        "event_order": {
            "highest sample first": "lowest sample first",
            "lowest sample first": "highest sample first",
        },
        "trend": {
            "last-quarter mean higher": "last-quarter mean lower",
            "last-quarter mean lower": "last-quarter mean higher",
        },
        "turn_timing": {"before midpoint": "after midpoint", "after midpoint": "before midpoint"},
        "duration": {
            "above-zero state longer": "above-zero state longer",
            "at-or-below-zero state longer": "at-or-below-zero state longer",
        },
    }[case.family][answer]


def _visible_label(case):
    if case.family == "event_order":
        return "highest sample first" if np.argmax(case.context) < np.argmin(case.context) else "lowest sample first"
    if case.family == "trend":
        return (
            "last-quarter mean higher"
            if np.mean(case.context[-32:]) > np.mean(case.context[:32])
            else "last-quarter mean lower"
        )
    if case.family == "turn_timing":
        bend = int(np.argmax(np.abs(np.diff(case.context, n=2)))) + 1
        return "before midpoint" if bend < 64 else "after midpoint"
    above = np.count_nonzero(case.context > 0)
    return "above-zero state longer" if above > 64 else "at-or-below-zero state longer"


def test_public_transform_helpers_preserve_or_flip_answer_as_intended():
    case = next(
        case for case in build_cases("dev")
        if case.task == "observed" and case.family == "event_order"
        and case.metadata["intervention"] == "original"
    )

    reversed_case = reverse_case_time(case)
    dilated_case = dilate_case_time(case, 2.0)

    assert reversed_case.options[reversed_case.answer] == _reversed_label(case)
    assert dilated_case.options[dilated_case.answer] == case.options[case.answer]
    assert dilated_case.step_hours == case.step_hours * 2.0
    with pytest.raises(ValueError, match="greater than zero"):
        dilate_case_time(case, 0)


def test_direct_forecast_scorer_maps_prediction_to_public_option(etth1_csv):
    case = next(case for case in build_cases("dev", etth1_csv) if case.task == "forecast")
    prediction = np.zeros(24)
    prediction[14] = 100.0

    assert case.options[direct_answer(case, prediction)] == "Q3"


def test_build_cases_rejects_unknown_split_and_too_short_real_data(tmp_path, etth1_csv):
    with pytest.raises(ValueError, match="split must"):
        build_cases("test")
    short_path = _write_etth1_csv(tmp_path / "short.csv", rows=500)
    with pytest.raises(ValueError, match="at least"):
        build_cases("dev", short_path)


def test_real_csv_requires_exact_hourly_timestamps(tmp_path):
    irregular = _write_etth1_csv(
        tmp_path / "irregular.csv", rows=3, step=timedelta(hours=2)
    )

    with pytest.raises(ValueError, match="exact 1-hour cadence"):
        build_cases("dev", irregular)
