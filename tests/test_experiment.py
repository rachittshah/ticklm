from dataclasses import replace

import numpy as np
import pytest

from temporal_bridge.benchmark import build_cases
from temporal_bridge.experiment import (
    _series_key,
    make_evidence,
    make_prompt,
    paired_difference,
    validate_frozen_identity,
)


def test_prompt_and_evidence_do_not_read_hidden_answers_or_future():
    case = next(c for c in build_cases("dev") if c.task == "forecast")
    changed = replace(case, future=np.full(24, 987654), answer=(case.answer + 1) % 4,
                      metadata={"private": "secret-target"})
    assert make_prompt(case, "history") == make_prompt(changed, "history")
    assert "secret-target" not in make_prompt(changed, "history")
    left = make_evidence(case, "raw", "uniform", 384, len, {})[0]
    right = make_evidence(changed, "raw", "uniform", 384, len, {})[0]
    assert left.text == right.text


def test_observed_capsules_do_not_receive_any_forecast():
    case = build_cases("dev")[0]
    evidence = [make_evidence(case, arm, "adaptive", 1600, len, {})[0].text
                for arm in ("cheap", "timesfm", "mismatched")]
    assert evidence[0] == evidence[1] == evidence[2]
    assert "Projected" not in evidence[0]


def test_mismatched_control_has_explicit_distinct_donor_and_budget():
    cases = [c for c in build_cases("dev") if c.task == "forecast"][:2]
    forecasts = {}
    for i, case in enumerate(cases):
        forecast = {"q10": np.arange(24) + i, "point": np.arange(24) + i + 1,
                    "q90": np.arange(24) + i + 2, "elapsed_s": .01}
        forecasts[_series_key(case)] = {"case_id": case.case_id, "mean": case.context.mean(),
                                       "std": case.context.std(), "timesfm": forecast}
    evidence, _, donor = make_evidence(cases[0], "mismatched", "adaptive", 1600, len, forecasts)
    assert donor == cases[1].case_id
    assert evidence.evidence_tokens == len(evidence.text) <= 1600


def test_bootstrap_clusters_pairs_and_requires_identical_ids():
    rows = []
    for i in range(4):
        for config, correct in (("new", True), ("old", False)):
            rows.append({"case_id": str(i), "series_id": str(i // 2), "task": "observed",
                         "config_id": config, "correct": correct})
    result = paired_difference(rows, "new", "old", "observed")
    assert result["clusters"] == 2
    assert result["difference"] == 1
    assert result["ci95"] == [1, 1]
    with pytest.raises(ValueError, match="identical"):
        paired_difference(rows[:-1], "new", "old", "observed")


@pytest.mark.parametrize("field", ["weights", "tokenizer", "dependencies", "tsfm_source", "checkpoint"])
def test_frozen_runtime_identity_detects_any_changed_component(field):
    frozen = {"weights": "w", "tokenizer": "t", "dependencies": "d",
              "tsfm_source": "s", "checkpoint": "c"}
    validate_frozen_identity(frozen, dict(frozen))
    changed = {**frozen, field: "different"}
    with pytest.raises(ValueError, match="identity"):
        validate_frozen_identity(frozen, changed)
