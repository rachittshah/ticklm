import pytest

from temporal_bridge.surgery_live import make_case, release_assessment

CHOICES = {"A": "alert", "B": "wait", "C": "restart", "D": "cancel"}


def test_clock_crossing_can_invalidate_a_correct_snapshot_answer():
    case = make_case("silence", {"now": 1.0, "last_event": 0.0, "threshold": 2.0}, CHOICES, "before")
    result = release_assessment(case, "B", {"now": 2.1, "last_event": 0.0, "threshold": 2.0})
    assert result["snapshot_correct"] is True
    assert result["current_correct"] is False
    assert result["publishable"] is False


def test_new_observation_invalidates_an_old_silence_alert():
    case = make_case("silence", {"now": 3.0, "last_event": 0.0, "threshold": 2.0}, CHOICES, "alert")
    result = release_assessment(case, "A", {"now": 3.2, "last_event": 3.1, "threshold": 2.0})
    assert result["snapshot_correct"] is True
    assert result["publishable"] is False


def test_cancelled_plan_changes_while_model_is_working():
    choices = {"A": "continue", "B": "replan", "C": "stop", "D": "wait"}
    state = {"event": "cancel", "plan_id": "job-a", "target_id": "job-b", "plan_at": 0.0, "event_at": 0.1}
    case = make_case("relevance", state, choices, "cancel")
    result = release_assessment(case, "A", {**state, "target_id": "job-a", "event_at": 0.2})
    assert result["snapshot_correct"] is True
    assert result["current_correct"] is False
    assert result["publishable"] is False


def test_wrong_prediction_is_not_rescued_by_matching_a_later_state():
    case = make_case("silence", {"now": 1.0, "last_event": 0.0, "threshold": 2.0}, CHOICES, "wrong")
    result = release_assessment(case, "A", {"now": 3.0, "last_event": 0.0, "threshold": 2.0})
    assert result["snapshot_correct"] is False
    assert result["current_correct"] is True
    assert result["publishable"] is False


def test_live_prompt_does_not_include_gold_metadata():
    case = make_case("silence", {"now": 1.0, "last_event": 0.0, "threshold": 2.0}, CHOICES, "private-id")
    assert "private-id" not in case["prompt"]
    assert "correct_action" not in case["prompt"]
    with pytest.raises(ValueError):
        release_assessment(case, "Z", case["state"])


def test_live_feature_view_preserves_state_and_guard_semantics():
    state = {"now": 1.5, "last_event": 0.5, "threshold": 2.0}
    raw = make_case("silence", state, CHOICES, "raw")
    featured = make_case("silence", state, CHOICES, "featured", input_view="clock-features")
    assert featured["state"] == raw["state"]
    assert featured["answer"] == raw["answer"]
    assert featured["prompt"] != raw["prompt"]
    assert featured["prompt"].endswith(raw["prompt"])
    assert release_assessment(featured, "B", state) == release_assessment(raw, "B", state)
