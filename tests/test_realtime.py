from types import SimpleNamespace

import numpy as np
import pytest

from realtime import (
    _read_event_chunk,
    evaluate_trial,
    exact_reply,
    numeric_reply,
    parse_args,
    predictive_delay,
    regular_history,
    response_measurements,
)


def test_cli_providers_choose_model_names_without_local_model_paths():
    for provider, model in (("claude", "opus"), ("codex", "gpt-6-astra")):
        args = parse_args(["self-test", "--run-dir", "artifacts/trial", "--provider", provider])
        assert args.model == model
        assert args.claude_profile == "startup"
        assert args.effort == "xhigh"
        assert args.trial_timeout_s == 70
    args = parse_args(["watch", "--run-dir", "artifacts/watch", "--provider", "codex",
                       "--model", "explicit-model", "--trial-timeout-s", "240"])
    assert args.model == "explicit-model"
    assert args.trial_timeout_s == 240


@pytest.mark.parametrize("flag", ["--request-timeout-s", "--trial-timeout-s"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_unbounded_or_invalid_timeouts(flag, value):
    with pytest.raises(SystemExit):
        parse_args(["self-test", "--run-dir", "artifacts/trial", flag, value])


def test_latency_report_uses_ticket_to_visible_text_and_retains_expired_answer():
    runtime = SimpleNamespace(
        worker=SimpleNamespace(model_path="requested-model"),
        messages=[{"type": "started", "request_id": "run:q", "monotonic": 102},
                  {"type": "first_token", "request_id": "run:q", "monotonic": 105,
                   "evidence": "visible_text_delta"}],
        tickets={"run:q": ({"issued_mono": 100}, "q", False)},
        results={"q": {"monotonic": 108, "model": "served-model", "valid": False,
                       "validity_reason": "expired", "text": "4.528", "input_tokens": 40}},
    )
    result = response_measurements(runtime)["q"]
    assert result["first_visible_text_s"] == 5
    assert result["completion_s"] == 8
    assert result["valid"] is False
    assert result["model"] == "served-model"
    assert result["output_tokens"] is None


def test_correct_but_expired_clock_answer_cannot_pass_overall_trial():
    results = {
        "fresh_clock": {"text": "EXPIRED", "valid": True},
        "elapsed_number": {"text": "4.528", "snapshot_elapsed_s": 4.5277, "valid": False},
        "refresh": {"text": "ALARM; EXPIRED", "valid": True},
        "freshness": {"text": "STALE", "valid": True},
    }
    assessment = evaluate_trial({"existing_scripted_check": True}, results)
    assert all(assessment["answer_correctness"].values())
    assert assessment["scripted_checks_passed"] is True
    assert assessment["publication_checks"]["elapsed_number"] is False
    assert assessment["all_checks_passed"] is False


def test_predictive_wake_is_bounded_and_uses_first_meaningful_change():
    assert predictive_delay([0, 0, 0, 2], 0, sample_s=.5) == 2
    assert predictive_delay([0, 0, 0], 0) == 3
    assert predictive_delay([10], 0) == .25
    with pytest.raises(ValueError):
        predictive_delay([float("nan")], 0)


def test_resampling_never_invents_missing_history_or_fills_long_gap():
    samples = [(i * .1, float(i)) for i in range(130)]
    values = regular_history(samples)
    assert values.shape == (128,)
    assert values[-1] == 129
    assert regular_history(samples[:30]) is None
    missing = samples[:60] + samples[80:]
    assert regular_history(missing) is None
    # Enough records do not make a multi-second sensor blackout regularly sampled.
    missing += [(13 + i * .1, 130.0 + i) for i in range(25)]
    assert len(missing) > 128
    assert regular_history(missing) is None


def test_regular_resampling_uses_only_past_points():
    samples = [(i * .101, float(i)) for i in range(140)]
    values = regular_history(samples)
    assert np.isfinite(values).all()
    assert values.min() >= 0
    assert values.max() <= samples[-1][1]


def test_live_grading_rejects_negation_and_ambiguous_answers():
    assert exact_reply({"text": "EXPIRED."}, "EXPIRED")
    assert not exact_reply({"text": "NOT EXPIRED"}, "EXPIRED")
    assert not exact_reply({"text": "STALE or FRESH"}, "STALE")
    assert numeric_reply({"text": "4.25"}) == 4.25
    assert numeric_reply({"text": "about 4"}) is None


def test_event_reader_ignores_backlog_and_handles_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("old event\n")
    chunk, offset, identity, reset = _read_event_chunk(path, None, None)
    assert chunk == ""
    assert not reset
    with path.open("a") as stream:
        stream.write("new event\n")
    chunk, offset, identity, reset = _read_event_chunk(path, offset, identity)
    assert chunk == "new event\n"
    path.write_text("")
    chunk, offset, identity, reset = _read_event_chunk(path, offset, identity)
    assert reset and chunk == "" and offset == 0
