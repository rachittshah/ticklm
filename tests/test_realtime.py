import numpy as np
import pytest

from realtime import (
    _read_event_chunk,
    exact_reply,
    numeric_reply,
    predictive_delay,
    regular_history,
)


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
