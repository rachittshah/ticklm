from datetime import UTC, datetime

import pytest

from temporal_bridge.realtime_state import TemporalState, render_clock


class FakeClock:
    def __init__(self, mono=100.0, wall=1_800_000_000.0):
        self.mono = mono
        self.wall = wall

    def now_mono(self):
        return self.mono

    def now_wall(self):
        return self.wall

    def advance(self, seconds):
        self.mono += seconds
        self.wall += seconds


def make_state(clock, run_id="run-a"):
    return TemporalState(run_id, now_mono=clock.now_mono, now_wall=clock.now_wall)


def test_utc_jumps_do_not_change_monotonic_elapsed_or_deadline():
    clock = FakeClock()
    state = make_state(clock)
    state.set_deadline("break", after_s=5)

    clock.mono += 3
    clock.wall -= 3_600
    before = state.snapshot()
    assert before["elapsed_s"] == 3
    assert before["deadlines"][0]["remaining_s"] == 2
    assert before["deadlines"][0]["status"] == "pending"

    clock.mono += 2
    clock.wall += 86_400
    due = state.tick()
    assert len(due) == 1
    assert due[0]["deadline_id"] == "break"
    assert due[0]["lateness_s"] == 0
    assert state.snapshot()["deadlines"][0]["status"] == "fired"
    assert state.snapshot()["elapsed_s"] == 5


def test_samples_update_data_revision_while_events_and_deadlines_update_semantic_revision():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(1.0)
    state.observe(2.0)
    assert state.revision == 0
    assert state.data_revision == 2

    state.event("threshold_warning", {"level": 2})
    assert state.revision == 1
    state.set_deadline("follow-up", after_s=0)
    assert state.revision == 2
    assert state.snapshot()["deadlines"][0]["status"] == "due"

    first = state.tick()
    assert [item["deadline_id"] for item in first] == ["follow-up"]
    assert state.revision == 3
    assert state.tick() == []
    assert state.revision == 3
    snapshot = state.snapshot()
    assert [event["name"] for event in snapshot["events"]] == [
        "threshold_warning",
        "deadline_due",
    ]


def test_observation_rejects_nonfinite_regressing_and_future_sample_times():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(1.0, observed_mono=99)

    with pytest.raises(ValueError, match="finite"):
        state.observe(float("nan"))
    with pytest.raises(ValueError, match="older"):
        state.observe(2.0, observed_mono=98)
    with pytest.raises(ValueError, match="future"):
        state.observe(2.0, observed_mono=101)

    assert state.data_revision == 1
    assert state.snapshot()["observation"] == {
        "value": 1.0,
        "age_s": 1.0,
        "freshness": "FRESH",
        "freshness_limit_s": 1.0,
    }


def test_request_validation_rejects_semantic_change_and_ttl_expiry_without_new_data():
    clock = FakeClock()
    state = make_state(clock)
    request = state.issue_request("req-1", ttl_s=3)
    assert state.validate_result(request) == (True, "valid")

    clock.advance(3.01)
    assert state.data_revision == 0
    assert state.validate_result(request) == (False, "expired")

    current = state.issue_request("req-2", ttl_s=3)
    state.event("user_changed_goal")
    assert state.validate_result(current) == (False, "state_changed")


def test_request_identity_is_bound_to_run_and_engine_instance():
    clock = FakeClock()
    first = make_state(clock, "run-a")
    request = first.issue_request("same-id", ttl_s=10)

    other_run = make_state(clock, "run-b")
    assert other_run.validate_result(request) == (False, "wrong_run")

    restarted = make_state(clock, "run-a")
    assert restarted.validate_result(request) == (False, "unknown_request")


def test_unbound_request_remains_valid_after_a_new_data_revision():
    clock = FakeClock()
    state = make_state(clock)
    ticket = state.issue_request("as-of-story", ttl_s=3)
    state.observe(10.0)

    assert state.data_revision == 1
    assert state.revision == ticket["revision"]
    assert state.validate_result(ticket) == (True, "valid")


def test_observation_bound_request_rejects_newer_sample():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(10.0)
    ticket = state.issue_request("observation-bound", bind_observation=True)
    state.observe(11.0)

    assert state.validate_result(ticket) == (False, "observation_changed")


def test_observation_bound_request_expires_when_fresh_sample_becomes_stale():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(10.0)
    ticket = state.issue_request("freshness-bound", bind_observation=True)
    clock.mono += 1.001

    assert state.data_revision == ticket["data_revision"]
    assert state.validate_result(ticket) == (False, "observation_stale")


def test_deadline_crossing_invalidates_before_tick_and_after_one_shot_tick():
    clock = FakeClock()
    state = make_state(clock)
    state.set_deadline("urgent", after_s=1)
    ticket = state.issue_request("before-deadline", ttl_s=5)
    revision = state.revision
    clock.mono += 1

    assert state.validate_result(ticket) == (False, "deadline_crossed")
    assert state.revision == revision  # Validation does not fire or record the timer.
    assert state.snapshot()["deadlines"][0]["status"] == "due"
    assert state.tick()[0]["deadline_id"] == "urgent"
    assert state.validate_result(ticket) == (False, "deadline_crossed")
    assert state.tick() == []


def test_request_identity_covers_observation_and_deadline_bindings():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(3.0)
    state.set_deadline("review", after_s=4)
    ticket = state.issue_request("identified", bind_observation=True)

    tampered = dict(ticket)
    tampered["bind_observation"] = False
    assert state.validate_result(tampered) == (False, "unknown_request")

    tampered = dict(ticket)
    tampered["pending_deadline_expiry_bounds"] = []
    assert state.validate_result(tampered) == (False, "unknown_request")


def test_nested_deadline_identity_is_detached_from_returned_ticket():
    clock = FakeClock()
    state = make_state(clock)
    state.set_deadline("review", after_s=4)
    ticket = state.issue_request("nested-identity")

    ticket["pending_deadline_expiry_bounds"][0]["expires_mono"] += 10

    assert state.validate_result(ticket) == (False, "unknown_request")


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_deadline_and_request_intervals_must_be_finite_and_nonnegative(seconds):
    state = make_state(FakeClock())
    with pytest.raises(ValueError, match="after_s"):
        state.set_deadline("bad", after_s=seconds)
    with pytest.raises(ValueError, match="ttl_s"):
        state.issue_request("bad", ttl_s=seconds)


def test_snapshot_and_clock_render_include_only_captured_live_clock_facts():
    clock = FakeClock(wall=datetime(2027, 1, 15, tzinfo=UTC).timestamp())
    state = make_state(clock)
    state.observe(12.5)
    state.event("sensor_connected", {"source": "demo"})
    state.set_deadline("check", after_s=4)
    clock.advance(1.25)

    snapshot = state.snapshot()
    rendered = render_clock(snapshot)
    assert snapshot["wall_utc"].endswith("Z")
    assert snapshot["elapsed_s"] == 1.25
    assert snapshot["observation"] == {
        "value": 12.5,
        "age_s": 1.25,
        "freshness": "STALE",
        "freshness_limit_s": 1.0,
    }
    assert "Current UTC wall-clock reading:" in rendered
    assert "Run elapsed time: 1.250 seconds" in rendered
    assert "The latest measurement was observed 1.250 seconds ago" in rendered
    assert "Freshness: STALE (limit 1.000 seconds)" in rendered
    assert "Deadline check: PENDING" in rendered
    assert "2.750 seconds remain" in rendered
    assert "Recent event sensor_connected was observed 1.250 seconds ago" in rendered


def test_rendered_deadline_expiry_is_explicit_before_and_after_tick():
    clock = FakeClock()
    state = make_state(clock)
    state.set_deadline("probe", after_s=1)
    assert "Deadline probe: PENDING" in render_clock(state.snapshot())

    clock.mono += 1.25
    expired = state.snapshot()
    assert expired["deadlines"][0]["expiry_status"] == "EXPIRED"
    before_tick = render_clock(expired)
    assert "Deadline probe: EXPIRED" in before_tick
    assert "its scheduled time passed 0.250 seconds ago" in before_tick
    assert "The deadline event was not yet delivered" in before_tick
    assert "fired" not in before_tick.lower()

    state.tick()
    after_tick = render_clock(state.snapshot())
    assert "Deadline probe: EXPIRED" in after_tick
    assert "The deadline event was delivered" in after_tick


def test_observation_freshness_boundary_uses_monotonic_age():
    clock = FakeClock()
    state = make_state(clock)
    state.observe(5.0)
    clock.mono += 1
    assert state.snapshot()["observation"]["freshness"] == "FRESH"
    clock.mono += 0.001
    assert state.snapshot()["observation"]["freshness"] == "STALE"
