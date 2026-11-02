"""CPU-only, versioned temporal state for live observations and deadlines."""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock


def _utc(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_payload(payload: dict | None) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary or None")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must contain only JSON-safe finite values") from exc


@dataclass
class Observation:
    value: float
    observed_mono: float


@dataclass
class Event:
    name: str
    observed_mono: float
    revision: int
    payload: dict = field(default_factory=dict)


@dataclass
class Deadline:
    deadline_id: str
    expires_mono: float
    fired_at_mono: float | None = None


@dataclass
class TemporalState:
    """Own a run's clocks, observations, semantic revision, events and deadlines."""

    run_id: str
    now_mono: Callable[[], float] = time.monotonic
    now_wall: Callable[[], float] = time.time
    max_age_s: float = 1.0
    start_mono: float = field(init=False)
    start_wall: float = field(init=False)
    revision: int = field(default=0, init=False)
    data_revision: int = field(default=0, init=False)
    _observations: deque = field(
        default_factory=lambda: deque(maxlen=128), init=False, repr=False
    )
    _events: deque = field(default_factory=lambda: deque(maxlen=4), init=False, repr=False)
    _deadlines: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _requests: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _last_reason_mono: float | None = field(default=None, init=False, repr=False)
    _last_reason_name: str | None = field(default=None, init=False, repr=False)
    _last_read_mono: float = field(default=float("-inf"), init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not callable(self.now_mono) or not callable(self.now_wall):
            raise TypeError("now_mono and now_wall must be callable")
        self.max_age_s = _finite_float(self.max_age_s, "max_age_s")
        if self.max_age_s < 0:
            raise ValueError("max_age_s must be nonnegative")
        self.start_mono, self.start_wall = self._read_clock()

    def _read_clock(self) -> tuple[float, float]:
        mono = _finite_float(self.now_mono(), "monotonic clock")
        wall = _finite_float(self.now_wall(), "wall clock")
        if mono < self._last_read_mono:
            raise ValueError("monotonic clock moved backwards")
        self._last_read_mono = mono
        return mono, wall

    def observe(self, value: float, observed_mono: float | None = None) -> int:
        """Append one finite sample; its timestamp is local monotonic arrival time."""
        numeric_value = _finite_float(value, "observation value")
        with self._lock:
            now_mono, _ = self._read_clock()
            sample_mono = (
                now_mono if observed_mono is None
                else _finite_float(observed_mono, "observed_mono")
            )
            if sample_mono > now_mono:
                raise ValueError("observed_mono cannot be in the future")
            if self._observations and sample_mono < self._observations[-1].observed_mono:
                raise ValueError("observed_mono cannot be older than the previous sample")
            self._observations.append(Observation(numeric_value, sample_mono))
            self.data_revision += 1
            return self.data_revision

    def _record_event(self, name: str, payload: dict, at_mono: float) -> Event:
        self.revision += 1
        record = Event(name, at_mono, self.revision, payload)
        self._events.append(record)
        self._last_reason_mono = at_mono
        self._last_reason_name = name
        return record

    def event(self, name: str, payload: dict | None = None) -> int:
        """Record a semantic event and return the new semantic revision."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("event name must be a nonempty string")
        safe_payload = _json_payload(payload)
        with self._lock:
            now_mono, _ = self._read_clock()
            return self._record_event(name, safe_payload, now_mono).revision

    @staticmethod
    def _deadline_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("deadline_id must be a nonempty string")
        return value

    def set_deadline(self, deadline_id: str, after_s: float) -> dict:
        """Schedule or replace a deadline relative to the monotonic clock."""
        key = self._deadline_id(deadline_id)
        delay = _finite_float(after_s, "after_s")
        if delay < 0:
            raise ValueError("after_s must be nonnegative")
        with self._lock:
            now_mono, _ = self._read_clock()
            deadline = Deadline(key, now_mono + delay)
            self._deadlines[key] = deadline
            self.revision += 1
            self._last_reason_mono = now_mono
            self._last_reason_name = "deadline_scheduled"
            return {
                "run_id": self.run_id,
                "deadline_id": key,
                "expires_mono": deadline.expires_mono,
                "revision": self.revision,
            }

    def tick(self) -> list[dict]:
        """Fire every due deadline once, independently of any model response."""
        with self._lock:
            now_mono, now_wall = self._read_clock()
            due = [
                deadline for deadline in self._deadlines.values()
                if deadline.fired_at_mono is None and now_mono >= deadline.expires_mono
            ]
            fired = []
            for deadline in due:
                deadline.fired_at_mono = now_mono
                lateness = max(0.0, now_mono - deadline.expires_mono)
                event = self._record_event(
                    "deadline_due", {"deadline_id": deadline.deadline_id}, now_mono
                )
                fired.append({
                    "run_id": self.run_id,
                    "deadline_id": deadline.deadline_id,
                    "scheduled_mono": deadline.expires_mono,
                    "fired_at_mono": now_mono,
                    "lateness_s": lateness,
                    "wall_utc": _utc(now_wall),
                    "revision": event.revision,
                })
            return fired

    def _snapshot_at(self, now_mono: float, now_wall: float) -> dict:
        latest = self._observations[-1] if self._observations else None
        observation = None
        if latest is not None:
            age = max(0.0, now_mono - latest.observed_mono)
            observation = {
                "value": latest.value,
                "age_s": age,
                "freshness": "STALE" if age > self.max_age_s else "FRESH",
                "freshness_limit_s": self.max_age_s,
            }
        deadlines = []
        for deadline in sorted(self._deadlines.values(), key=lambda item: item.deadline_id):
            remaining = deadline.expires_mono - now_mono
            expired = remaining <= 0
            status = (
                "fired" if deadline.fired_at_mono is not None
                else "due" if expired
                else "pending"
            )
            deadlines.append({
                "id": deadline.deadline_id,
                "remaining_s": max(0.0, remaining),
                "overdue_s": max(0.0, -remaining),
                "status": status,
                "expiry_status": "EXPIRED" if expired else "PENDING",
                "event_status": "delivered" if deadline.fired_at_mono is not None
                else "not_delivered",
            })
        events = [
            {
                "name": event.name,
                "age_s": max(0.0, now_mono - event.observed_mono),
                "revision": event.revision,
                "payload": json.loads(json.dumps(event.payload, allow_nan=False)),
            }
            for event in self._events
        ]
        return {
            "run_id": self.run_id,
            "revision": self.revision,
            "data_revision": self.data_revision,
            "wall_utc": _utc(now_wall),
            "start_wall_utc": _utc(self.start_wall),
            "elapsed_s": max(0.0, now_mono - self.start_mono),
            "now_mono": now_mono,
            "observation": observation,
            "deadlines": deadlines,
            "events": events,
            "last_reason": self._last_reason_name,
            "time_since_last_reason_s": (
                None if self._last_reason_mono is None
                else max(0.0, now_mono - self._last_reason_mono)
            ),
        }

    def snapshot(self) -> dict:
        """Capture JSON-safe clock and state fields at one pair of clock readings."""
        with self._lock:
            now_mono, now_wall = self._read_clock()
            return self._snapshot_at(now_mono, now_wall)

    def issue_request(
        self, request_id: str, ttl_s: float = 3, *, bind_observation: bool = False
    ) -> dict:
        """Capture a versioned request ticket with a monotonic publication expiry."""
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(bind_observation, bool):
            raise TypeError("bind_observation must be a boolean")
        ttl = _finite_float(ttl_s, "ttl_s")
        if ttl <= 0:
            raise ValueError("ttl_s must be positive")
        with self._lock:
            now_mono, now_wall = self._read_clock()
            self._requests = OrderedDict(
                (key, value) for key, value in self._requests.items()
                if now_mono <= value["valid_until_monotonic"]
            )
            if request_id in self._requests:
                raise ValueError("request_id is already outstanding")
            result = self._snapshot_at(now_mono, now_wall)
            pending_deadlines = [
                {"id": deadline.deadline_id, "expires_mono": deadline.expires_mono}
                for deadline in sorted(
                    self._deadlines.values(), key=lambda item: item.deadline_id
                )
                if deadline.fired_at_mono is None and deadline.expires_mono > now_mono
            ]
            bound_observation = result["observation"] if bind_observation else None
            result.update({
                "request_id": request_id,
                "issued_mono": now_mono,
                "ttl_s": ttl,
                "valid_until_monotonic": now_mono + ttl,
                "bind_observation": bind_observation,
                "bound_data_revision": self.data_revision if bind_observation else None,
                "bound_observation_freshness": (
                    None if bound_observation is None else bound_observation["freshness"]
                ),
                "bound_freshness_limit_s": (
                    None if bound_observation is None else self.max_age_s
                ),
                "pending_deadline_expiry_bounds": pending_deadlines,
            })
            self._requests[request_id] = deepcopy({
                key: result[key]
                for key in (
                    "run_id", "revision", "issued_mono", "ttl_s", "valid_until_monotonic",
                    "bind_observation", "bound_data_revision", "bound_observation_freshness",
                    "bound_freshness_limit_s", "pending_deadline_expiry_bounds",
                )
            })
            while len(self._requests) > 256:
                self._requests.popitem(last=False)
            return result

    def validate_result(self, request: dict) -> tuple[bool, str]:
        """Check the original ticket before publishing its associated model result."""
        if not isinstance(request, dict):
            return False, "invalid_request"
        if request.get("run_id") != self.run_id:
            return False, "wrong_run"
        request_id = request.get("request_id")
        if not isinstance(request_id, str):
            return False, "unknown_request"
        with self._lock:
            expected = self._requests.get(request_id)
            if expected is None:
                return False, "unknown_request"
            identity_keys = (
                "run_id", "revision", "issued_mono", "ttl_s", "valid_until_monotonic",
                "bind_observation", "bound_data_revision", "bound_observation_freshness",
                "bound_freshness_limit_s", "pending_deadline_expiry_bounds",
            )
            if any(request.get(key) != expected[key] for key in identity_keys):
                return False, "unknown_request"
            now_mono, _ = self._read_clock()
            if any(
                now_mono >= deadline["expires_mono"]
                for deadline in expected["pending_deadline_expiry_bounds"]
            ):
                return False, "deadline_crossed"
            if now_mono - expected["issued_mono"] > expected["ttl_s"]:
                return False, "expired"
            if self.revision != expected["revision"]:
                return False, "state_changed"
            if expected["bind_observation"]:
                if self.data_revision != expected["bound_data_revision"]:
                    return False, "observation_changed"
                if expected["bound_observation_freshness"] == "FRESH":
                    latest = self._observations[-1] if self._observations else None
                    if latest is None or (
                        now_mono - latest.observed_mono > expected["bound_freshness_limit_s"]
                    ):
                        return False, "observation_stale"
            return True, "valid"


def render_clock(snapshot: dict) -> str:
    """Render only captured clock, observation-age, deadline and event facts."""
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a dictionary")
    observation = snapshot.get("observation")
    lines = [
        f"Current UTC wall-clock reading: {snapshot['wall_utc']}.",
        (
            f"Run elapsed time: {snapshot['elapsed_s']:.3f} seconds since run start, "
            "measured with a monotonic clock."
        ),
    ]
    if observation is None:
        lines.append("Latest measurement: unavailable. Freshness: UNKNOWN.")
    else:
        lines.append(
            f"Latest measurement value: {observation['value']!r}. The latest measurement was observed "
            f"{observation['age_s']:.3f} seconds ago. Freshness: {observation['freshness']} "
            f"(limit {observation['freshness_limit_s']:.3f} seconds)."
        )
    deadlines = snapshot.get("deadlines", [])
    if deadlines:
        for deadline in deadlines:
            if deadline["expiry_status"] == "EXPIRED":
                line = (
                    f"Deadline {deadline['id']}: EXPIRED; its scheduled time passed "
                    f"{deadline['overdue_s']:.3f} seconds ago."
                )
                line += (
                    " The deadline event was delivered."
                    if deadline["event_status"] == "delivered"
                    else " The deadline event was not yet delivered."
                )
                lines.append(line)
            else:
                lines.append(
                    f"Deadline {deadline['id']}: PENDING; "
                    f"{deadline['remaining_s']:.3f} seconds remain. "
                    "The deadline event was not yet delivered."
                )
    else:
        lines.append("No deadlines are scheduled.")
    events = snapshot.get("events", [])
    if events:
        for event in events:
            payload = json.dumps(event["payload"], sort_keys=True, separators=(",", ":"))
            lines.append(
                f"Recent event {event['name']} was observed {event['age_s']:.3f} seconds ago; "
                f"payload={payload}."
            )
    else:
        lines.append("No recent events are recorded.")
    return "\n".join(lines)
