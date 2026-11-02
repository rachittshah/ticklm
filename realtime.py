"""Live clock, event intake, stale-answer guard, and optional predictive wakeups."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from temporal_bridge.core import baseline_forecast
from temporal_bridge.live_worker import ProcessModelWorker
from temporal_bridge.realtime_state import TemporalState, render_clock

DEFAULT_MODEL = "/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictive_delay(point, current, sample_s=0.1, change=1.0, maximum=3.0):
    """Use a point trajectory for a bounded wake hint, never a crossing probability."""
    values = np.asarray(point, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional point forecast required")
    if sample_s <= 0 or change <= 0 or maximum < 0.25:
        raise ValueError("invalid predictive-wake parameters")
    crossing = np.flatnonzero(np.abs(values - current) >= change)
    proposal = (int(crossing[0]) + 1) * sample_s if len(crossing) else maximum
    return max(0.25, min(maximum, proposal))


def regular_history(samples, count=128, sample_s=.1, max_gap_s=.3):
    """Causally resample actual arrivals; refuse a gap or insufficient live history."""
    if len(samples) < count:
        return None
    values = np.array(samples, dtype=float)
    end = values[-1, 0]
    grid = end - np.arange(count - 1, -1, -1) * sample_s
    before = np.flatnonzero(values[:, 0] <= grid[0])
    if not len(before):
        return None
    selected = values[before[-1]:]
    if np.any(np.diff(selected[:, 0]) > max_gap_s):
        return None
    return np.interp(grid, selected[:, 0], selected[:, 1])


class LiveRuntime:
    def __init__(self, run_dir, model_path, use_tsfm=True):
        self.out = Path(run_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = (self.out / "trace.jsonl").open("x")
        self.worker = ProcessModelWorker(model_path, use_tsfm=use_tsfm)
        self.state = None
        self.active = None
        self.tickets = {}
        self.forecast_tickets = {}
        self.results = {}
        self.messages = []
        self.samples = deque(maxlen=512)
        self.facts = {"mode": "NORMAL"}
        self.race = {}
        self.deadline_events = []
        self.loop_intervals = []
        self.calls = 0
        self.use_tsfm = use_tsfm

    def log(self, kind, **fields):
        record = {"kind": kind, "recorded_mono": time.monotonic(), **fields}
        self.trace.write(json.dumps(record, allow_nan=False) + "\n")
        self.trace.flush()
        return record

    async def start(self):
        self.worker.start()
        self.worker.submit("warmup", kind="warmup")
        started = time.monotonic()
        while True:
            for message in self.worker.poll():
                self.log("worker", **message)
                if message["type"] == "error":
                    raise RuntimeError(message["error"])
                if message["type"] == "result":
                    metadata = dict(message.get("metadata", {}))
                    if metadata.get("timesfm_checkpoint"):
                        metadata["timesfm_weight_sha256"] = file_sha256(metadata["timesfm_checkpoint"])
                    self.state = TemporalState(self.out.name)
                    self.log("run_started", snapshot=self.state.snapshot(),
                             warmup_wall_s=time.monotonic() - started,
                             metadata=metadata)
                    return
            if time.monotonic() - started > 90:
                raise TimeoutError("model warmup exceeded 90 seconds")
            await asyncio.sleep(.01)

    def ask(self, name, question, *, snapshot=None, ttl_s=6, max_tokens=48, control=False,
            bind_observation=False):
        if self.active is not None:
            raise RuntimeError("Only one model task may run at a time")
        request_id = f"{self.state.run_id}:{name}"
        ticket = self.state.issue_request(request_id, ttl_s=ttl_s,
                                          bind_observation=bind_observation)
        context = ticket if snapshot is None else snapshot
        prompt = ("The following is your temporal context, supplied by the live runtime.\n"
                  + render_clock(context) + "\nCurrent facts=" + json.dumps(self.facts)
                  + "\n\n" + question)
        self.tickets[request_id] = (ticket, name, control)
        self.worker.submit(request_id, prompt, max_tokens=max_tokens)
        self.active = request_id
        self.calls += 1
        self.log("request", request_id=request_id, name=name, prompt=prompt,
                 ticket=ticket, context_snapshot=context, control_only=control)

    def forecast(self, context, horizon=64, name="forecast"):
        if self.active is not None:
            raise RuntimeError("Only one model task may run at a time")
        request_id = f"{self.state.run_id}:{name}"
        ticket = self.state.issue_request(request_id, ttl_s=min(5, horizon * .1))
        self.forecast_tickets[request_id] = (ticket, self.samples[-1][0], float(context[-1]))
        self.active = request_id
        self.worker.submit(request_id, kind="forecast", context=context, horizon=horizon)
        self.log("forecast_requested", request_id=request_id, context=list(map(float, context)),
                 actual_samples_seen=self.state.data_revision, sample_interval_s=.1,
                 horizon=horizon, snapshot=self.state.snapshot())

    def service(self, *, inject_race=False):
        for fired in self.state.tick():
            self.deadline_events.append(fired)
            self.log("deadline_fired", **fired)
        for message in self.worker.poll():
            self.messages.append(message)
            self.log("worker", **message)
            request_id = message["request_id"]
            if (inject_race and request_id.endswith(":race")
                    and message["type"] == "first_token" and not self.race):
                self.facts["mode"] = "ALARM"
                self.state.event("mode_changed", {"mode": "ALARM"})
                alarm_at = time.monotonic()
                deadline = self.state.set_deadline("urgent", .25)
                self.race = {"first_token_mono": message["monotonic"],
                             "alarm_mono": alarm_at, "urgent_due_mono": deadline["expires_mono"]}
                self.log("alarm_injected", request_id=request_id, **self.race)
            if message["type"] in {"result", "error"}:
                self.active = None
                name = request_id.split(":")[-1]
                if message["type"] == "error":
                    self.results[name] = message
                    raise RuntimeError(message["error"])
                result = dict(message)
                if request_id in self.tickets:
                    ticket, name, control = self.tickets[request_id]
                    valid, reason = self.state.validate_result(ticket)
                    result.update({"valid": valid, "validity_reason": reason, "control_only": control})
                    result["snapshot_elapsed_s"] = ticket["elapsed_s"]
                    result["answer_as_of_mono"] = ticket["issued_mono"]
                    result["answer_as_of_utc"] = ticket["wall_utc"]
                    result["live_publication_clock"] = self.state.snapshot()
                    self.log("control_result" if control else "published" if valid else "rejected",
                             request_id=request_id, validity_reason=reason, text=message["text"],
                             answer_as_of_mono=ticket["issued_mono"],
                             publication_snapshot=self.state.snapshot())
                elif request_id in self.forecast_tickets:
                    ticket, context_end, last_value = self.forecast_tickets[request_id]
                    valid, reason = self.state.validate_result(ticket)
                    result.update({"valid": valid, "validity_reason": reason,
                                   "context_end_mono": context_end,
                                   "context_last_value": last_value,
                                   "forecast_origin_data_revision": ticket["data_revision"]})
                    self.log("forecast_accepted" if valid else "forecast_rejected",
                             request_id=request_id, reason=reason, context_end_mono=context_end)
                self.results[name] = result

    def close(self):
        self.worker.close()
        self.trace.close()


def exact_reply(result, answer):
    return result.get("text", "").strip().rstrip(".").strip().upper() == answer.upper()


def numeric_reply(result):
    text = result.get("text", "").strip()
    return float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else None


async def self_test(args):
    runtime = LiveRuntime(args.run_dir, args.model, use_tsfm=not args.clock_only)
    try:
        await runtime.start()
        state = runtime.state
        state.event("mode", {"value": "NORMAL"})
        state.set_deadline("probe", 3)
        frozen = state.snapshot()
        next_sample = 0.0
        quiet_start = None
        quiet_ticket = None
        quiet_expiry = None
        quiet_snapshot = None
        forecast_context = None
        wake_at = None
        wake_record = None
        last_loop = time.monotonic()
        while True:
            now = time.monotonic()
            runtime.loop_intervals.append(now - last_loop)
            last_loop = now
            elapsed = now - state.start_mono
            quiet = quiet_start is not None and quiet_start <= elapsed < quiet_start + 3
            if elapsed >= next_sample:
                next_sample = elapsed + .1
                if not quiet:
                    # Synthetic values are produced NOW, never backfilled or preloaded.
                    value = 50 + .25 * elapsed + 3 * math.sin(elapsed / 4)
                    if runtime.facts["mode"] == "ALARM":
                        value += 8
                    state.observe(value)
                    at = time.monotonic()
                    runtime.samples.append((at, value))
                    runtime.log("sample", value=value, elapsed_s=elapsed, sample_mono=at,
                                data_revision=state.data_revision, source="simulated_live_sensor")
            runtime.service(inject_race=True)
            if runtime.active is None:
                completed = runtime.results
                if elapsed >= 4 and "frozen_clock" not in completed:
                    runtime.ask("frozen_clock", "Report the status of the probe deadline in the temporal context. "
                                "Reply with exactly EXPIRED or PENDING.",
                                snapshot=frozen, max_tokens=12, control=True)
                elif "frozen_clock" in completed and "fresh_clock" not in completed:
                    runtime.ask("fresh_clock", "Report the status of the probe deadline in the temporal context. "
                                "Reply with exactly EXPIRED or PENDING.",
                                max_tokens=12)
                elif "fresh_clock" in completed and "elapsed_number" not in completed:
                    runtime.ask("elapsed_number", "How many seconds has this run been active, "
                                "as of this clock snapshot? Reply with only the decimal number, no units.",
                                max_tokens=12)
                elif "elapsed_number" in completed and "race" not in completed:
                    runtime.ask("race", "Describe the current NORMAL mode and a monitoring plan in 12 numbered sentences. "
                                "Use only the supplied situation. Produce all twelve sentences.",
                                ttl_s=15, max_tokens=200)
                elif "race" in completed and "refresh" not in completed:
                    runtime.ask("refresh", "What is the current mode, and is the urgent deadline expired? "
                                "Reply exactly MODE; DEADLINE_STATUS using ALARM or NORMAL and EXPIRED or PENDING.",
                                max_tokens=20)
                elif "refresh" in completed and quiet_start is None:
                    quiet_start = elapsed
                    state.event("source_paused", {"duration_s": 3})
                    quiet_ticket = state.issue_request(f"{state.run_id}:quiet-expiry", ttl_s=.3)
                    runtime.log("source_pause_started", elapsed_s=elapsed, duration_s=3)
                elif (quiet_start is not None and elapsed >= quiet_start + 1.6
                      and "freshness" not in completed):
                    quiet_expiry = state.validate_result(quiet_ticket)
                    quiet_snapshot = state.snapshot()
                    runtime.log("quiet_expiry_check", valid=quiet_expiry[0], reason=quiet_expiry[1],
                                snapshot=quiet_snapshot)
                    runtime.ask("freshness", "An observation older than 1 second is STALE. "
                                "Is the latest observation STALE or FRESH? Reply with one word.",
                                max_tokens=12, bind_observation=True)
                elif "freshness" in completed and args.clock_only:
                    break
                elif "freshness" in completed and "forecast" not in completed:
                    context = regular_history(runtime.samples)
                    if context is not None and elapsed >= quiet_start + 16:
                        forecast_context = context
                        runtime.forecast(context)
                elif "forecast" in completed and wake_at is None:
                    forecast = completed["forecast"]
                    if not forecast["valid"]:
                        raise RuntimeError("Live forecast lost validity before scheduling")
                    delay = predictive_delay(forecast["point"], float(forecast_context[-1]))
                    cheap = baseline_forecast(forecast_context, 64)
                    cheap_delay = predictive_delay(cheap["point"], float(forecast_context[-1]))
                    wake_at = max(time.monotonic(), forecast["context_end_mono"] + delay)
                    wake_record = {"timesfm_delay_s": delay, "cheap_delay_s": cheap_delay,
                                   "timesfm_forecast_s": forecast["elapsed_s"],
                                   "cheap_forecast_s": cheap["elapsed_s"],
                                   "cheap_model": cheap["model_id"], "scheduled_mono": wake_at,
                                   "maximum_silence_s": 3, "minimum_delay_s": .25,
                                   "forecast_origin_mono": forecast["context_end_mono"],
                                   "claim": "point-forecast heuristic, not an event probability"}
                    runtime.log("predictive_wake_scheduled", **wake_record)
                elif wake_at is not None and now >= wake_at and "predictive_wake" not in completed:
                    runtime.log("predictive_wake_fired", scheduled_mono=wake_at,
                                fired_mono=time.monotonic())
                    runtime.ask("predictive_wake", "You were awakened by a forecast-based timer. "
                                "Briefly report the elapsed time, whether the latest observation is fresh "
                                "within 1 second, and the current mode. Do not claim future events occurred.",
                                max_tokens=80)
                elif "predictive_wake" in completed:
                    break
            if elapsed > 70:
                raise TimeoutError("live trial did not complete within 70 seconds")
            await asyncio.sleep(.01)
        results = runtime.results
        race_finish = results["race"]["monotonic"]
        urgent = next((d for d in runtime.deadline_events if d["deadline_id"] == "urgent"), None)
        checks = {
            "actual_wait_before_clock_query": results["fresh_clock"]["monotonic"] - state.start_mono >= 4,
            "fresh_clock_answer_expired": exact_reply(results["fresh_clock"], "EXPIRED"),
            "model_reports_elapsed_seconds": numeric_reply(results["elapsed_number"]) is not None and
                abs(numeric_reply(results["elapsed_number"]) - results["elapsed_number"]["snapshot_elapsed_s"]) <= .15,
            "event_arrived_during_real_generation": bool(runtime.race) and
                runtime.race["first_token_mono"] < runtime.race["alarm_mono"] < race_finish,
            "urgent_deadline_fired_during_generation": urgent is not None and
                runtime.race["alarm_mono"] < urgent["fired_at_mono"] < race_finish,
            "stale_inflight_answer_rejected": results["race"].get("valid") is False,
            "refresh_answer_uses_new_alarm": exact_reply(results["refresh"], "ALARM; EXPIRED") and
                results["refresh"].get("valid") is True,
            "clock_expires_without_semantic_event": quiet_expiry == (False, "expired"),
            "observation_really_older_than_one_second": quiet_snapshot["observation"]["age_s"] > 1,
            "model_identifies_stale_observation": exact_reply(results["freshness"], "STALE") and
                results["freshness"].get("valid") is True,
        }
        if not args.clock_only:
            checks.update({"tsfm_received_128_live_samples": len(forecast_context) == 128,
                           "predictive_wake_executed": "predictive_wake" in results})
        summary = {
            "run_id": state.run_id, "mode": "clock_only" if args.clock_only else "clock_plus_tsfm",
            "checks": checks, "all_checks_passed": all(checks.values()),
            "elapsed_wall_s": time.monotonic() - state.start_mono,
            "samples_ingested": state.data_revision, "llm_calls": runtime.calls,
            "frozen_context_control": results["frozen_clock"],
            "fresh_context_response": results["fresh_clock"],
            "results": results, "race": runtime.race,
            "deadlines": runtime.deadline_events, "predictive_wake": wake_record,
            "loop_interval_p50_ms": float(np.median(runtime.loop_intervals) * 1000),
            "loop_interval_p95_ms": float(np.quantile(runtime.loop_intervals, .95) * 1000),
            "loop_interval_max_ms": float(max(runtime.loop_intervals) * 1000),
            "limits": ["A scripted live trial, not a benchmark of general temporal cognition.",
                       "Sensor values are simulated at real wall time; they are not physical-device readings.",
                       "Ordinary samples update data_revision; semantic events and expiry govern output invalidation.",
                       "Clock and deadline behavior comes from runtime; TimesFM only supplies a wake-time hint.",
                       "No hard real-time scheduling guarantee; timings reflect this observed run."],
        }
        with (runtime.out / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        print(json.dumps(summary, indent=2), flush=True)
        if not summary["all_checks_passed"]:
            raise RuntimeError("One or more live checks failed; inspect retained trace")
    finally:
        runtime.close()


def _read_event_chunk(path, offset, identity):
    with Path(path).open() as stream:
        stat = os.fstat(stream.fileno())
        current_identity = (stat.st_dev, stat.st_ino)
        reset = identity is not None and (identity != current_identity or offset > stat.st_size)
        if offset is None or reset:
            stream.seek(0, os.SEEK_END)
            return "", stream.tell(), current_identity, reset
        stream.seek(offset)
        chunk = stream.read(65536)
        return chunk, stream.tell(), current_identity, False


async def watch(args):
    """Monitor actual file arrivals or the machine's load while maintaining deadlines."""
    runtime = LiveRuntime(args.run_dir, args.model, use_tsfm=not args.clock_only)
    offset, file_identity, buffer, last_announced, next_report = None, None, "", -1, 0.0
    next_forecast, forecasts_made, last_value_notice = 0.0, 0, None

    def observe(value):
        nonlocal last_value_notice
        runtime.state.observe(value)
        numeric = float(value)
        runtime.samples.append((time.monotonic(), numeric))
        if last_value_notice is None:
            last_value_notice = numeric
        elif abs(numeric - last_value_notice) >= args.change_threshold:
            runtime.state.event("value_changed", {"from": last_value_notice, "to": numeric})
            last_value_notice = numeric
    try:
        if args.events:
            _, offset, file_identity, _ = await asyncio.to_thread(
                _read_event_chunk, args.events, None, None)
            runtime.log("source_opened", source=args.events, offset=offset,
                        policy="ignore existing backlog; retain bytes appended during model warmup")
        await runtime.start()
        runtime.facts = {"observation_source": "live file arrivals" if args.events
                         else "system one-minute load average"}
        runtime.log("watch_ready", snapshot=runtime.state.snapshot())
        if args.deadline_seconds is not None:
            runtime.state.set_deadline("user", args.deadline_seconds)
        while time.monotonic() - runtime.state.start_mono < args.seconds:
            if args.events:
                chunk, offset, file_identity, reset = await asyncio.to_thread(
                    _read_event_chunk, args.events, offset, file_identity)
                if reset:
                    buffer = ""
                    runtime.state.event("source_reset", {"reason": "rotation_or_truncation"})
                    runtime.log("source_reset", source=args.events,
                                policy="skip replacement backlog; records written before reset detection may be lost")
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    try:
                        item = json.loads(line)
                        allowed = {"sample": {"type", "value"}, "event": {"type", "name", "payload"},
                                   "deadline": {"type", "id", "after_s"}}
                        if not isinstance(item, dict) or set(item) - allowed.get(item.get("type"), set()):
                            raise ValueError("Unsupported fields; this source uses live arrival time")
                        if item["type"] == "sample":
                            observe(item["value"])
                        elif item["type"] == "event":
                            runtime.state.event(item["name"], item.get("payload"))
                            runtime.facts[item["name"]] = item.get("payload", {})
                        elif item["type"] == "deadline":
                            runtime.state.set_deadline(item["id"], item["after_s"])
                        else:
                            raise ValueError("unknown event type")
                        runtime.log("input", item=item)
                    except (ValueError, KeyError, TypeError) as error:
                        runtime.log("invalid_input", error=str(error), line=line)
            else:
                observe(os.getloadavg()[0])
            runtime.service()
            snapshot = runtime.state.snapshot()
            if runtime.active is None and (snapshot["revision"] != last_announced or time.monotonic() >= next_report):
                runtime.ask(f"watch-{runtime.calls}", "As of this snapshot, report what changed, elapsed time, "
                            "observation age, and any expired or upcoming deadline. "
                            "Use the supplied clock; be brief.", max_tokens=80)
                last_announced = snapshot["revision"]
                next_report = time.monotonic() + 5
                print(render_clock(snapshot), flush=True)
            elif (not args.clock_only and runtime.active is None
                  and time.monotonic() >= next_forecast):
                context = regular_history(runtime.samples)
                if context is not None:
                    runtime.forecast(context, name=f"forecast-{forecasts_made}")
                    forecasts_made += 1
                    next_forecast = time.monotonic() + 5
            for name in list(runtime.results):
                response = runtime.results.pop(name)
                if name.startswith("forecast-"):
                    if not response["valid"]:
                        continue
                    delay = predictive_delay(response["point"], response["context_last_value"],
                                             change=args.change_threshold)
                    next_report = min(next_report, max(time.monotonic(), response["context_end_mono"] + delay))
                    runtime.log("predictive_wake_scheduled", delay_s=delay,
                                request_id=response["request_id"], due_mono=next_report)
                else:
                    publication = {"name": name, "published": response.get("valid"),
                                   "reason": response.get("validity_reason")}
                    if response.get("valid"):
                        publication.update({"text": response.get("text"),
                                            "narrative_as_of_utc": response.get("answer_as_of_utc"),
                                            "live_clock": response.get("live_publication_clock")})
                    print(json.dumps(publication), flush=True)
            await asyncio.sleep(.05)
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "watch"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--clock-only", action="store_true", help="Skip TimesFM; clocks/events still work")
    parser.add_argument("--events", help="Append-only JSONL of sample/event/deadline objects")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--change-threshold", type=float, default=1.0)
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.seconds):
        parser.error("seconds must be finite and positive")
    if args.change_threshold <= 0 or not math.isfinite(args.change_threshold):
        parser.error("change-threshold must be finite and positive")
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), *Path("src/temporal_bridge").glob("*.py")]
    model_files = [p for p in Path(args.model).iterdir()
                   if p.suffix in {".safetensors", ".json", ".txt", ".model"}]
    tsfm_source = Path(importlib.util.find_spec("mlx_tsfm").origin).parent
    with (out / "manifest.json").open("x") as stream:
        json.dump({"arguments": vars(args), "platform": platform.platform(),
                   "source_sha256": {str(p): file_sha256(p) for p in source_paths},
                   "model_file_sha256": {p.name: file_sha256(p) for p in model_files},
                   "tsfm_source_sha256": {str(p.relative_to(tsfm_source)): file_sha256(p)
                                          for p in tsfm_source.rglob("*.py")},
                   "dependencies": {d.metadata["Name"]: d.version
                                    for d in importlib.metadata.distributions()}}, stream, indent=2)
    asyncio.run(self_test(args) if args.mode == "self-test" else watch(args))


if __name__ == "__main__":
    main()
