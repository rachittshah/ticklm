"""Exercise adapted decisions against actual elapsed time and an in-flight event."""

from __future__ import annotations

import argparse
import hashlib
import threading
import time
from pathlib import Path

from mlx_lm import load

from .surgery import (
    LABELS,
    atomic_json,
    base_identity,
    context_limit,
    evaluate_cases,
    load_adapter,
    reserve_output,
    runtime_provenance,
    write_records,
)
from .surgery_data import deterministic_answer, render_prompt


def make_case(family, state, choices, case_id):
    """Build a live snapshot; only the rendered causal state reaches the model."""
    case = {"id": case_id, "family": family, "split": "live", "state": dict(state),
            "choices": dict(choices)}
    case["prompt"] = render_prompt("test", family, state, choices)
    case["answer"] = deterministic_answer(case)
    case["correct_action"] = choices[case["answer"]]
    return case


def release_assessment(case, prediction, current_state):
    """Separate snapshot accuracy from a deterministic check of current action truth."""
    if prediction not in LABELS:
        raise ValueError("prediction must be a choice letter")
    snapshot_correct = prediction == deterministic_answer(case)
    current_answer = deterministic_answer({**case, "state": current_state})
    current_correct = prediction == current_answer
    return {"snapshot_correct": snapshot_correct, "current_correct": current_correct,
            "current_answer": current_answer,
            "publishable": snapshot_correct and current_correct,
            "guard_uses_deterministic_reference": True}


def run_live(model, tokenizer, limit, *, threshold_s=1.5, cancel_delay_s=0.005):
    if threshold_s <= 0 or cancel_delay_s < 0:
        raise ValueError("threshold must be positive and cancellation delay nonnegative")
    silence_choices = {"A": "alert", "B": "wait", "C": "restart", "D": "cancel"}
    relevance_choices = {"A": "continue", "B": "replan", "C": "stop", "D": "wait"}
    warm = make_case("silence", {"now": 0.0, "last_event": 0.0, "threshold": threshold_s},
                     silence_choices, "warmup")
    evaluate_cases(model, tokenizer, [warm], limit, 1)
    origin = time.monotonic()
    lock = threading.RLock()
    latest_event = 0.0
    records = []
    events = []

    def elapsed():
        return time.monotonic() - origin

    def silence_state():
        with lock:
            return {"now": elapsed(), "last_event": latest_event, "threshold": threshold_s}

    def predict(case, current_state):
        started = elapsed()
        result, _ = evaluate_cases(model, tokenizer, [case], limit, 1)
        completed = elapsed()
        # This observation is taken before waiting for any producer thread.
        state_at_release = current_state()
        record = {**result[0], "started_elapsed_s": started, "completed_elapsed_s": completed,
                  "latency_s": completed - started, "state_at_release": state_at_release,
                  **release_assessment(case, result[0]["prediction"], state_at_release)}
        records.append(record)
        return record

    predict(make_case("silence", silence_state(), silence_choices, "before-silence"), silence_state)
    remaining = threshold_s + 0.15 - elapsed()
    if remaining > 0:
        time.sleep(remaining)
    predict(make_case("silence", silence_state(), silence_choices, "after-silence"), silence_state)
    with lock:
        latest_event = elapsed()
        events.append({"type": "observation_arrived", "elapsed_s": latest_event})
    predict(make_case("silence", silence_state(), silence_choices, "after-new-event"), silence_state)

    live_plan = {"event": "cancel", "plan_id": "P", "target_id": "Q",
                 "plan_at": 0.0, "event_at": elapsed()}

    def plan_state():
        with lock:
            return dict(live_plan)

    stale_case = make_case("relevance", plan_state(), relevance_choices, "cancel-during-inference")

    def cancel():
        time.sleep(cancel_delay_s)
        with lock:
            live_plan.update(target_id="P", event_at=elapsed())
            events.append({"type": "plan_cancelled", "elapsed_s": live_plan["event_at"]})

    producer = threading.Thread(target=cancel, name="ticklm-surgery-event", daemon=True)
    producer.start()
    stale_record = predict(stale_case, plan_state)
    producer.join(timeout=2.0)
    if producer.is_alive():
        raise RuntimeError("cancellation producer did not finish")
    cancel_at = next(event["elapsed_s"] for event in events if event["type"] == "plan_cancelled")
    stale_record["event_during_inference"] = (
        stale_record["started_elapsed_s"] <= cancel_at <= stale_record["completed_elapsed_s"])
    predict(make_case("relevance", plan_state(), relevance_choices, "after-cancellation"), plan_state)
    summary = {
        "clock": "time.monotonic", "duration_s": elapsed(), "threshold_s": threshold_s,
        "cancel_delay_s": cancel_delay_s, "count": len(records),
        "snapshot_correct": sum(record["snapshot_correct"] for record in records),
        "publishable": sum(record["publishable"] for record in records),
        "cancellation_during_inference": stale_record["event_during_inference"],
        "stale_cancellation_answer_rejected": not stale_record["publishable"],
        "cancellation_snapshot_answer_correct": stale_record["snapshot_correct"],
        "events": events,
        "scope": "Five scripted snapshots; external clock and reference guard; no speed or generalization claim.",
    }
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-s", type=float, default=1.5)
    parser.add_argument("--cancel-delay-s", type=float, default=0.005)
    args = parser.parse_args(argv)
    output = Path(args.output)
    reserve_output(output)
    identity = base_identity(args.model)
    model, tokenizer = load(args.model)
    counts = load_adapter(model, args.adapter, identity, args.revision) if args.adapter else None
    atomic_json(output / "run.json", {
        "args": vars(args), "base_hashes": identity, "adapter_parameters": counts,
        "runtime": runtime_provenance(),
        "live_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    records, summary = run_live(model, tokenizer, context_limit(args.model),
                                threshold_s=args.threshold_s, cancel_delay_s=args.cancel_delay_s)
    write_records(output / "cases.jsonl", records)
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
