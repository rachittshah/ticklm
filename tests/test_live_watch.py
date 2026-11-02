"""Opt-in integration through the real public watcher and local model process."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _records(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # The running process may still be writing the final line.
    return rows


@pytest.mark.skipif(os.environ.get("TEMPORAL_LIVE_TEST") != "1", reason="opt-in real model test")
def test_live_file_events_and_deadline_arrive_while_model_generates(tmp_path):
    out = Path(os.environ.get("TEMPORAL_WATCH_ARTIFACT_DIR", str(tmp_path / "watch")))
    source = tmp_path / "events.jsonl"
    source.write_text('{"type":"sample","value":12345}\n')
    log = tmp_path / "watch.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "realtime.py", "watch", "--clock-only", "--seconds", "6",
             "--events", str(source), "--run-dir", str(out)],
            stdout=stream, stderr=subprocess.STDOUT,
        )
        try:
            timeout = time.monotonic() + 45
            first = None
            while time.monotonic() < timeout and process.poll() is None:
                first = next((r for r in _records(out / "trace.jsonl")
                              if r["kind"] == "worker" and r.get("type") == "first_token"), None)
                if first:
                    break
                time.sleep(.02)
            assert first is not None, log.read_text()
            with source.open("a") as events:
                events.write('{"type":"sample","value":42.5}\n')
                events.write('{"type":"event","name":"door","payload":{"open":true}}\n')
                events.write('{"type":"deadline","id":"file-deadline","after_s":0.25}\n')
            assert process.wait(timeout=30) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    (out / "stdout.log").write_text(log.read_text())
    rows = _records(out / "trace.jsonl")
    inputs = [r for r in rows if r["kind"] == "input"]
    assert len(inputs) == 3
    assert all(r["item"].get("value") != 12345 for r in inputs)
    due = [r for r in rows if r["kind"] == "deadline_fired"]
    assert len(due) == 1 and due[0]["deadline_id"] == "file-deadline"
    assert due[0]["lateness_s"] < .5
    completion = next(r for r in rows if r["kind"] == "worker" and r.get("type") == "result"
                      and r.get("request_id") == first["request_id"])
    assert first["monotonic"] < inputs[0]["recorded_mono"] < completion["monotonic"]
    assert any(r["kind"] == "rejected" and r["request_id"] == first["request_id"] for r in rows)
    assert any(r["kind"] == "published" for r in rows)
    for line in log.read_text().splitlines():
        if line.startswith('{"name":'):
            displayed = json.loads(line)
            if not displayed["published"]:
                assert "text" not in displayed
