import time
from types import SimpleNamespace

import pytest

from temporal_bridge.live_worker import ProcessModelWorker, _handle_request


def test_generate_reports_actual_yields_with_identity_and_complete_text():
    class Runtime:
        def prepare_generation(self, prompt, max_tokens):
            assert (prompt, max_tokens) == ("question", 12)
            return iter([
                SimpleNamespace(text="first", token=42, generation_tokens=1, finish_reason=None),
                SimpleNamespace(text=" second", token=43, generation_tokens=2, finish_reason="stop"),
            ]), 5

    events = []
    _handle_request({"kind": "generate", "request_id": "run1/request9", "prompt": "question",
                     "max_tokens": 12}, Runtime(), events.append)
    assert [e["type"] for e in events] == ["started", "first_token", "result"]
    assert all(e["request_id"] == "run1/request9" for e in events)
    assert events[1]["token"] == 42
    assert events[2]["text"] == "first second"
    assert events[2]["input_tokens"] == 5
    assert events[2]["output_tokens"] == 2
    assert events[0]["monotonic"] <= events[1]["monotonic"] <= events[2]["monotonic"]


def test_worker_errors_are_structured_and_retain_request_identity():
    class Runtime:
        def prepare_generation(self, *args):
            raise RuntimeError("backend unavailable")

    events = []
    _handle_request({"kind": "generate", "request_id": "run/error", "prompt": "q",
                     "max_tokens": 4}, Runtime(), events.append)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["request_id"] == "run/error"
    assert "backend unavailable" in events[0]["error"]


def test_forecast_preserves_request_identity_and_backend_result():
    class Runtime:
        def forecast(self, context, horizon):
            assert context == [1.0, 2.0] and horizon == 2
            return {"point": [3, 4], "q10": [2, 3], "q90": [4, 5], "elapsed_s": .02}

    events = []
    _handle_request({"kind": "forecast", "request_id": "series/1", "context": [1.0, 2.0],
                     "horizon": 2}, Runtime(), events.append)
    assert [event["type"] for event in events] == ["started", "result"]
    assert events[-1]["request_id"] == "series/1"
    assert events[-1]["point"] == [3, 4]


def test_warmup_only_loads_models_without_forecasting():
    class Runtime:
        def warmup(self):
            return {"timesfm_revision": "fixed-revision"}

    events = []
    _handle_request({"kind": "warmup", "request_id": "warm"}, Runtime(), events.append)
    assert events[-1]["metadata"]["timesfm_revision"] == "fixed-revision"


def test_spawned_cpu_health_probe_and_clean_close():
    worker = ProcessModelWorker("unused", use_tsfm=False)
    try:
        worker.start()
        worker.submit("health-1", "", kind="health")
        deadline = time.monotonic() + 8
        messages = []
        while time.monotonic() < deadline:
            messages.extend(worker.poll())
            if any(item["type"] == "result" for item in messages):
                break
            time.sleep(0.01)
        assert any(item["type"] == "result" and item["request_id"] == "health-1"
                   for item in messages)
    finally:
        worker.close()
    assert not worker.is_alive()


def test_worker_rejects_invalid_messages_before_start():
    worker = ProcessModelWorker("unused")
    with pytest.raises(RuntimeError):
        worker.submit("id", "hello")
    worker.close()


def test_submitted_forecast_copies_context_and_checks_finiteness():
    worker = ProcessModelWorker("unused")
    worker._process = SimpleNamespace(is_alive=lambda: True)
    captured = []
    worker._requests = SimpleNamespace(put=captured.append)
    with pytest.raises(ValueError, match="finite"):
        worker.submit("bad", kind="forecast", context=[1.0, float("nan")])
    context = [1, 2]
    worker.submit("good", kind="forecast", context=context)
    context[0] = 99
    assert captured[0]["context"] == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="outstanding"):
        worker.submit("another", kind="health")
