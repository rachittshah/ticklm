import time
from io import StringIO
from types import SimpleNamespace

import pytest

from temporal_bridge.codex_backend import CodexBackend


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_rejects_nonpositive_or_nonfinite_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        CodexBackend(timeout_s=timeout)


def test_warmup_reports_requested_effort_and_disables_execution_paths(monkeypatch):
    backend = CodexBackend(effort="xhigh")
    command = []
    process = SimpleNamespace(stdin=StringIO(), stdout=StringIO(), poll=lambda: None)
    monkeypatch.setattr("temporal_bridge.codex_backend.subprocess.run", lambda *a, **k: (
        SimpleNamespace(stdout="codex-cli test")
    ))
    monkeypatch.setattr("temporal_bridge.codex_backend.subprocess.Popen", lambda args, **kwargs: (
        command.extend(args) or process
    ))
    monkeypatch.setattr(backend, "close", lambda: None)
    monkeypatch.setattr(backend, "_request", lambda method, params, deadline: (
        {"model": "gpt-6-astra", "modelProvider": "openai"} if method == "thread/start" else {}
    ))
    metadata = backend.warmup()
    assert metadata["requested_effort"] == "xhigh"
    assert "effort" not in metadata
    assert metadata["resolved_model"] == "gpt-6-astra"
    disabled = {command[i + 1] for i, arg in enumerate(command) if arg == "--disable"}
    assert {"shell_tool", "unified_exec", "code_mode", "code_mode_host"} <= disabled
    backend._temp.cleanup()


def test_generate_filters_identity_and_reports_visible_text(monkeypatch):
    backend = CodexBackend()
    calls = []
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_request", lambda method, params, deadline: (
        calls.append((method, params)) or ({"thread": {"id": "t"}, "model": "gpt-6-astra"}
        if method == "thread/start" else {"turn": {"id": "u"}})
    ))
    messages = iter([
        {"method": "item/agentMessage/delta", "params": {"threadId": "other", "turnId": "u", "delta": "wrong"}},
        {"method": "item/agentMessage/delta", "params": {"threadId": "t", "turnId": "other", "delta": "wrong"}},
        {"method": "item/started", "params": {"threadId": "t", "turnId": "u", "item": {"type": "agentMessage"}}},
        {"method": "item/completed", "params": {"threadId": "t", "turnId": "u", "item": {"id": "c", "type": "agentMessage", "phase": "commentary", "text": "wrong"}}},
        {"method": "item/agentMessage/delta", "params": {"threadId": "t", "turnId": "u", "delta": "Hello", "itemId": "i"}},
        {"method": "item/completed", "params": {"threadId": "t", "turnId": "u", "item": {"id": "i", "type": "agentMessage", "phase": "final_answer", "text": "Hello"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "t", "turnId": "u", "tokenUsage": {"last": {"inputTokens": 10, "outputTokens": 2}}}},
        {"method": "turn/completed", "params": {"threadId": "t", "turn": {"id": "u", "status": "completed"}}},
    ])
    monkeypatch.setattr(backend, "_next", lambda deadline: next(messages))
    events = []
    backend.generate("unchanged", 12, lambda kind, **data: events.append((kind, data)))
    assert [x[0] for x in events] == ["started", "first_token", "delta", "result"]
    assert events[1][1]["evidence"] == "visible_text_delta"
    assert events[-1][1]["text"] == "Hello"
    assert events[-1][1]["input_tokens"] == 10
    assert events[-1][1]["output_tokens"] == 2
    assert calls[0][1]["ephemeral"] is True
    assert calls[1][1]["input"][0]["text"] == "unchanged"


def test_failed_turn_is_not_success(monkeypatch):
    backend = CodexBackend()
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_request", lambda method, params, deadline: (
        {"thread": {"id": "t"}, "model": "gpt-6-astra"} if method == "thread/start"
        else {"turn": {"id": "u"}}
    ))
    monkeypatch.setattr(backend, "_next", lambda deadline: {
        "method": "turn/completed", "params": {"threadId": "t", "turn": {"id": "u", "status": "failed"}}
    })
    with pytest.raises(RuntimeError, match="failed"):
        backend.generate("test", 10, lambda *args, **kwargs: None)


def test_deadline_and_closed_transport():
    backend = CodexBackend()
    with pytest.raises(TimeoutError, match="timed out"):
        backend._next(time.monotonic() - 1)
    backend._queue.put(None)
    with pytest.raises(RuntimeError, match="transport closed"):
        backend._next(time.monotonic() + 1)


def test_request_buffers_notifications_before_response(monkeypatch):
    backend = CodexBackend()
    sent = []
    monkeypatch.setattr(backend, "_send", sent.append)
    notification = {"method": "item/started", "params": {"threadId": "t"}}
    backend._queue.put(notification)
    backend._queue.put({"id": 1, "result": {"thread": {"id": "t"}}})
    assert backend._request("thread/start", {}, time.monotonic() + 1)["thread"]["id"] == "t"
    assert backend._next(time.monotonic() + 1) == notification
    assert sent[0]["method"] == "thread/start"


@pytest.mark.parametrize("event", [
    {"method": "model/rerouted", "params": {"threadId": "t", "turnId": "u"}},
    {"method": "item/started", "params": {"threadId": "t", "turnId": "u", "item": {"type": "commandExecution"}}},
    {"id": 7, "method": "item/commandExecution/requestApproval", "params": {}},
])
def test_rejects_rerouting_and_tool_actions(monkeypatch, event):
    backend = CodexBackend()
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_request", lambda method, params, deadline: (
        {"thread": {"id": "t"}, "model": "gpt-6-astra"} if method == "thread/start"
        else {"turn": {"id": "u"}}
    ))
    monkeypatch.setattr(backend, "_next", lambda deadline: event)
    with pytest.raises(RuntimeError):
        backend.generate("test", 10, lambda *args, **kwargs: None)
