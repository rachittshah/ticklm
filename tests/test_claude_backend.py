import io
import json

import pytest

from temporal_bridge.claude_backend import ClaudeBackendError, ClaudeCodeBackend


class FakeProcess:
    def __init__(self, events, returncode=0):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
        self.stderr = io.StringIO("")
        self.returncode = returncode
        self.pid = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def test_warmup_reports_only_requested_cli_metadata(monkeypatch):
    class VersionProcess:
        pid = 0
        returncode = 0

        def communicate(self, timeout=None):
            return "Restored session noise\n2.1.316 (Claude Code)\n", ""

    backend = ClaudeCodeBackend()
    process = VersionProcess()
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_launch", lambda command, cwd: process)
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)

    metadata = backend.warmup()

    assert metadata == {
        "provider": "claude",
        "cli_version": "2.1.316",
        "requested_model": "opus",
        "requested_effort": "xhigh",
        "launcher_profile": "startup",
        "max_tokens_enforcement": "not_enforced",
    }


def test_generate_emits_only_visible_deltas_and_reports_completion(monkeypatch):
    backend = ClaudeCodeBackend()
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-4-6"},
        {"type": "stream_event", "event": {"delta": {"type": "thinking_delta", "thinking": "hidden"}}},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "Ready"}}},
        {"type": "result", "subtype": "success", "result": "Ready", "is_error": False,
         "usage": {"input_tokens": 12, "output_tokens": 1,
                   "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4},
         "total_cost_usd": 0.002,
         "stop_reason": "end_turn"},
    ]
    process = FakeProcess(events)
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)
    captured_command = []

    def launch(command, cwd):
        captured_command.extend(command)
        return process

    monkeypatch.setattr(backend, "_launch", launch)
    emitted = []
    backend.generate("Reply Ready", 8, lambda kind, **data: emitted.append((kind, data)))

    assert [kind for kind, _ in emitted] == ["started", "first_token", "delta", "result"]
    assert emitted[1][1]["evidence"] == "visible_text_delta"
    assert emitted[2][1]["text"] == "Ready"
    result = emitted[-1][1]
    assert result["text"] == "Ready"
    assert result["model"] == "claude-opus-4-6"
    assert result["resolved_model"] == "claude-opus-4-6"
    assert result["provider"] == "claude"
    assert result["input_tokens"] == 12
    assert result["cache_creation_input_tokens"] == 3
    assert result["cache_read_input_tokens"] == 4
    assert result["input_tokens_total"] == 19
    assert result["output_tokens"] == 1
    assert result["estimated_cost_usd"] == pytest.approx(0.002)
    assert result["cost_is_estimate"] is True
    assert result["requested_effort"] == "xhigh"
    assert result["finish_reason"] == "end_turn"
    assert result["max_tokens_enforcement"] == "not_enforced"
    assert "Reply Ready" not in captured_command


def test_generate_uses_assistant_model_and_model_usage_fallback(monkeypatch):
    backend = ClaudeCodeBackend()
    process = FakeProcess([
        {"type": "assistant", "message": {"model": "claude-opus-5-5"}},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "ok"}}},
        {"type": "result", "subtype": "success", "result": "ok",
         "modelUsage": {"claude-opus-5-5": {
             "inputTokens": 5, "outputTokens": 2,
             "cacheCreationInputTokens": 1, "cacheReadInputTokens": 2,
         }}},
    ])
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)
    monkeypatch.setattr(backend, "_launch", lambda command, cwd: process)
    emitted = []

    backend.generate("test", 8, lambda kind, **data: emitted.append((kind, data)))

    result = emitted[-1][1]
    assert result["resolved_model"] == "claude-opus-5-5"
    assert result["input_tokens"] == 5
    assert result["cache_creation_input_tokens"] == 1
    assert result["cache_read_input_tokens"] == 2
    assert result["input_tokens_total"] == 8
    assert result["output_tokens"] == 2


def test_generate_reports_model_conflict_without_claiming_one_resolved(monkeypatch):
    backend = ClaudeCodeBackend()
    process = FakeProcess([
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
        {"type": "assistant", "message": {"model": "claude-sonnet-4-5"}},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "ok"}}},
        {"type": "result", "subtype": "success", "result": "ok",
         "modelUsage": {"claude-opus-5-5": {}, "claude-sonnet-4-5": {}}},
    ])
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)
    monkeypatch.setattr(backend, "_launch", lambda command, cwd: process)
    emitted = []

    backend.generate("test", 8, lambda kind, **data: emitted.append((kind, data)))

    result = emitted[-1][1]
    assert result["model"] is None
    assert result["resolved_model"] is None
    assert result["models_used"] == ["claude-opus-5-5", "claude-sonnet-4-5"]


def test_generate_does_not_promote_requested_alias_to_resolved_model(monkeypatch):
    backend = ClaudeCodeBackend()
    process = FakeProcess([
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "ok"}}},
        {"type": "result", "subtype": "success", "result": "ok"},
    ])
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)
    monkeypatch.setattr(backend, "_launch", lambda command, cwd: process)
    emitted = []

    backend.generate("test", 8, lambda kind, **data: emitted.append((kind, data)))

    result = emitted[-1][1]
    assert result["model"] is None
    assert result["resolved_model"] is None
    assert result["requested_model"] == "opus"


def test_baseexception_releases_and_terminates_the_owned_process(monkeypatch):
    backend = ClaudeCodeBackend()
    process = FakeProcess([
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "ok"}}},
    ])
    terminated = []
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_launch", lambda command, cwd: process)
    monkeypatch.setattr(backend, "_terminate_process", lambda child: terminated.append(child))

    def stop_on_delta(kind, **data):
        if kind == "delta":
            raise SystemExit("worker stopping")

    with pytest.raises(SystemExit, match="worker stopping"):
        backend.generate("test", 8, stop_on_delta)

    assert terminated == [process]


def test_generate_rejects_error_result_and_requires_text_delta(monkeypatch):
    backend = ClaudeCodeBackend()
    monkeypatch.setattr(backend, "warmup", dict)
    monkeypatch.setattr(backend, "_make_cwd", lambda: "/tmp/claude-test")
    monkeypatch.setattr(backend, "_release", lambda process, cwd, *, terminate=False: None)
    monkeypatch.setattr(backend, "_build_generate_command", list)
    emitted = []

    for result_event, expected_code in [
        ({"type": "result", "subtype": "error_during_execution", "is_error": True,
          "result": "private detail"}, "result_error"),
        ({"type": "result", "subtype": "success", "is_error": False,
          "result": "text without a delta"}, "missing_visible_text"),
    ]:
        process = FakeProcess([result_event])
        monkeypatch.setattr(backend, "_launch", lambda command, cwd, proc=process: proc)
        with pytest.raises(ClaudeBackendError) as exc:
            backend.generate("test", 8, lambda kind, **data: emitted.append((kind, data)))
        assert exc.value.code == expected_code
        assert "private detail" not in str(exc.value)


def test_commands_use_startup_safe_profile_and_disable_tools():
    backend = ClaudeCodeBackend(profile="startup")
    command = backend._build_generate_command()

    assert command[:5] == ["zsh", "-ic", 'cca "$@"', "ticklm", "startup"]
    assert command[5:7] == ["--safe", "--print"]
    assert "--safe-mode" in command
    assert "--strict-mcp-config" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--model") + 1] == "opus"
    assert command[command.index("--effort") + 1] == "xhigh"
    assert "--dangerously-skip-permissions" not in command


def test_terminate_signals_group_even_if_shell_leader_has_exited(monkeypatch):
    process = FakeProcess([])
    process.pid = 12345
    signals = []
    monkeypatch.setattr(
        "temporal_bridge.claude_backend.os.killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )

    ClaudeCodeBackend._terminate_process(process, grace_s=0)

    import signal

    assert (12345, signal.SIGTERM) in signals
    assert (12345, signal.SIGKILL) in signals

