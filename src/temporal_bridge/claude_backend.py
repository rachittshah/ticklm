"""Claude Code CLI streaming adapter for the temporal-bridge worker."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

_SAFE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SAFE_REASON = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_API_ERRORS = {
    "authentication_failed",
    "oauth_org_not_allowed",
    "account_on_hold",
    "billing_error",
    "rate_limit",
    "overloaded",
    "invalid_request",
    "model_not_found",
    "server_error",
    "max_output_tokens",
    "cloud_credential_error",
    "unknown",
}
_STREAM_END = object()


class ClaudeBackendError(RuntimeError):
    """A sanitized failure from a Claude Code CLI request."""

    def __init__(self, code: str, *, exit_code: int | None = None):
        self.code = code
        self.exit_code = exit_code
        detail = f"Claude Code request failed ({code})"
        if exit_code is not None:
            detail += f", exit status {exit_code}"
        super().__init__(detail)


@dataclass
class _StreamState:
    text_deltas: list[str] = field(default_factory=list)
    model: str | None = None
    models_used: list[str] = field(default_factory=list)
    model_conflict: bool = False
    result: dict | None = None
    api_error: str | None = None


def _safe_model(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_MODEL.fullmatch(value):
        return value
    return None


def _record_model(state: _StreamState, value: object) -> None:
    model = _safe_model(value)
    if model is None:
        return
    if model not in state.models_used:
        state.models_used.append(model)
    if state.model is None:
        state.model = model
    elif state.model != model:
        state.model_conflict = True


def _safe_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _safe_cost(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result) and result >= 0:
            return result
    return None


def _first_count(source: dict, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _safe_count(source.get(name))
        if value is not None:
            return value
    return None


def _usage_counts(result: dict) -> tuple[int | None, int | None, int | None, int | None]:
    usage_counts = (None, None, None, None)
    usage = result.get("usage")
    if isinstance(usage, dict):
        usage_counts = (
            _first_count(usage, ("input_tokens", "inputTokens")),
            _first_count(usage, ("output_tokens", "outputTokens")),
            _first_count(usage, ("cache_creation_input_tokens", "cacheCreationInputTokens")),
            _first_count(usage, ("cache_read_input_tokens", "cacheReadInputTokens")),
        )

    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict):
        return usage_counts
    model_counts = [[], [], [], []]
    for usage_by_model in model_usage.values():
        if not isinstance(usage_by_model, dict):
            continue
        names = (
            ("inputTokens", "input_tokens"),
            ("outputTokens", "output_tokens"),
            ("cacheCreationInputTokens", "cache_creation_input_tokens"),
            ("cacheReadInputTokens", "cache_read_input_tokens"),
        )
        for counts, aliases in zip(model_counts, names, strict=True):
            value = _first_count(usage_by_model, aliases)
            if value is not None:
                counts.append(value)
    fallback_counts = tuple(sum(counts) if counts else None for counts in model_counts)
    return tuple(current if current is not None else fallback
                 for current, fallback in zip(usage_counts, fallback_counts, strict=True))


def _consume_event(event: dict, state: _StreamState,
                   emit: Callable[..., None], requested_model: str) -> None:
    event_type = event.get("type")
    subtype = event.get("subtype")
    if event_type == "system":
        if subtype == "init":
            _record_model(state, event.get("model"))
        elif subtype == "api_retry" and event.get("error") in _API_ERRORS:
            state.api_error = event["error"]
    elif event_type == "assistant":
        message = event.get("message")
        if isinstance(message, dict):
            _record_model(state, message.get("model"))
    elif event_type == "stream_event":
        stream_event = event.get("event")
        delta = stream_event.get("delta") if isinstance(stream_event, dict) else None
        if not isinstance(delta, dict) or delta.get("type") != "text_delta":
            return
        text = delta.get("text")
        if not isinstance(text, str) or not text:
            return
        if not state.text_deltas:
            emit("first_token", evidence="visible_text_delta",
                 resolved_model=state.model if not state.model_conflict else None,
                 requested_model=requested_model)
        state.text_deltas.append(text)
        emit("delta", text=text)
    elif event_type == "result":
        state.result = event
        _record_model(state, event.get("model"))
        model_usage = event.get("modelUsage")
        if isinstance(model_usage, dict):
            for model in model_usage:
                _record_model(state, model)


class ClaudeCodeBackend:
    """Run one-shot Claude Code requests through the selected `cca` profile."""

    def __init__(self, model: str = "opus", effort: str = "xhigh", profile: str = "startup",
                 timeout_s: float = 90):
        if not isinstance(model, str) or not _SAFE_MODEL.fullmatch(model):
            raise ValueError("model must be a safe nonempty CLI model name")
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("effort must be low, medium, high, xhigh, or max")
        if not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", profile):
            raise ValueError("profile must be a safe profile name")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be a positive finite number")
        self.model = model
        self.effort = effort
        self.profile = profile
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._closed = False
        self._processes: set[subprocess.Popen] = set()
        self._tempdirs: set[str] = set()
        self._metadata: dict | None = None

    def _launcher(self, *claude_args: str) -> list[str]:
        # The first argument after -c is zsh's $0; the remaining args become $@.
        return ["zsh", "-ic", 'cca "$@"', "ticklm", self.profile, "--safe", *claude_args]

    def _build_generate_command(self) -> list[str]:
        return self._launcher(
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--tools", "",
            "--strict-mcp-config",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--effort", self.effort,
            "--model", self.model,
        )

    def _make_cwd(self) -> str:
        path = tempfile.mkdtemp(prefix="temporal-bridge-claude-")
        with self._lock:
            if self._closed:
                shutil.rmtree(path, ignore_errors=True)
                raise ClaudeBackendError("backend_closed")
            self._tempdirs.add(path)
        return path

    def _launch(self, command: list[str], cwd: str) -> subprocess.Popen:
        env = os.environ.copy()
        env["SHELL_SESSIONS_DISABLE"] = "1"
        with self._lock:
            if self._closed:
                raise ClaudeBackendError("backend_closed")
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self._processes.add(process)
        return process

    def _release(self, process: subprocess.Popen | None, cwd: str, *, terminate: bool = False) -> None:
        if process is not None and terminate:
            self._terminate_process(process)
        with self._lock:
            if process is not None:
                self._processes.discard(process)
            self._tempdirs.discard(cwd)
        shutil.rmtree(cwd, ignore_errors=True)

    @staticmethod
    def _terminate_process(process: subprocess.Popen, grace_s: float = 0.25) -> None:
        process_group = process.pid
        if not isinstance(process_group, int) or process_group <= 0:
            return
        leader_running = process.poll() is None
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            if leader_running:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            return
        except PermissionError:
            if leader_running:
                process.terminate()

        if leader_running:
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    process.kill()
                try:
                    process.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    pass
                return

        # The shell may have exited while a child still holds one of its pipes.
        # Its process group remains tied to the session leader's process ID.
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                break
            time.sleep(min(0.025, max(0, deadline - time.monotonic())))
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                process.kill()

    def warmup(self) -> dict:
        """Return safe local CLI metadata without reading or exposing auth details."""
        with self._lock:
            if self._closed:
                raise ClaudeBackendError("backend_closed")
            if self._metadata is not None:
                return dict(self._metadata)

        cwd = self._make_cwd()
        process = None
        completed = False
        try:
            process = self._launch(self._launcher("--safe-mode", "--version"), cwd)
            try:
                stdout, _ = process.communicate(timeout=min(self.timeout_s, 10.0))
            except subprocess.TimeoutExpired as exc:
                raise ClaudeBackendError("cli_version_timeout") from exc
            if process.returncode != 0:
                raise ClaudeBackendError("cli_version_failed", exit_code=process.returncode)
            version = next((match.group(1) for line in stdout.splitlines()
                            if (match := re.match(r"^\s*(\d+\.\d+\.\d+)(?:\s|$)", line))), None)
            if version is None:
                raise ClaudeBackendError("cli_version_unavailable")
            metadata = {
                "provider": "claude",
                "cli_version": version,
                "requested_model": self.model,
                "requested_effort": self.effort,
                "launcher_profile": self.profile,
                "max_tokens_enforcement": "not_enforced",
            }
            with self._lock:
                self._metadata = metadata
            completed = True
            return dict(metadata)
        except OSError as exc:
            raise ClaudeBackendError("cli_unavailable") from exc
        finally:
            self._release(process, cwd, terminate=not completed)

    def _stream(self, process: subprocess.Popen, prompt: str,
                emit: Callable[..., None]) -> _StreamState:
        output_queue: queue.Queue = queue.Queue()
        state = _StreamState()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(_STREAM_END)

        def drain_stderr() -> None:
            for _ in process.stderr:
                pass

        def write_prompt() -> None:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        stdout_thread = threading.Thread(target=read_output, daemon=True)
        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stdin_thread = threading.Thread(target=write_prompt, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        stdin_thread.start()

        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClaudeBackendError("request_timeout")
            try:
                line = output_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if process.poll() is not None and not stdout_thread.is_alive():
                    break
                continue
            if line is _STREAM_END:
                break
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                _consume_event(event, state, emit, self.model)

        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise ClaudeBackendError("request_timeout") from exc
        if process.returncode != 0:
            raise ClaudeBackendError("cli_exit", exit_code=process.returncode)
        return state

    def _result(self, state: _StreamState, elapsed_s: float, max_tokens: int) -> dict:
        result = state.result
        if result is None:
            raise ClaudeBackendError(state.api_error or "missing_result")
        subtype = result.get("subtype")
        if result.get("is_error") is True or (isinstance(subtype, str) and subtype.startswith("error")):
            raise ClaudeBackendError(state.api_error or "result_error")
        if not state.text_deltas:
            raise ClaudeBackendError("missing_visible_text")
        text = result.get("result")
        if not isinstance(text, str) or not text.strip():
            raise ClaudeBackendError("missing_result_text")

        (input_tokens, output_tokens, cache_creation_input_tokens,
         cache_read_input_tokens) = _usage_counts(result)
        input_tokens_total = None
        if (input_tokens is not None and cache_creation_input_tokens is not None
                and cache_read_input_tokens is not None):
            input_tokens_total = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
        cost = _safe_cost(result.get("total_cost_usd"))
        if cost is None:
            cost = _safe_cost(result.get("cost_usd"))
        finish_reason = result.get("stop_reason", result.get("finish_reason"))
        if not isinstance(finish_reason, str) or not _SAFE_REASON.fullmatch(finish_reason):
            finish_reason = None
        resolved_model = state.model if not state.model_conflict else None
        return {
            "text": text,
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "input_tokens_total": input_tokens_total,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "elapsed_s": elapsed_s,
            "model": resolved_model,
            "resolved_model": resolved_model,
            "models_used": list(state.models_used),
            "provider": "claude",
            "requested_model": self.model,
            "estimated_cost_usd": cost,
            "cost_is_estimate": True,
            "requested_effort": self.effort,
            "requested_max_tokens": max_tokens,
            "max_tokens_enforcement": "not_enforced",
        }

    def generate(self, prompt: str, max_tokens: int, emit: Callable[..., None]) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not callable(emit):
            raise TypeError("emit must be callable")
        self.warmup()
        started = time.perf_counter()
        emit("started", provider="claude", requested_model=self.model,
             requested_effort=self.effort,
             requested_max_tokens=max_tokens, max_tokens_enforcement="not_enforced")
        cwd = self._make_cwd()
        process = None
        completed = False
        try:
            process = self._launch(self._build_generate_command(), cwd)
            state = self._stream(process, prompt, emit)
            result = self._result(state, time.perf_counter() - started, max_tokens)
            emit("result", **result)
            completed = True
        except OSError as exc:
            raise ClaudeBackendError("cli_launch_failed") from exc
        finally:
            self._release(process, cwd, terminate=not completed)

    def close(self) -> None:
        """Terminate owned CLI process groups and remove their temporary workdirs."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            processes = list(self._processes)
            tempdirs = list(self._tempdirs)
            self._processes.clear()
            self._tempdirs.clear()
        for process in processes:
            self._terminate_process(process)
        for path in tempdirs:
            shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe the Claude Code streaming adapter")
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--prompt", default="Reply with exactly: READY")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--profile", default="startup")
    parser.add_argument("--timeout-s", type=float, default=90)
    args = parser.parse_args(argv)
    backend = ClaudeCodeBackend(args.model, args.effort, args.profile, args.timeout_s)
    emitted = []
    try:
        metadata = backend.warmup()
        backend.generate(args.prompt, 8, lambda kind, **data: emitted.append({"type": kind, **data}))
        print(json.dumps({"metadata": metadata, "events": emitted}, indent=2))
    except ClaudeBackendError as exc:
        parser.exit(1, f"Claude Code probe failed ({exc.code})\n")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
