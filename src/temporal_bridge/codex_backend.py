"""Fresh-thread Codex app-server generation with visible-text timing evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque


class CodexBackend:
    def __init__(self, model="gpt-6-astra", effort="xhigh", timeout_s=90):
        if not isinstance(model, str) or not model or not isinstance(effort, str) or not effort:
            raise ValueError("model and effort must be nonempty strings")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.model, self.effort, self.timeout_s = model, effort, timeout_s
        self._process = None
        self._temp = None
        self._queue = queue.Queue()
        self._pending = deque()
        self._id = 0
        self._lock = threading.Lock()
        self._metadata = None

    def _send(self, message):
        try:
            self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (OSError, BrokenPipeError, AttributeError) as exc:
            raise RuntimeError("Codex app-server transport closed") from exc

    def _next(self, deadline):
        if self._pending:
            return self._pending.popleft()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex app-server timed out")
        try:
            message = self._queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Codex app-server timed out") from exc
        if message is None:
            raise RuntimeError("Codex app-server transport closed")
        if not isinstance(message, dict):
            raise RuntimeError("Codex app-server sent invalid JSON")  # noqa: TRY004
        return message

    def _request(self, method, params, deadline):
        self._id += 1
        request_id = self._id
        self._send({"id": request_id, "method": method, "params": params})
        buffered = []
        try:
            while True:
                message = self._next(deadline)
                if message.get("id") == request_id and "method" not in message:
                    if "error" in message:
                        raise RuntimeError(f"Codex app-server rejected {method}")
                    result = message.get("result")
                    if not isinstance(result, dict):
                        raise RuntimeError(f"Codex app-server returned invalid {method} response")
                    return result
                if "method" in message and "id" in message:
                    raise RuntimeError("Codex app-server requested unsupported client action")
                buffered.append(message)
        finally:
            self._pending.extendleft(reversed(buffered))

    def warmup(self):
        if self._metadata is not None and self._process and self._process.poll() is None:
            return dict(self._metadata)
        self.close()
        try:
            version = subprocess.run(["codex", "--version"], capture_output=True, text=True,
                                     timeout=min(self.timeout_s, 10), check=True).stdout.strip()
            self._temp = tempfile.TemporaryDirectory(prefix="temporal-codex-")
            self._process = subprocess.Popen(
                ["codex", "app-server", "--stdio", "--disable", "shell_tool", "--disable", "hooks",
                 "--disable", "apps", "--disable", "browser_use", "--disable", "computer_use",
                 "--disable", "skill_search", "--disable", "unified_exec", "--disable", "code_mode",
                 "--disable", "code_mode_host", "-c", "mcp_servers={}", "-c", 'web_search="disabled"'],
                cwd=self._temp.name, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True,
            )
            self._queue = queue.Queue()
            output_queue = self._queue
            process = self._process

            def read_messages():
                try:
                    for line in process.stdout:
                        try:
                            output_queue.put(json.loads(line))
                        except json.JSONDecodeError:
                            output_queue.put(False)
                finally:
                    output_queue.put(None)

            threading.Thread(target=read_messages, daemon=True).start()
            deadline = time.monotonic() + self.timeout_s
            self._request("initialize", {"clientInfo": {"name": "temporal_bridge",
                          "version": "0.1.0"}, "capabilities": {"experimentalApi": True}}, deadline)
            self._send({"method": "initialized", "params": {}})
            thread = self._new_thread(deadline)
            self._metadata = {"cli_version": version, "requested_model": self.model,
                              "resolved_model": thread["model"], "requested_effort": self.effort,
                              "model_provider": thread.get("modelProvider"),
                              "provider": "codex", "transport": "app-server-stdio",
                              "max_tokens_enforced": False}
            return dict(self._metadata)
        except (OSError, subprocess.SubprocessError) as exc:
            self.close()
            raise RuntimeError("Unable to start Codex app-server") from exc
        except Exception:
            self.close()
            raise

    def _new_thread(self, deadline):
        thread = self._request("thread/start", {
            "model": self.model, "cwd": self._temp.name if self._temp else "/tmp",
            "ephemeral": True, "approvalPolicy": "never", "sandbox": "read-only",
            "allowProviderModelFallback": False, "environments": [],
            "baseInstructions": "Answer the user's question directly. Do not use tools.",
        }, deadline)
        if thread.get("model") != self.model:
            raise RuntimeError("Codex resolved a different model")
        return thread

    def generate(self, prompt, max_tokens, emit):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        with self._lock:
            start = time.monotonic()
            deadline = start + self.timeout_s
            try:
                self.warmup()
                thread = self._new_thread(deadline)
                resolved = thread.get("model")
                if resolved != self.model:
                    raise RuntimeError("Codex resolved a different model")
                thread_id = thread["thread"]["id"]
                if self._metadata is not None:
                    self._metadata["resolved_model"] = resolved
                    self._metadata["model_provider"] = thread.get("modelProvider")
                turn = self._request("turn/start", {"threadId": thread_id, "model": self.model,
                    "effort": self.effort, "input": [{"type": "text", "text": prompt}],
                    "environments": []}, deadline)
                turn_id = turn["turn"]["id"]
                emit("started", model=resolved, provider="codex", max_tokens_enforced=False)
                first = False
                final = None
                usage = {}
                while True:
                    message = self._next(deadline)
                    if "method" in message and "id" in message:
                        raise RuntimeError("Codex requested unsupported client action")
                    method, data = message.get("method"), message.get("params", {})
                    if data.get("threadId") != thread_id:
                        continue
                    event_turn_id = (data.get("turn", {}).get("id") if method == "turn/completed"
                                     else data.get("turnId"))
                    if event_turn_id != turn_id:
                        continue
                    if method == "model/rerouted":
                        raise RuntimeError("Codex rerouted the requested model")
                    if method == "item/agentMessage/delta" and data.get("delta"):
                        if not first:
                            first = True
                            emit("first_token", evidence="visible_text_delta",
                                 elapsed_s=time.monotonic() - start)
                        emit("delta", text=data["delta"])
                    elif method == "thread/tokenUsage/updated":
                        usage = data.get("tokenUsage", {}).get("last", {})
                    elif method in ("item/started", "item/completed"):
                        item = data.get("item", {})
                        kind = item.get("type")
                        if kind not in ("userMessage", "agentMessage", "reasoning"):
                            raise RuntimeError("Codex attempted an unsupported tool or action")
                        if (method == "item/completed" and kind == "agentMessage"
                                and item.get("phase") in (None, "final_answer")):
                            final = item.get("text")
                    elif method == "turn/completed":
                        status = data["turn"].get("status")
                        if status != "completed":
                            raise RuntimeError(f"Codex turn {status if status in ('failed', 'interrupted') else 'did not complete'}")
                        if not isinstance(final, str) or not final or not first:
                            raise RuntimeError("Codex completed without streamed final text")
                        emit("result", text=final, input_tokens=usage.get("inputTokens"),
                             output_tokens=usage.get("outputTokens"), usage=usage,
                             finish_reason="completed", elapsed_s=time.monotonic() - start,
                             model=resolved, provider="codex")
                        return
            except (KeyError, TypeError) as exc:
                self.close()
                raise RuntimeError("Codex app-server sent an invalid response") from exc
            except Exception:
                self.close()
                raise

    def close(self):
        process, self._process = self._process, None
        self._metadata = None
        self._pending.clear()
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            # A child may outlive the app-server itself while still holding stdout open.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=0.5)
            if process.stdout:
                process.stdout.close()
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    args = parser.parse_args()
    if args.probe:
        backend = CodexBackend()
        try:
            print(json.dumps({"warmup": backend.warmup()}), flush=True)
            backend.generate("Reply with exactly: CODEX_PROBE_OK", 16,
                             lambda event, **data: print(json.dumps({"event": event, **data}), flush=True))
        finally:
            backend.close()


if __name__ == "__main__":
    main()
