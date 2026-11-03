"""Spawn-isolated model worker for a responsive local real-time event loop.

Every message includes an explicit request ID and a process-comparable monotonic
timestamp. Timing and validity checks remain the responsibility of the event owner.
"""

from __future__ import annotations

import math
import multiprocessing
import queue
import signal
import time


class _Runtime:
    """All MLX imports and model instances stay in the spawned child."""

    def __init__(self, model_path, use_tsfm, *, provider="mlx", effort="xhigh",
                 claude_profile="startup", request_timeout_s=90):
        self.model_path = model_path
        self.use_tsfm = use_tsfm
        self.language = None
        self.forecaster = None
        self.cli_backend = None
        if provider == "codex":
            from temporal_bridge.codex_backend import CodexBackend

            self.cli_backend = CodexBackend(model=model_path, effort=effort,
                                            timeout_s=request_timeout_s)
        elif provider == "claude":
            from temporal_bridge.claude_backend import ClaudeCodeBackend

            self.cli_backend = ClaudeCodeBackend(model=model_path, effort=effort,
                                                 profile=claude_profile,
                                                 timeout_s=request_timeout_s)

    def _language(self):
        if self.language is None:
            from temporal_bridge.backends import MLXLanguageModel

            self.language = MLXLanguageModel(self.model_path)
        return self.language

    def _forecaster(self):
        if not self.use_tsfm:
            raise ValueError("TimesFM is disabled for this worker")
        if self.forecaster is None:
            from temporal_bridge.backends import TimesFMBackend

            self.forecaster = TimesFMBackend()
        return self.forecaster

    def warmup(self):
        if self.cli_backend is not None:
            info = self.cli_backend.warmup()
        else:
            language = self._language()
            language._load()
            info = {"provider": "mlx", "model_path": self.model_path,
                    "language_load_s": language.load_s}
        if self.use_tsfm:
            forecaster = self._forecaster()
            forecaster._load()
            info.update({"timesfm_repo": forecaster.repo, "timesfm_revision": forecaster.revision,
                         "timesfm_checkpoint": forecaster.checkpoint_path,
                         "timesfm_load_s": forecaster.load_s})
        return info

    def close(self):
        if self.cli_backend is not None:
            self.cli_backend.close()

    def prepare_generation(self, prompt, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        language = self._language()
        model, tokenizer = language._load()
        tokens = language._tokens(tokenizer, prompt, "Answer the user's question concisely and accurately.")
        iterator = stream_generate(model, tokenizer, prompt=tokens, max_tokens=max_tokens,
                                   sampler=make_sampler(temp=0))
        return iterator, len(tokens)

    def forecast(self, context, horizon):
        import numpy as np

        forecaster = self._forecaster()
        result = forecaster.forecast(np.asarray(context, dtype=np.float32)[None, :], horizon)
        return {"point": result.point[0].tolist(), "q10": result.q10[0].tolist(),
                "q90": result.q90[0].tolist(), "elapsed_s": result.elapsed_s,
                "model_id": result.model_id, "revision": forecaster.revision}


def _handle_request(request, runtime, emit):
    """Execute one request and emit its events, including structured failures."""
    request_id = request.get("request_id")

    def send(kind, **details):
        emit({"type": kind, "request_id": request_id, "monotonic": time.monotonic(), **details})

    try:
        kind = request["kind"]
        if kind == "health":
            send("result", status="ok")
        elif kind == "warmup":
            send("started")
            info = runtime.warmup()
            send("result", metadata=info)
        elif kind == "forecast":
            send("started")
            result = runtime.forecast(request["context"], request["horizon"])
            send("result", **result)
        elif kind == "generate":
            if getattr(runtime, "cli_backend", None) is not None:
                runtime.cli_backend.generate(request["prompt"], request["max_tokens"], send)
                return
            iterator, input_tokens = runtime.prepare_generation(request["prompt"], request["max_tokens"])
            started = time.monotonic()
            emit({"type": "started", "request_id": request_id, "monotonic": started})
            fragments = []
            last = None
            for response in iterator:
                if last is None:
                    send("first_token", token=int(response.token),
                         finish_reason=response.finish_reason)
                fragments.append(response.text)
                last = response
            if last is None:
                raise RuntimeError("model stream ended without yielding a token")
            send("result", text="".join(fragments), input_tokens=input_tokens,
                 output_tokens=int(last.generation_tokens), finish_reason=last.finish_reason,
                 elapsed_s=time.monotonic() - started)
        else:
            raise ValueError(f"unsupported worker kind: {kind}")
    except Exception as error:  # noqa: BLE001 - isolate backend failures at process boundary
        send("error", error=f"{type(error).__name__}: {error}")


def _stop_worker(signum, frame):
    # Let the worker close its owned CLI subprocesses before the parent escalates.
    raise SystemExit(0)


def _worker_main(requests, responses, model_path, use_tsfm, options):
    signal.signal(signal.SIGTERM, _stop_worker)
    runtime = _Runtime(model_path, use_tsfm, **options)
    try:
        while True:
            request = requests.get()
            if request is None:
                return
            _handle_request(request, runtime, responses.put)
    finally:
        runtime.close()


class ProcessModelWorker:
    """Serial model work in a spawned process; parent methods never load MLX."""

    def __init__(self, model_path: str, use_tsfm: bool = True, *, provider="mlx",
                 effort="xhigh", claude_profile="startup", request_timeout_s=90):
        if provider not in {"mlx", "claude", "codex"}:
            raise ValueError("unknown language provider")
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request timeout must be finite and positive")
        self.model_path = str(model_path)
        self.use_tsfm = bool(use_tsfm)
        self.options = {"provider": provider, "effort": effort,
                        "claude_profile": claude_profile, "request_timeout_s": request_timeout_s}
        self._process = None
        self._requests = None
        self._responses = None
        self._pending = set()
        self._exit_reported = False

    def start(self):
        if self._process is not None:
            raise RuntimeError("worker has already been started")
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(target=_worker_main,
                                        args=(self._requests, self._responses,
                                              self.model_path, self.use_tsfm, self.options), daemon=True)
        self._process.start()

    def submit(self, request_id, prompt="", max_tokens=128, kind="generate", context=None,
               horizon=24):
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("worker is not running")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if self._pending:
            raise RuntimeError("worker already has an outstanding request")
        if kind not in {"health", "warmup", "generate", "forecast"}:
            raise ValueError(f"unsupported worker kind: {kind}")
        if kind == "generate" and (not isinstance(prompt, str) or not prompt.strip()
                                   or isinstance(max_tokens, bool)
                                   or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("generation needs a nonempty prompt and positive max_tokens")
        if kind == "forecast" and (context is None or isinstance(horizon, bool)
                                   or not isinstance(horizon, int) or horizon < 1):
            raise ValueError("forecast needs context and a positive horizon")
        if kind == "forecast":
            try:
                context = [float(value) for value in context]
            except (TypeError, ValueError) as error:
                raise ValueError("forecast context must be a finite one-dimensional sequence") from error
            if not context or not all(math.isfinite(value) for value in context):
                raise ValueError("forecast context must be a finite one-dimensional sequence")
        self._pending.add(request_id)
        self._requests.put({"request_id": request_id, "prompt": prompt, "max_tokens": max_tokens,
                            "kind": kind, "context": context, "horizon": horizon})

    def poll(self):
        if self._process is None:
            return []
        messages = []
        while True:
            try:
                message = self._responses.get_nowait()
            except queue.Empty:
                break
            messages.append(message)
            if message["type"] in {"result", "error"}:
                self._pending.discard(message["request_id"])
        if not self._process.is_alive() and not self._exit_reported:
            self._exit_reported = True
            for request_id in sorted(self._pending) or [None]:
                messages.append({"type": "error", "request_id": request_id,
                                 "monotonic": time.monotonic(),
                                 "error": f"worker exited (exit code {self._process.exitcode})"})
            self._pending.clear()
        return messages

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def close(self, timeout=2.0):
        if self._process is None:
            return
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=max(0.0, timeout))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)
        for channel in (self._requests, self._responses):
            channel.cancel_join_thread()
            channel.close()
