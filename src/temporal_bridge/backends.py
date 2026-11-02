"""Frozen local forecaster and language model used by the experiment.

Forced-choice scores normalize only the four answer-token logits; they are not
unrestricted generation probabilities or calibrated correctness probabilities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

TIMESFM_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


@dataclass
class ForecastResult:
    point: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    elapsed_s: float
    model_id: str


class TimesFMBackend:
    def __init__(self, repo: str = "google/timesfm-3.0-pytorch", *,
                 revision: str = TIMESFM_REVISION):
        self.repo = repo
        self.revision = revision
        self.model = None
        self.load_s = None
        self.checkpoint_path = None

    def _load(self):
        if self.model is None:
            from huggingface_hub import hf_hub_download
            from mlx_tsfm.models.timesfm import TimesFM3

            start = perf_counter()
            path = hf_hub_download(self.repo, "model.safetensors", revision=self.revision)
            self.model = TimesFM3.from_pretrained(weights_path=path)
            # Retain the snapshot path: resolving its symlink loses the source revision.
            self.checkpoint_path = str(Path(path).absolute())
            self.load_s = perf_counter() - start
        return self.model

    def forecast(self, contexts: np.ndarray, horizon: int) -> ForecastResult:
        contexts = np.asarray(contexts, dtype=np.float32)
        if contexts.ndim != 2 or not all(contexts.shape) or not np.isfinite(contexts).all():
            raise ValueError("contexts must be a nonempty finite (batch, context) array")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        model = self._load()
        start = perf_counter()
        result = model.forecast_batch(contexts, horizon, quantiles=[0.1, 0.9])
        mx.eval(result.point, result.quantiles)
        point = np.asarray(result.point)
        quantiles = np.asarray(result.quantiles)
        elapsed = perf_counter() - start
        if point.shape != (len(contexts), horizon) or quantiles.shape != (len(contexts), horizon, 2):
            raise ValueError("forecast model returned unexpected output shapes")
        if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
            raise ValueError("forecast model returned non-finite values")
        return ForecastResult(point, quantiles[:, :, 0], quantiles[:, :, 1], elapsed, self.repo)


class MLXLanguageModel:
    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self.model = None
        self.tokenizer = None
        self.load_s = None

    def _load(self):
        if self.model is None:
            from mlx_lm import load

            start = perf_counter()
            self.model, self.tokenizer = load(self.model_path)
            self.load_s = perf_counter() - start
        return self.model, self.tokenizer

    @staticmethod
    def _tokens(tokenizer, prompt: str, system: str):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def score(self, prompt: str) -> dict:
        total_start = perf_counter()
        model, tokenizer = self._load()
        labels = "ABCD"
        candidates = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(candidate) != 1 for candidate in candidates):
            raise ValueError("each answer label must encode as a single token")
        ids = [candidate[0] for candidate in candidates]
        if len(set(ids)) != 4:
            raise ValueError("answer labels must have distinct token IDs")
        tokens = self._tokens(
            tokenizer, prompt, "Choose the correct answer. Reply with only one letter: A, B, C, or D."
        )
        start = perf_counter()
        # NumPy cannot consume MLX's bfloat16 buffer directly.
        logits = model(mx.array([tokens]))[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        logits = np.asarray(logits, dtype=np.float64)
        scores = logits[ids]
        if not np.isfinite(scores).all():
            raise ValueError("language model returned non-finite candidate logits")
        weights = np.exp(scores - np.max(scores))
        probabilities = weights / weights.sum()
        elapsed = perf_counter() - start
        top_id = int(np.argmax(logits))
        result = {
            "choice": labels[int(np.argmax(probabilities))],
            "probabilities": dict(zip(labels, probabilities.tolist(), strict=True)),
            "input_tokens": len(tokens),
            "elapsed_s": elapsed,
            "candidate_token_ids": dict(zip(labels, ids, strict=True)),
            "unrestricted_top_token": tokenizer.decode([top_id]),
            "unrestricted_top_token_id": top_id,
        }
        result["total_s"] = perf_counter() - total_start
        return result

    def generate(self, prompt: str, max_tokens: int = 128) -> dict:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        from mlx_lm import generate

        model, tokenizer = self._load()
        tokens = self._tokens(tokenizer, prompt, "Answer the user's question concisely and accurately.")
        start = perf_counter()
        text = generate(model, tokenizer, prompt=tokens, max_tokens=max_tokens, verbose=False)
        return {"text": text, "input_tokens": len(tokens), "elapsed_s": perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the real local inference path")
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--model", required=True, help="Local MLX language model directory")
    parser.add_argument("--timesfm-repo", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--timesfm-revision", default=TIMESFM_REVISION)
    args = parser.parse_args()
    forecaster = TimesFMBackend(args.timesfm_repo, revision=args.timesfm_revision)
    result = forecaster.forecast(np.sin(np.linspace(0, 12, 128))[None, :], 16)
    language = MLXLanguageModel(args.model)
    score = language.score("Which number is even? A) 1 B) 2 C) 3 D) 5")
    print(json.dumps({
        "forecast": {"shape": list(result.point.shape), "finite": bool(np.isfinite(result.point).all()),
                     "q10_finite": bool(np.isfinite(result.q10).all()),
                     "q90_finite": bool(np.isfinite(result.q90).all()), "elapsed_s": result.elapsed_s,
                     "load_s": forecaster.load_s, "checkpoint": forecaster.checkpoint_path,
                     "revision": forecaster.revision},
        "language": {**score, "load_s": language.load_s, "model": args.model},
    }, indent=2))


if __name__ == "__main__":
    main()
