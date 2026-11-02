from types import SimpleNamespace

import numpy as np
import pytest

from temporal_bridge.backends import MLXLanguageModel, TimesFMBackend


@pytest.mark.parametrize("revision", [None, "test-commit"])
def test_timesfm_load_pins_revision_and_caches_model(monkeypatch, revision):
    import huggingface_hub
    from mlx_tsfm.models.timesfm import TimesFM3

    from temporal_bridge.backends import TIMESFM_REVISION

    calls = []
    sentinel = object()
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, name, **kw: calls.append((repo, name, kw)) or "/tmp/model.safetensors",
    )
    monkeypatch.setattr(TimesFM3, "from_pretrained", lambda **kw: sentinel)
    backend = TimesFMBackend(**({"revision": revision} if revision is not None else {}))
    assert backend._load() is sentinel
    assert backend._load() is sentinel
    assert calls == [("google/timesfm-3.0-pytorch", "model.safetensors",
                      {"revision": revision or TIMESFM_REVISION})]
    assert backend.revision == (revision or TIMESFM_REVISION)


def test_forecast_checks_input_and_requests_native_quantiles(monkeypatch):
    calls = []

    class Model:
        def forecast_batch(self, contexts, horizon, quantiles):
            calls.append((contexts.shape, horizon, quantiles))
            values = np.ones((len(contexts), horizon))
            return SimpleNamespace(point=values, quantiles=np.stack((values - 1, values + 1), -1))

    backend = TimesFMBackend()
    monkeypatch.setattr(backend, "_load", lambda: Model())
    monkeypatch.setattr("temporal_bridge.backends.mx.eval", lambda *args: None)
    for invalid in (np.ones(4), np.empty((0, 4)), np.empty((2, 0)), [[1, float("nan")]]):
        with pytest.raises(ValueError):
            backend.forecast(invalid, 2)
    with pytest.raises(ValueError):
        backend.forecast(np.ones((1, 4)), 0)
    result = backend.forecast(np.ones((2, 4)), 3)
    assert calls == [((2, 4), 3, [0.1, 0.9])]
    assert result.point.shape == result.q10.shape == result.q90.shape == (2, 3)
    assert np.all(result.q10 == 0) and np.all(result.q90 == 2)


def test_score_uses_last_position_and_forced_choice_distribution(monkeypatch):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [{"A": 1, "B": 2, "C": 3, "D": 4}[text]]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert messages[-1]["content"] == "test"
            return [9, 8]

        def decode(self, tokens):
            return str(tokens[0])

    class Logits:
        def __getitem__(self, indices):
            assert indices == (0, -1, slice(None))
            return self

        def astype(self, dtype):
            assert dtype == __import__("mlx.core", fromlist=["float32"]).float32
            return np.array([0, 1, 2, 3, 4, 9], dtype=np.float32)

    def model(tokens):
        assert tokens.tolist() == [[9, 8]]
        return Logits()

    backend = MLXLanguageModel("unused")
    monkeypatch.setattr(backend, "_load", lambda: (model, Tokenizer()))
    monkeypatch.setattr("temporal_bridge.backends.mx.array", np.asarray)
    monkeypatch.setattr("temporal_bridge.backends.mx.eval", lambda *args: None)
    result = backend.score("test")
    assert result["choice"] == "D"
    assert result["unrestricted_top_token"] == "5"
    assert result["input_tokens"] == 2
    assert result["total_s"] >= result["elapsed_s"] >= 0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["probabilities"]["D"] > result["probabilities"]["A"]


def test_rejects_multitoken_choice_labels(monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    backend = MLXLanguageModel("unused")
    monkeypatch.setattr(backend, "_load", lambda: (None, tokenizer))
    with pytest.raises(ValueError, match="single token"):
        backend.score("test")
