import json

import numpy as np
import pytest

from app import answer, read_series


def test_series_input_preserves_physical_interval(tmp_path):
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"values": [1, 2, 3], "step_hours": .5}))
    values, interval = read_series(path)
    np.testing.assert_array_equal(values, [1, 2, 3])
    assert interval == .5


@pytest.mark.parametrize("payload", [
    {"values": [1, 2]},
    {"values": [1, 2], "step_hours": 0},
    {"values": [[1, 2]], "step_hours": 1},
    {"values": [1, 2], "step_hours": 1, "timestamps": [0, 3]},
    {"values": [1, 2], "step_hours": 1, "dates": [0, 3]},
])
def test_invalid_series_is_rejected_without_model_loading(tmp_path, payload):
    path = tmp_path / "series.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_series(path)


def test_history_only_request_never_needs_a_forecaster():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    class Language:
        model_path = "test"
        tokenizer = Tokenizer()

        def _load(self):
            pass

        def generate(self, prompt, **kwargs):
            assert "projected" not in prompt.lower()
            return {"text": "Increasing observations.", "input_tokens": 50, "elapsed_s": .01}

    result = answer(np.arange(128), 1, "What changed?", language=Language())
    assert result["forecast"] is None
    assert result["timings_s"]["forecast_forward"] == 0


def test_reader_preserves_distinctions_smaller_than_float32_precision(tmp_path):
    path = tmp_path / "series.json"
    path.write_text(json.dumps({"values": [16777216, 16777217], "step_hours": 1}))
    values, _ = read_series(path)
    assert values[1] - values[0] == 1


def test_invalid_request_fails_before_loading_language_model():
    class NeverLoad:
        def _load(self):
            pytest.fail("invalid request must not load a model")

    with pytest.raises(ValueError, match="48"):
        answer(np.arange(2), 1, "Forecast?", language=NeverLoad(), cheap=True)
    with pytest.raises(ValueError, match="positive"):
        answer(np.arange(128), 1, "Forecast?", language=NeverLoad(), max_tokens=-1)
    with pytest.raises(ValueError, match="exclusive"):
        answer(np.arange(128), 1, "Forecast?", language=NeverLoad(),
               forecaster=object(), cheap=True)
