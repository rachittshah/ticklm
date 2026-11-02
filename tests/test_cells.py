import inspect

import numpy as np
import pytest

from temporal_bridge.cells import compile_cells


def _history_cells(text):
    return [line for line in text.splitlines() if line.startswith("past [")]


def _forecast_cells(text):
    return [line for line in text.splitlines() if line.startswith("future [")]


def test_history_cells_report_exact_sample_stats_and_elapsed_intervals():
    history = np.array([-2.0, 0.0, 1.5, 4.0, -1.0, 3.0, 2.0, 5.0])

    evidence = compile_cells(history, 2.0, history_bins=4, budget=4000)

    cells = _history_cells(evidence.text)
    assert cells == [
        "past [-14,-12]h n=2 mean=-1 min=-2 max=0 positive=0/2",
        "past [-10,-8]h n=2 mean=2.75 min=1.5 max=4 positive=2/2",
        "past [-6,-4]h n=2 mean=1 min=-1 max=3 positive=1/2",
        "past [-2,0]h n=2 mean=3.5 min=2 max=5 positive=2/2",
    ]
    assert evidence.history_knots == 4
    assert evidence.history_error is None


def test_forecast_cells_use_point_extrema_and_keep_quantiles_separate():
    horizon = 24
    forecast = np.column_stack(
        [
            np.full(horizon, -10.0),
            np.arange(horizon, dtype=float),
            np.full(horizon, 100.0),
        ]
    )
    forecast[:6, 0] = -20.0
    forecast[0, 0] = 10.0  # Keep this crossing visible; the point forecast remains 0.

    evidence = compile_cells(
        np.arange(128, dtype=float),
        1,
        forecast,
        history_bins=4,
        forecast_bins=4,
        budget=6000,
    )

    cells = _forecast_cells(evidence.text)
    assert len(cells) == 4
    assert cells[0] == (
        "future [1,6]h n=6 point_mean=2.5 point_min=0 point_max=5 "
        "q10_mean=-15 q90_mean=100"
    )
    assert cells[2] == (
        "future [13,18]h n=6 point_mean=14.5 point_min=12 point_max=17 "
        "q10_mean=-10 q90_mean=100"
    )
    assert "not path probabilities" in evidence.text
    assert "point extrema use only point values" in evidence.text
    assert evidence.quantile_crossings == 1


def test_forecast_only_mode_retains_the_current_observation():
    forecast = np.column_stack([np.arange(6), np.arange(6) + 1, np.arange(6) + 2])

    evidence = compile_cells(
        np.arange(8, dtype=float),
        0.5,
        forecast,
        include_history=False,
        history_bins=4,
        forecast_bins=2,
        budget=3000,
    )

    assert "Observed current sample: t=0h value=7" in evidence.text
    assert not _history_cells(evidence.text)
    assert evidence.history_knots == 0
    with pytest.raises(ValueError, match="history"):
        compile_cells(np.arange(8), 1, include_history=False, budget=3000)


def test_reversal_reorders_cell_statistics_and_dilation_scales_clocks():
    history = np.arange(8, dtype=float)
    forecast = np.column_stack(
        [np.arange(24, dtype=float), np.arange(24, dtype=float) + 1,
         np.arange(24, dtype=float) + 2]
    )
    forward = compile_cells(history, 1, forecast, history_bins=4, forecast_bins=4,
                            budget=6000)
    reversed_evidence = compile_cells(history[::-1], 1, forecast, history_bins=4,
                                      forecast_bins=4, budget=6000)
    dilated = compile_cells(history, 2, forecast, history_bins=4, forecast_bins=4,
                            budget=6000)

    forward_stats = [line.split(" mean=", 1)[1] for line in _history_cells(forward.text)]
    reverse_stats = [line.split(" mean=", 1)[1] for line in _history_cells(reversed_evidence.text)]
    assert reverse_stats == list(reversed(forward_stats))
    assert "past [-14,-12]h" in dilated.text
    assert "future [2,12]h" in dilated.text
    assert "future [38,48]h" in dilated.text
    assert "sample_interval=2h" in dilated.text


def test_token_budget_reduces_history_resolution_and_never_overflows():
    history = np.sin(np.arange(128, dtype=float) / 7)
    one_bin = compile_cells(history, 1, history_bins=1, budget=10000, count_tokens=len)

    reduced = compile_cells(history, 1, history_bins=8, budget=one_bin.evidence_tokens,
                            count_tokens=len)

    assert reduced.history_knots == 1
    assert reduced.evidence_tokens == len(reduced.text) <= one_bin.evidence_tokens
    with pytest.raises(ValueError, match="budget"):
        compile_cells(history, 1, history_bins=8, budget=one_bin.evidence_tokens - 1,
                      count_tokens=len)


def test_compiler_has_no_question_or_answer_inputs():
    assert "question" not in inspect.signature(compile_cells).parameters
    assert "answer" not in inspect.signature(compile_cells).parameters
    evidence = compile_cells(np.arange(8), 1, history_bins=2, budget=3000)
    assert "answer=" not in evidence.text.lower()
    assert "question=" not in evidence.text.lower()


@pytest.mark.parametrize("bad", [[], [1, np.nan], [[1, 2], [3, 4]]])
def test_invalid_histories_are_rejected(bad):
    with pytest.raises(ValueError):
        compile_cells(bad, 1, budget=3000)


@pytest.mark.parametrize("bad", [np.zeros((24, 2)), np.zeros((0, 3)), np.array([[1, 2, np.inf]])])
def test_invalid_forecasts_are_rejected(bad):
    with pytest.raises(ValueError, match="Forecast"):
        compile_cells(np.arange(8), 1, bad, budget=3000)
