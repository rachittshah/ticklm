import numpy as np
import pytest

from temporal_bridge.core import (
    baseline_forecast,
    compile_evidence,
    interpolation_error,
    knot_indices,
    render_raw,
    render_stats,
)


def test_adaptive_knots_preserve_a_local_spike_missed_by_uniform():
    x = np.zeros(17)
    x[5] = 9
    adaptive = knot_indices(x, 5, "adaptive")
    uniform = knot_indices(x, 5, "uniform")
    assert 5 in adaptive
    assert interpolation_error(x, adaptive) == 0
    assert interpolation_error(x, uniform) == 9


def test_joint_curve_error_and_rounded_certificate():
    x = np.array([[0.00129, 1.54321], [3.92481, 2.22222], [0.01111, 6.76666]])
    assert interpolation_error(x, [0, 2]) > 3
    evidence = compile_evidence(x[:, 0], 0.5, budget=400, count_tokens=len)
    assert evidence.history_error >= 0
    assert evidence.evidence_tokens <= 400
    assert "hours" in evidence.text
    assert "observed" in evidence.text.lower()


def test_clock_uses_elapsed_time_and_ends_history_at_zero():
    evidence = compile_evidence(np.arange(8), 2.5, budget=1500, count_tokens=len)
    assert "-17.5" in evidence.text
    assert "0:7" in evidence.text
    assert "sample_interval=2.5 hours" in render_raw(np.arange(8), 2.5).text


def test_token_budget_never_silently_overflows():
    evidence = compile_evidence(np.sin(np.arange(128)), 1, budget=250, count_tokens=len)
    assert len(evidence.text) <= 250
    with pytest.raises(ValueError, match="budget"):
        compile_evidence(np.arange(10), 1, budget=12, count_tokens=len)


def test_forecast_projection_separate_and_quantile_crossings_recorded():
    forecast = np.array([[2.0, 1.0, 3.0], [4.0, 5.0, 3.0]])
    evidence = compile_evidence(np.arange(8), 1, forecast, budget=1600, count_tokens=len)
    assert evidence.quantile_crossings == 2
    assert "projected" in evidence.text.lower()
    assert "not path probabilities" in evidence.text
    assert evidence.forecast_error is not None


@pytest.mark.parametrize("bad", [[], [1, np.nan], [[1, 2], [3, 4]]])
def test_invalid_histories_rejected(bad):
    with pytest.raises(ValueError):
        compile_evidence(bad, 1, budget=1000, count_tokens=len)


@pytest.mark.parametrize("bad", [0, -1, np.inf])
def test_invalid_clock_rejected(bad):
    with pytest.raises(ValueError):
        compile_evidence(np.arange(8), bad, budget=1000, count_tokens=len)


def test_classical_forecaster_selects_periodic_history_without_future():
    x = np.tile(np.arange(24, dtype=float), 6)[:128]
    result = baseline_forecast(x, 24)
    expected = np.tile(np.arange(24, dtype=float), 7)[128:152]
    np.testing.assert_allclose(result["point"], expected)
    assert np.all(result["q10"] <= result["point"])
    assert np.all(result["q90"] >= result["point"])


def test_classical_forecaster_extrapolates_a_linear_trend_and_constant():
    x = np.arange(128, dtype=float) * 0.2
    result = baseline_forecast(x, 24)
    np.testing.assert_allclose(result["point"], np.arange(128, 152) * 0.2, atol=1e-10)
    np.testing.assert_allclose(baseline_forecast(np.ones(128), 24)["point"], 1)


def test_certificate_covers_interpolation_using_actual_printed_clock():
    x = np.arange(5, dtype=float)
    interval = 1.23456
    evidence = compile_evidence(x, interval, budget=1500, count_tokens=len)
    printed = np.array([[float(v) for v in knot.split(":")]
                        for knot in evidence.text.splitlines()[1].split()])
    times = (np.arange(len(x)) - len(x) + 1) * interval
    actual_error = np.max(np.abs(x - np.interp(times, printed[:, 0], printed[:, 1])))
    assert actual_error <= evidence.history_error
    assert "sample_interval=1.23456 hours" in evidence.text


def test_statistics_accept_two_observations():
    evidence = render_stats([1, 2], 1, budget=500, count_tokens=len)
    assert evidence.history_knots == 2


def test_classical_selection_does_not_use_nonexistent_training_window():
    result = baseline_forecast(np.arange(48), 24)
    assert result["model_id"] != "linear-64"


def test_crossing_repair_preserves_point_forecast():
    forecast = np.array([[2.0, 1.0, 3.0], [4.0, 5.0, 3.0]])
    evidence = compile_evidence(np.arange(8), 1, forecast, budget=1600, count_tokens=len)
    assert "1:1/1/3" in evidence.text
    assert "2:3/5/5" in evidence.text
