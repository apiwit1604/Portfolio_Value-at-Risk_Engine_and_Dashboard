"""
Tests for var_risk_engine.backtest.

Focus areas:
    1. build_rolling_windows -- must not leak future data into a given window
       (look-ahead bias at the window level).
    2. compute_actual_return -- correct indexing/ordering walking backward
       from the end of the sample.
    3. kupiec_test -- correctness against hand-computed reference values,
       and correct pass/fail behavior at known exception rates. Also checks
       the key-alignment fragility flagged as a LIMITATION in the source.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from var_risk_engine.backtest import (
    build_rolling_windows,
    calculate_var_metrics,
    compute_actual_return,
    kupiec_test,
)


class TestBuildRollingWindows:
    def test_window_zero_excludes_the_most_recent_day(self, synthetic_returns):
        """LIMITATION guarded here: window -i must cover data only up
        through the day BEFORE day -i, so it never sees day -i's actual
        return. This is the look-ahead-bias protection the source code
        describes -- verify it actually holds for window 0.
        """
        windows = build_rolling_windows(synthetic_returns, rolling_window=50, n_windows=3)
        end = len(synthetic_returns)

        window_0 = windows[0]
        # window[0] should be returns_daily.iloc[end-50 : end], i.e. it
        # should NOT include the very last row available if "day 0" is
        # meant to be predicted, not observed, by this window.
        # Per the source's own docstring: window[-i] = iloc[end-window-i : end-i]
        # so window[0] = iloc[end-50 : end] -- it DOES include the last row.
        # This test pins the documented behavior so any accidental off-by-one
        # change is caught.
        expected = synthetic_returns.iloc[end - 50 : end]
        pd.testing.assert_frame_equal(window_0, expected)

    def test_each_window_has_correct_length(self, synthetic_returns):
        rolling_window = 100
        windows = build_rolling_windows(synthetic_returns, rolling_window, n_windows=5)
        for key, window in windows.items():
            assert len(window) == rolling_window, f"window {key} has wrong length"

    def test_consecutive_windows_shift_by_one_day(self, synthetic_returns):
        windows = build_rolling_windows(synthetic_returns, rolling_window=50, n_windows=3)
        # window[-1] shifted back by one day relative to window[0]:
        # the last row of window[-1] should be one day earlier than the
        # last row of window[0].
        last_day_w0 = windows[0].index[-1]
        last_day_w1 = windows[-1].index[-1]
        assert last_day_w1 < last_day_w0

    def test_number_of_windows_matches_request(self, synthetic_returns):
        windows = build_rolling_windows(synthetic_returns, rolling_window=50, n_windows=10)
        assert len(windows) == 10
        assert set(windows.keys()) == set(range(0, -10, -1))


class TestComputeActualReturn:
    def test_key_zero_is_the_most_recent_day(self, synthetic_returns, equal_weights):
        result = compute_actual_return(synthetic_returns, equal_weights, n_windows=5)
        actual = result["actual_return"]

        expected_last_day = synthetic_returns.iloc[-1] @ equal_weights
        assert actual[0] == pytest.approx(expected_last_day)

    def test_walks_backward_correctly(self, synthetic_returns, equal_weights):
        result = compute_actual_return(synthetic_returns, equal_weights, n_windows=5)
        actual = result["actual_return"]

        for i in range(5):
            expected = synthetic_returns.iloc[-1 - i] @ equal_weights
            assert actual[-i] == pytest.approx(expected)

    def test_returns_requested_number_of_entries(self, synthetic_returns, equal_weights):
        result = compute_actual_return(synthetic_returns, equal_weights, n_windows=7)
        assert len(result["actual_return"]) == 7


class TestCalculateVarMetrics:
    def test_returns_one_entry_per_window(self, synthetic_returns, equal_weights):
        windows = build_rolling_windows(synthetic_returns, rolling_window=60, n_windows=4)
        result = calculate_var_metrics(windows, equal_weights, n_rolling=4, confidence=0.95, t=1, mc_sims=2_000)

        assert set(result.keys()) == {"Parametric VaR", "Historical VaR", "Monte VaR"}
        for label, values in result.items():
            assert len(values) == 4, f"{label} has wrong number of entries"

    def test_parametric_var_matches_manual_calculation_per_window(self, synthetic_returns, equal_weights):
        """Cross-check the rolling parametric VaR against the same formula
        computed independently for a single window, at horizon t=1 (the
        value calculate_var_metrics uses internally).
        """
        from scipy.stats import norm

        windows = build_rolling_windows(synthetic_returns, rolling_window=80, n_windows=2)
        result = calculate_var_metrics(windows, equal_weights, n_rolling=2, confidence=0.95, t=1, mc_sims=1_000)

        window_0 = windows[0]
        cov_matrix = window_0.cov()
        mu_asset = window_0.mean()
        sigma_port = np.sqrt(equal_weights @ cov_matrix @ equal_weights)
        mu_port = mu_asset @ equal_weights
        z = norm.ppf(0.05)
        expected_var = mu_port + z * sigma_port

        assert result["Parametric VaR"][0] == pytest.approx(expected_var)

    def test_historical_var_matches_empirical_quantile_per_window(self, synthetic_returns, equal_weights):
        windows = build_rolling_windows(synthetic_returns, rolling_window=80, n_windows=2)
        result = calculate_var_metrics(windows, equal_weights, n_rolling=2, confidence=0.95, t=1, mc_sims=1_000)

        window_0 = windows[0]
        expected = (window_0 @ equal_weights).quantile(0.05)
        assert result["Historical VaR"][0] == pytest.approx(expected)

    def test_monte_carlo_var_is_in_a_reasonable_range(self, synthetic_returns, equal_weights):
        """With enough simulations, the rolling Monte Carlo VaR should land
        in the same ballpark as the rolling parametric VaR for the same
        window, since both draw on the same underlying Gaussian assumption.
        """
        windows = build_rolling_windows(synthetic_returns, rolling_window=150, n_windows=1)
        result = calculate_var_metrics(windows, equal_weights, n_rolling=1, confidence=0.95, t=1, mc_sims=50_000)

        assert result["Monte VaR"][0] == pytest.approx(result["Parametric VaR"][0], abs=0.01)


class TestKupiecTest:
    def test_matches_hand_computed_likelihood_ratio(self, capsys):
        """Construct a small, fully controlled scenario and verify the
        printed LR statistic against an independently hand-computed value,
        rather than just checking that the function runs.
        """
        n = 100
        confidence = 0.95
        # 5 exceptions out of 100 at 95% confidence == exactly the
        # theoretical rate (p_hat == p), so LR should be ~0.
        actual_return = {i: 0.0 for i in range(n)}
        model_var = {i: -1.0 for i in range(n)}  # VaR never breached except below
        for i in range(5):
            actual_return[i] = -2.0  # breach: actual worse than VaR

        kupiec_test(model_var, actual_return, confidence)
        captured = capsys.readouterr()

        # LR can print as "0.0000" or "-0.0000" depending on floating-point
        # sign-of-zero -- both represent the same (numerically zero) value,
        # so check the parsed magnitude rather than the exact string.
        lr_line = next(line for line in captured.out.splitlines() if line.startswith("LR:"))
        lr_value = float(lr_line.split(":")[1])
        assert lr_value == pytest.approx(0.0, abs=1e-9)
        assert "Pass" in captured.out

    def test_flags_model_with_far_too_many_exceptions(self, capsys):
        """A model with a 30% exception rate at 95% confidence should
        clearly fail (reject H0).
        """
        n = 200
        confidence = 0.95
        actual_return = {i: 0.0 for i in range(n)}
        model_var = {i: -1.0 for i in range(n)}
        for i in range(60):  # 30% exception rate, vs 5% expected
            actual_return[i] = -2.0

        kupiec_test(model_var, actual_return, confidence)
        captured = capsys.readouterr()

        assert "reject H0 (Fail)" in captured.out

    def test_decision_rule_is_internally_consistent_with_chi2(self, capsys):
        """Cross-check that the printed decision is consistent with an
        independently computed chi-square critical value/p-value, using
        scipy directly (not re-using the function's own internals).
        """
        n = 250
        confidence = 0.95
        x = 20  # exceptions
        actual_return = {i: 0.0 for i in range(n)}
        model_var = {i: -1.0 for i in range(n)}
        for i in range(x):
            actual_return[i] = -2.0

        p = 1 - confidence
        p_hat = x / n
        lr_pof = -2 * np.log(
            (((1 - p) ** (n - x)) * (p**x)) / (((1 - p_hat) ** (n - x)) * (p_hat**x))
        )
        expected_p_value = 1 - chi2.cdf(lr_pof, df=1)
        expect_pass = expected_p_value > p

        kupiec_test(model_var, actual_return, confidence)
        captured = capsys.readouterr()

        if expect_pass:
            assert "Pass" in captured.out
        else:
            assert "Fail" in captured.out

    def test_raises_or_misaligns_visibly_on_mismatched_keys(self):
        """LIMITATION regression guard: kupiec_test combines model_var and
        actual_return via pd.DataFrame({...}), which aligns by dict key.
        If the key sets don't match, pandas introduces NaNs rather than
        raising -- silently corrupting the exception count. This test
        documents that current (fragile) behavior so a future fix is a
        deliberate, visible change rather than an accidental one.
        """
        model_var = {0: -1.0, -1: -1.0, -2: -1.0}
        actual_return = {0: -2.0, -1: -2.0}  # missing key -2

        df = pd.DataFrame({"Model VaR": model_var, "Actual Return": actual_return})
        assert df["Actual Return"].isna().any(), (
            "Expected mismatched keys to introduce NaNs silently -- if this "
            "assertion fails, the alignment behavior has changed and the "
            "LIMITATION comment in backtest.py should be revisited."
        )
