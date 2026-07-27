"""
Tests for var_risk_engine.reporting.

Display code is otherwise untested in this project, so the bar here is
lower than for the calculation modules: mainly "does it run without
crashing on realistic input, and does the printed output contain the
numbers it's supposed to contain." matplotlib is forced to a non-interactive
backend so plot_rolling_var can run in a headless test environment.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from var_risk_engine.reporting import plot_rolling_var, show_portfolio, show_var_comparison


@pytest.fixture
def sample_portfolio():
    return {
        "portfolio_weight": np.array([0.5, 0.3, 0.2]),
        "portfolio_return": 0.01234,
        "portfolio_risk": 0.05678,
        "portfolio_var": -0.04321,
    }


class TestShowPortfolio:
    def test_runs_without_error_and_prints_key_numbers(self, sample_portfolio, capsys):
        show_portfolio("Test Portfolio", sample_portfolio, ["A", "B", "C"])
        captured = capsys.readouterr()

        assert "Test Portfolio" in captured.out
        assert "1.23" in captured.out  # portfolio_return * 100, rounded
        assert "-4.32" in captured.out  # portfolio_var * 100, rounded

    def test_prints_all_tickers(self, sample_portfolio, capsys):
        show_portfolio("Test Portfolio", sample_portfolio, ["A", "B", "C"])
        captured = capsys.readouterr()
        for ticker in ["A", "B", "C"]:
            assert ticker in captured.out


class TestShowVarComparison:
    def test_prints_all_labels_and_values(self, capsys):
        show_var_comparison("VaR Comparison", {"initial": -0.05, "minrisk": -0.02})
        captured = capsys.readouterr()

        assert "VaR Comparison" in captured.out
        assert "initial" in captured.out
        assert "minrisk" in captured.out
        assert "-5.00" in captured.out
        assert "-2.00" in captured.out


class TestPlotRollingVar:
    def test_runs_without_error(self):
        result_var = pd.DataFrame(
            {
                "Actual Return": [-0.01, 0.02, -0.03],
                "Parametric VaR": [-0.04, -0.04, -0.04],
                "Historical VaR": [-0.05, -0.05, -0.05],
                "Monte VaR": [-0.045, -0.045, -0.045],
            },
            index=[0, -1, -2],
        )
        # Should not raise -- this is the main thing worth checking for a
        # plotting function in an automated test.
        plot_rolling_var(result_var)
