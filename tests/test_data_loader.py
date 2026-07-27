"""
Tests for var_risk_engine.data_loader.

load_close_prices hits a live network API (Yahoo Finance via yfinance) and
is intentionally NOT covered here with a real network call -- that belongs
in an integration test run separately, not in the offline unit suite. What
IS covered:
    1. compute_returns's math, fully offline.
    2. load_close_prices's per-ticker try/except isolation, using a fake
       yf.Ticker so one failing ticker doesn't take down the whole fetch.
"""

import numpy as np
import pandas as pd

from var_risk_engine.data_loader import compute_returns, load_close_prices


class TestComputeReturns:
    def test_daily_log_return_matches_formula(self):
        prices = pd.DataFrame(
            {"X": [100.0, 105.0, 103.0, 110.0]},
            index=pd.bdate_range("2024-01-01", periods=4),
        )
        returns_daily, _ = compute_returns(prices)

        expected = np.log(prices / prices.shift(1)).dropna()
        pd.testing.assert_frame_equal(returns_daily, expected)

    def test_daily_returns_drop_first_row(self):
        prices = pd.DataFrame(
            {"X": [100.0, 105.0, 103.0]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        returns_daily, _ = compute_returns(prices)
        assert len(returns_daily) == len(prices) - 1

    def test_annual_return_uses_simple_not_log_return(self):
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="B")
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            {"X": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates))))},
            index=dates,
        )
        _, returns_annual = compute_returns(prices)

        annual_close = prices.resample("YE").last()
        expected = (annual_close / annual_close.shift(1) - 1).dropna()
        pd.testing.assert_frame_equal(returns_annual, expected)


class TestLoadClosePrices:
    def test_one_failing_ticker_does_not_crash_the_whole_fetch(self, monkeypatch):
        """Regression test for the try/except fix: the original notebook's
        markdown claimed per-ticker error isolation that the code did not
        actually implement. This pins down that the fix works: a ticker
        that raises should be skipped (with a warning), not propagate.
        """

        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, start=None, end=None, interval="1d"):
                if self.symbol == "BROKEN":
                    raise RuntimeError("simulated fetch failure")
                dates = pd.bdate_range(start=start, end=end)
                return pd.DataFrame({"Close": np.full(len(dates), 100.0)}, index=dates)

        import var_risk_engine.data_loader as data_loader_module

        monkeypatch.setattr(data_loader_module.yf, "Ticker", FakeTicker)

        prices = load_close_prices(["GOOD", "BROKEN"], "2024-01-01", "2024-01-10")

        assert "GOOD" in prices.columns
        assert "BROKEN" not in prices.columns

    def test_all_tickers_succeed_returns_all_columns(self, monkeypatch):
        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, start=None, end=None, interval="1d"):
                dates = pd.bdate_range(start=start, end=end)
                return pd.DataFrame({"Close": np.full(len(dates), 100.0)}, index=dates)

        import var_risk_engine.data_loader as data_loader_module

        monkeypatch.setattr(data_loader_module.yf, "Ticker", FakeTicker)

        prices = load_close_prices(["A", "B", "C"], "2024-01-01", "2024-01-10")
        assert list(prices.columns) == ["A", "B", "C"]
