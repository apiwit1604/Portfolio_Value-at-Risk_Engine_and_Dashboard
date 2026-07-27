"""
Price loading and return computation.

load_close_prices: fetches each ticker's daily close price from Yahoo Finance
and merges them into a single DataFrame. Each ticker is wrapped in its own
try/except so that one failed ticker (e.g. delisted, or rate-limited) prints
a warning and is skipped, instead of crashing the entire script.
The except clause is intentionally broad (catches Exception, not a specific
error type) because yfinance can fail in several distinct ways depending on
the underlying cause -- network errors, a missing "Close" column if the
response shape changes, HTTP errors for delisted/invalid tickers, etc. -- and
all of them should be handled the same way here: warn and skip that ticker.

compute_returns: computes
    - Daily log return:    ln(P_t / P_{t-1})
      Log returns are used because they are additive across time.
    - Annual simple return: uses year-end close prices, simple return
      (P_t / P_{t-1}) - 1, to show a yearly overview of performance.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def load_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch each ticker's daily close price and merge into one DataFrame."""
    series_list = []
    for ticker in tickers:
        try:
            close = yf.Ticker(ticker).history(start=start, end=end, interval="1d")["Close"]
            close.name = ticker
            series_list.append(close)
        except Exception as exc:  # noqa: BLE001 -- broad on purpose, see module docstring
            print(f"[WARNING] Could not fetch data for {ticker}: {exc}")
    prices = pd.concat(series_list, axis=1)
    return prices


def compute_returns(close_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily log returns and annual simple returns."""
    returns_daily = np.log(close_prices / close_prices.shift(1)).dropna()
    annual_close = close_prices.resample("YE").last()
    returns_annual = (annual_close / annual_close.shift(1) - 1).dropna()
    return returns_daily, returns_annual
