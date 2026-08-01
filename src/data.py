"""
Data
====
Load price data from Yahoo Finance and compute returns.

load_close_prices: fetches each ticker's daily close price and merges them
into a single DataFrame. Each ticker is wrapped in its own try/except -- if
a given ticker cannot be fetched (e.g. delisted, or rate-limited), a warning
is printed and it is skipped instead of crashing the whole run.

compute_returns computes:
  - Daily log return: ln(P_t / P_{t-1}) -- log returns are used because they
    are additive across time.
  - Annual simple return: uses year-end close prices, simple return
    (P_t/P_{t-1}) - 1, to show a yearly overview of performance.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def load_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch each ticker's daily close price and merge into one DataFrame.
    """
    series_list = []
    for ticker in tickers:
        try:
            close = yf.Ticker(ticker).history(start=start, end=end, interval="1d")["Close"]
            close.name = ticker
            series_list.append(close)
        except Exception as e:
            print(f"[warning] could not fetch {ticker}: {e}")
    prices = pd.concat(series_list, axis=1)
    return prices


def compute_returns(close_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily log returns and annual simple returns.
    """
    returns_daily = np.log(close_prices / close_prices.shift(1)).dropna()
    annual_close = close_prices.resample("YE").last()
    returns_annual = (annual_close / annual_close.shift(1) - 1).dropna()
    return returns_daily, returns_annual
