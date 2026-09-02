# -*- coding: utf-8 -*-
"""
Caching layer for external data fetches (Yahoo Finance, FRED).
================================================================

Why this file exists
---------------------
In the original notebook, every pricing/risk function called `yfinance`
or FRED directly, with no caching. That is fine for a single top-to-bottom
notebook run, but it breaks down anywhere the same (ticker, start, end)
is fetched more than once — which happens in two places in this engine:

1. `build_portfolio(..., strategy="min_risk"/"min_var"/"max_sharpe")` calls
   `scipy.optimize.minimize`, and its `objective()` re-prices the *entire*
   portfolio (i.e. re-fetches every stock/FX ticker and the FRED curve) on
   **every single optimizer iteration**. SLSQP routinely takes 10-50+
   evaluations, so optimizing a 3-asset portfolio can mean 30-150+ network
   calls to Yahoo Finance for data that never changes within that call.
2. The Streamlit app re-runs top-to-bottom on every widget interaction, so
   without caching, changing an unrelated slider re-fetches all market data.

Both are real reliability risks, not just slowness: Yahoo Finance rate-limits
aggressively, so an uncached optimization run can and does fail intermittently
with HTTP 429s.

What this module does
----------------------
Wraps the two network-touching primitives (`yfinance` price history and the
FRED CSV endpoint) in `functools.lru_cache`, keyed on the exact arguments.
This changes *nothing* about the math — same inputs still produce the same
outputs — it just guarantees each (ticker, start, end) or (series_id, start,
end) combination is fetched at most once per process.

`clear_cache()` is exposed for the Streamlit app so users can force a refresh.
"""

from __future__ import annotations

import functools
import io

import pandas as pd
import requests
import yfinance as yf

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@functools.lru_cache(maxsize=256)
def fetch_stock_history(ticker: str, start_date: str, end_date: str | None) -> pd.Series:
    """
    Cached wrapper around `yfinance`'s daily close price history.

    Returns a pd.Series of Close prices (tz-naive DatetimeIndex, ascending).
    Raises ValueError if the ticker returns no data (typo'd/delisted ticker).
    """
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
    if hist.empty or "Close" not in hist:
        raise ValueError(
            f"No price data returned for ticker '{ticker}' between "
            f"{start_date} and {end_date}. Check the ticker on "
            f"https://finance.yahoo.com/ and that the date range has trading days."
        )
    close = hist["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


@functools.lru_cache(maxsize=64)
def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Cached wrapper around one FRED series CSV download.

    Returns a pd.Series indexed by observation date (DatetimeIndex), values
    as raw (not yet /100'd) numbers, NaNs coerced from FRED's "." markers.
    """
    url = f"{FRED_CSV_URL}?id={series_id}&cosd={start_date}&coed={end_date}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    temp = pd.read_csv(io.StringIO(resp.text))
    temp["observation_date"] = pd.to_datetime(temp["observation_date"])
    temp[series_id] = pd.to_numeric(temp[series_id], errors="coerce")
    return temp.set_index("observation_date")[series_id]


def clear_cache() -> None:
    """Drop all cached market-data fetches (e.g. a Streamlit 'Refresh data' button)."""
    fetch_stock_history.cache_clear()
    fetch_fred_series.cache_clear()
