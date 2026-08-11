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
    close_prices = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start=start, end=end, interval="1d")["Close"]
            
            if df.empty:
                print(f"[warning] No data returned for {t}")
                continue
                
            df.index = df.index.tz_localize(None).normalize()
            df.name = t
            close_prices.append(df)
            
        except Exception as e:
            print(f"[warning] could not fetch {t}: {e}")

    if not close_prices:
        print("[error] No data fetched for any tickers.")
        return pd.DataFrame()
        
    prices = pd.concat(close_prices, axis=1)
    return prices

def compute_returns(close_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily log returns and annual simple returns.
    """
    returns_daily = np.log(close_prices / close_prices.shift(1)).dropna()
    annual_close = close_prices.resample("YE").last()
    returns_annual = (annual_close / annual_close.shift(1) - 1).dropna()
    return returns_daily, returns_annual
