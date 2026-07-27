"""
Portfolio statistics with consistent scaling across the whole project.

scale_mu and scale_cov are the single place where daily mean/covariance are
scaled to the desired horizon. Using these shared functions everywhere
avoids `* investment_horizon` being written ad hoc in multiple places
(a source of inconsistency in earlier versions of this codebase).

    - Mean scales linearly with horizon: mu * horizon
    - Covariance scales linearly with horizon as well: cov * horizon,
      because Var(n-day return) = n * Var(1-day return) under the i.i.d.
      returns assumption.
"""

import pandas as pd


def scale_mu(daily_mean: pd.Series, horizon: int) -> pd.Series:
    """Scale daily mean log-return to the given horizon (linear in time)."""
    return daily_mean * horizon


def scale_cov(daily_cov: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Scale daily covariance matrix to the given horizon (linear in time)."""
    return daily_cov * horizon
