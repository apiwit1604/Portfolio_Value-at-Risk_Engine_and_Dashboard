"""
Portfolio Statistics (Consistent Scaling)
==========================================
scale_mu and scale_cov are the single place where mean and covariance are
scaled from daily to the desired horizon -- used consistently everywhere
in the project instead of writing `* investment_horizon` in multiple
scattered places (a source of inconsistency in earlier versions of this
codebase).

  - Mean scales linearly with horizon (mu * horizon)
  - Covariance scales linearly with horizon as well (cov * horizon),
    because Var(n-day return) = n x Var(1-day return) under the i.i.d.
    returns assumption.
"""

import pandas as pd


def scale_mu(daily_mean: pd.Series, horizon: int) -> pd.Series:
    """
    Scale daily mean log-return to the given horizon (linear in time).
    """
    return daily_mean * horizon


def scale_cov(daily_cov: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Scale daily covariance matrix to the given horizon (linear in time).
    """
    return daily_cov * horizon
