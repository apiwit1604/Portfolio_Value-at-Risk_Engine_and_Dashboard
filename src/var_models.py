"""
VaR Models
==========
Three independent VaR estimation methods, plus the shared parametric
performance function used by the optimizers in optimize.py.

1. Parametric (Variance-Covariance)
   performance_portfolio computes:
     - Portfolio return: w @ mu
     - Portfolio risk (std): sqrt(w @ sigma @ w)
     - VaR: return + z * risk, where z = norm.ppf(1 - confidence) is the
       left-tail quantile of the standard normal.
       E.g. confidence = 0.95 -> z = norm.ppf(0.05) ~ -1.645

2. Historical
   Finds VaR from the empirical quantile of actual historical returns,
   without assuming returns are normally distributed.

3. Monte Carlo
   Simulates correlated log-returns using Cholesky decomposition of the
   correlation matrix, then finds the quantile of the SIMULATED PORTFOLIO
   return (not a weighted sum of per-asset quantiles -- these are not
   equal in general, because VaR is not a linear operator on marginal
   distributions).

   Steps:
   1. L = cholesky(corr) -- decompose the correlation matrix to generate
      correlated shocks
   2. Draw Z_independent as independent standard normals, then transform
      to Z_correlated = Z_independent @ L.T
   3. Simulate each asset's log return via the GBM formula:
      (mu - 0.5*sigma^2) + sigma*Z_correlated
   4. Combine into the portfolio return: log_returns @ w_port
   5. Take the quantile of the simulated portfolio
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def performance_portfolio(
    w_port: np.ndarray,
    mu_port: pd.Series,
    sigma_port: pd.DataFrame,
    confidence: float,
) -> dict:
    """
    Parametric (variance-covariance) portfolio return, risk, and VaR.

    z is the left-tail quantile (1 - confidence) of the standard normal.
    E.g. confidence=0.95 -> z = norm.ppf(0.05) ~ -1.645
    """
    portfolio_return = w_port @ mu_port
    portfolio_risk = np.sqrt(w_port @ sigma_port @ w_port)

    z_score = norm.ppf(1 - confidence)
    portfolio_var = portfolio_return + z_score * portfolio_risk

    return {
        "portfolio_weight": w_port,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "portfolio_var": portfolio_var,
    }


def historical_var(returns_daily: pd.DataFrame, w_port: np.ndarray, alpha: float) -> float:
    """
    Historical VaR: empirical quantile of the weighted portfolio return
    (NOT a weighted sum of per-asset quantiles).
    """
    portfolio_returns = returns_daily @ w_port
    return portfolio_returns.quantile(alpha)


def monte_carlo_var(
    mu_daily: np.ndarray,
    sigma_daily: np.ndarray,
    corr: np.ndarray,
    w_port: np.ndarray,
    horizon: int,
    alpha: float,
    n_sims: int = 300_000,
) -> float:
    """
    Monte Carlo VaR via simulated correlated log-returns (GBM).

    Finds the quantile of the SIMULATED PORTFOLIO return
    (w @ simulated_asset_returns), not a weighted sum of per-asset
    quantiles -- these are not equal in general, because VaR is not a
    linear operator on marginal distributions.
    """
    n_assets = len(w_port)
    mu_h = mu_daily * horizon
    sigma_h = sigma_daily * np.sqrt(horizon)

    L = np.linalg.cholesky(corr)
    z_independent = np.random.standard_normal((n_sims, n_assets))
    z_correlated = z_independent @ L.T

    log_returns = (mu_h - 0.5 * sigma_h**2) + sigma_h * z_correlated  # (n_sims, n_assets)
    portfolio_sim_returns = log_returns @ w_port

    return np.quantile(portfolio_sim_returns, alpha)
