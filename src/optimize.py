"""
Portfolio Optimization
======================
Two functions find portfolio weights via scipy.optimize.minimize (SLSQP),
subject to: weights sum to 1, and each asset's weight lies within
[min_weight, max_weight].

find_min_risk_portfolio -- Global Minimum-Variance Portfolio
  Objective: minimize sqrt(w @ sigma @ w) only (expected return is ignored
  entirely).

find_min_loss_portfolio (previously named find_min_var_portfolio --
renamed to reduce confusion)
  VaR in this project is stored as a NEGATIVE value (more negative = worse
  loss). So "maximizing VaR" mathematically is the same as "minimizing the
  magnitude of the tail loss". The old name find_min_var_portfolio made
  this easy to misread as "minimizing VaR" (which would actually mean
  approaching -inf, the worst outcome) -- hence the rename to
  find_min_loss_portfolio, which matches what the function actually does.
  Objective: minimize -(return + z * risk)
           = maximize (return + z * risk)
           = minimize tail loss
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .var_models import performance_portfolio


def find_min_risk_portfolio(
    mu_port: pd.Series,
    sigma_port: pd.DataFrame,
    w0: np.ndarray,
    min_weight: float,
    max_weight: float,
    confidence: float,
) -> dict:
    """
    Global minimum-variance portfolio.

    Note: mu_port and confidence are not used in the objective function.
    This function minimizes risk only, ignoring expected return, subject to
    weights summing to 1 and per-asset bounds.
    """

    def objective(weights: np.ndarray) -> float:
        return np.sqrt(weights @ sigma_port @ weights)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(min_weight, max_weight)] * len(w0)

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    return performance_portfolio(result.x, mu_port, sigma_port, confidence)


def find_min_loss_portfolio(
    mu_port: pd.Series,
    sigma_port: pd.DataFrame,
    w0: np.ndarray,
    min_weight: float,
    max_weight: float,
    confidence: float,
) -> dict:
    """
    Find the portfolio weights that minimize tail loss (= maximize
    parametric VaR).

    VaR in this project is stored as a negative value
    (portfolio_return + z*risk, where z < 0). So maximizing VaR pushes the
    VaR value toward 0, i.e. minimizes the magnitude of the tail loss (this
    is NOT the same as "minimizing VaR" per the old function name, which
    would mean the worst possible loss -- see module docstring above).

    Subject to weights summing to 1 and per-asset bounds.
    """

    z = norm.ppf(1 - confidence)

    def objective(weights: np.ndarray) -> float:
        portfolio_return = weights @ mu_port
        portfolio_risk = np.sqrt(weights @ sigma_port @ weights)
        var = portfolio_return + z * portfolio_risk
        return -var  # minimize negative VaR == maximize VaR == minimize tail loss

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(min_weight, max_weight)] * len(w0)

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    return performance_portfolio(result.x, mu_port, sigma_port, confidence)
