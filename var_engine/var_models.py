# -*- coding: utf-8 -*-
"""VaR engines: Parametric, Historical, Monte Carlo — plus ESG score and Sharpe ratio."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def calculate_parametric_var(investment_horizon, risk_weights, risk_mean, risk_cov, confidence=0.99):
    """
    Parametric (delta-normal / variance-covariance) VaR: assumes risk
    factor changes are jointly normal, so the portfolio return is too.

    Returns (portfolio_return, portfolio_risk, portfolio_var), all
    scaled to `investment_horizon` trading days.
    """
    weights = np.asarray(risk_weights)
    mean = np.asarray(risk_mean)

    z_score = norm.ppf(1 - confidence)

    portfolio_return = (weights @ mean) * investment_horizon
    portfolio_risk = np.sqrt(weights @ risk_cov @ weights) * np.sqrt(investment_horizon)
    portfolio_var = portfolio_return + z_score * portfolio_risk

    return portfolio_return, portfolio_risk, portfolio_var


def historical_var(data_risk, final_output, investment_horizon, confidence=0.99):
    """
    Historical VaR: apply the *current* risk sensitivities to
    *historical* risk-factor changes, then take the empirical
    (1 - confidence) quantile of the resulting portfolio-return series.
    Makes no distributional assumption, but is limited by however much
    history `data_risk` contains.
    """
    w_risk = final_output["adj_weight"].values
    historical_port_returns = (data_risk.values @ w_risk) * np.sqrt(investment_horizon)
    return np.percentile(historical_port_returns, (1 - confidence) * 100)


def monte_carlo_var(data_risk, final_output, investment_horizon, n_simulations=100_000,
                     confidence=0.99, random_state=None):
    """
    Monte Carlo VaR: fit a multivariate normal to the historical risk
    factors, simulate `n_simulations` correlated draws over the
    horizon, and take the empirical (1 - confidence) quantile of the
    simulated portfolio returns.

    Pass `random_state` (an int or np.random.Generator) for reproducible
    results — the original script did not seed this, so re-running it
    gives slightly different numbers every time.
    """
    w_risk = final_output["adj_weight"].values

    cov_matrix = data_risk.cov().values
    mean_vector = data_risk.mean().values

    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state
    simulated_changes = rng.multivariate_normal(
        mean_vector * investment_horizon,
        cov_matrix * investment_horizon,
        n_simulations,
    )
    simulated_port_returns = simulated_changes @ w_risk

    return np.percentile(simulated_port_returns, (1 - confidence) * 100)


def esg_score_invest(weight_list, esg_score_list):
    """Weighted-average ESG score of the portfolio."""
    return np.dot(weight_list, esg_score_list)


def sharpratio(portfolio_return, portfolio_risk, rf):
    """Sharpe ratio: excess return over the risk-free rate, per unit of risk."""
    return (portfolio_return - rf) / portfolio_risk
