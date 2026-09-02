# -*- coding: utf-8 -*-
"""
Portfolio construction & optimization — one function, four strategies.

strategy=
    "given"      : use the weights as written in `portfolio` (no
                   optimization).
    "min_risk"   : minimize portfolio return volatility.
    "min_var"    : minimize the size of the loss at the VaR confidence
                   level (== maximize the signed parametric VaR, since
                   VaR is usually a negative number).
    "max_sharpe" : maximize (return - risk_free) / volatility.

Pass `esg_target` (0-100) with any strategy above to additionally
require the portfolio's weighted ESG score to be >= esg_target.

Performance note
-----------------
For "min_risk" / "min_var" / "max_sharpe", `scipy.optimize.minimize`
calls `objective()` many times (SLSQP: typically 10-50+ evaluations for
a handful of assets), and every evaluation re-prices the whole
portfolio — including every Yahoo Finance / FRED fetch. That is why
`cache.py` wraps those fetches in `functools.lru_cache`: without it,
optimizing a 3-asset portfolio means 30-150+ live network calls for
data that doesn't change within the optimization, and Yahoo Finance
will eventually rate-limit that.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from .assembly import (
    build_asset_data,
    build_portfolio_risk_matrix,
    build_risk_data,
    build_stock_risk,
    build_yield_risk,
)
from .pricing import validate_portfolio
from .var_models import calculate_parametric_var, esg_score_invest, historical_var, monte_carlo_var, sharpratio


def _price_and_measure(portfolio, weights, target_capital, data, x_known,
                        start_date, end_date, investment_horizon, confidence):
    """
    Price every asset at `weights`, assemble the portfolio risk matrix,
    and compute the parametric return/risk/VaR. This is the expensive
    step (does all the yfinance/FRED-derived pricing) shared by every
    optimizer objective and by the final "price at the optimum" call.
    """
    pricing_asset, data_asset, risk_matrices, asset_values, asset_units = build_asset_data(
        portfolio, weights, target_capital, data, x_known, start_date, end_date
    )
    output_num, output_str, final_output = build_portfolio_risk_matrix(risk_matrices, target_capital)
    _, risk_yields = build_yield_risk(data, x_known, output_num["risk"])
    risk_stock = build_stock_risk(output_str["risk"], start_date, end_date)
    data_risk, data_risk_cov, data_risk_mean = build_risk_data(risk_yields, risk_stock)

    sensitivities = final_output["adj_weight"].values
    portfolio_return, portfolio_risk, portfolio_var_parametric = calculate_parametric_var(
        investment_horizon, sensitivities, data_risk_mean, data_risk_cov, confidence=confidence
    )

    return {
        "pricing_asset": pricing_asset, "data_asset": data_asset,
        "risk_matrices": risk_matrices, "asset_values": asset_values,
        "asset_units": asset_units, "final_output": final_output,
        "data_risk": data_risk, "data_risk_cov": data_risk_cov,
        "data_risk_mean": data_risk_mean,
        "portfolio_return": portfolio_return, "portfolio_risk": portfolio_risk,
        "portfolio_var_parametric": portfolio_var_parametric,
    }


def build_portfolio(portfolio, investment_horizon, target_capital, data, x_known,
                     start_date, end_date, strategy="given",
                     min_weight=0, max_weight=1, confidence=0.99, esg_target=None,
                     mc_simulations=100_000, random_state=None):
    """
    Build a portfolio result under one of four weighting strategies (see
    module docstring), then compute its VaR (Parametric / Historical /
    Monte Carlo), ESG score, and Sharpe ratio.

    Returns a dict with keys: risk_matrices, weight_asset, value_asset,
    pricing_asset, data_asset, units_asset, final_output, data_risk,
    data_risk_cov, data_risk_mean, portfolio_value, portfolio_return,
    portfolio_risk, portfolio_var_parametric, portfolio_var_historical,
    portfolio_var_mc, esg_score, sharp.
    """
    validate_portfolio(portfolio)

    weight_list = [asset["weight"] for asset in portfolio]
    esg_score_list = [asset["esg_score"] for asset in portfolio]

    data_rf = data[0.25]
    latest_date = max(data_rf.keys())
    rf = data_rf[latest_date] * investment_horizon / 252

    if strategy == "given":
        weight_opt_x = np.asarray(weight_list, dtype=float)
    else:
        def objective(weights):
            m = _price_and_measure(portfolio, weights, target_capital, data, x_known,
                                    start_date, end_date, investment_horizon, confidence)
            if strategy == "min_risk":
                return m["portfolio_risk"]
            if strategy == "min_var":
                return -m["portfolio_var_parametric"]
            if strategy == "max_sharpe":
                return -(m["portfolio_return"] - rf) / m["portfolio_risk"]
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Use 'given', 'min_risk', 'min_var', or 'max_sharpe'."
            )

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if esg_target is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: np.dot(w, esg_score_list) - esg_target}
            )

        bounds = [(min_weight, max_weight)] * len(weight_list)
        opt = minimize(objective, weight_list, method="SLSQP", bounds=bounds, constraints=constraints)

        if not opt.success:
            warnings.warn(
                f"Optimizer did not converge for strategy={strategy!r}: {opt.message}. "
                "Check that min_weight/max_weight/esg_target are jointly feasible."
            )
        weight_opt_x = opt.x

    m = _price_and_measure(portfolio, weight_opt_x, target_capital, data, x_known,
                            start_date, end_date, investment_horizon, confidence)

    portfolio_var_historical = historical_var(m["data_risk"], m["final_output"], investment_horizon, confidence)
    portfolio_var_mc = monte_carlo_var(m["data_risk"], m["final_output"], investment_horizon,
                                        n_simulations=mc_simulations, confidence=confidence,
                                        random_state=random_state)

    esg_score = esg_score_invest(weight_opt_x, esg_score_list)
    sharp = sharpratio(m["portfolio_return"], m["portfolio_risk"], rf)

    return {
        "risk_matrices": m["risk_matrices"],
        "weight_asset": weight_opt_x,

        "value_asset": m["asset_values"],
        "pricing_asset": m["pricing_asset"],
        "data_asset": m["data_asset"],
        "units_asset": m["asset_units"],

        "final_output": m["final_output"],
        "data_risk": m["data_risk"],
        "data_risk_cov": m["data_risk_cov"],
        "data_risk_mean": m["data_risk_mean"],

        "portfolio_value": target_capital,
        "portfolio_return": m["portfolio_return"],
        "portfolio_risk": m["portfolio_risk"],

        "portfolio_var_parametric": m["portfolio_var_parametric"],
        "portfolio_var_historical": portfolio_var_historical,
        "portfolio_var_mc": portfolio_var_mc,
        "esg_score": esg_score,
        "sharp": sharp,
    }
