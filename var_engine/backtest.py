# -*- coding: utf-8 -*-
"""Backtesting: rolling VaR re-estimation + Kupiec Proportion-of-Failures test."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .assembly import build_portfolio_risk_matrix, build_stock_risk, build_yield_risk, build_risk_data
from .market_data import get_fred_yield_curve
from .var_models import calculate_parametric_var, historical_var, monte_carlo_var


def time_series_var(result, target_capital, new_start_date, n_test, confidence=0.99,
                     window_size=250, mc_simulations=10_000, random_state=None,
                     progress_callback=None):
    """
    Roll a `window_size`-day estimation window forward one day at a
    time over `n_test` out-of-sample days (starting at
    `new_start_date`), recomputing Parametric / Historical / Monte
    Carlo VaR at each step with the position sensitivities from
    `result` held fixed. Used to backtest whether a portfolio's VaR
    estimates would actually have covered its realized returns.

    `progress_callback(done, total)`, if given, is called after each
    day is processed — useful for a Streamlit progress bar, since this
    function is the slowest one in the engine (up to `n_test` Monte
    Carlo simulations, each `mc_simulations` draws).
    """
    adj_w = np.asarray(result["final_output"]["adj_weight"])
    horizon = 1

    output_num, output_str, _ = build_portfolio_risk_matrix(result["risk_matrices"], target_capital)

    start_dt = pd.to_datetime(new_start_date) - timedelta(days=500)
    fetch_start_str = start_dt.strftime("%Y-%m-%d")

    full_yield_data, x_known = get_fred_yield_curve(fetch_start_str, None)
    _, risk_yields = build_yield_risk(full_yield_data, x_known, output_num["risk"])
    risk_stock = build_stock_risk(output_str["risk"], fetch_start_str, None)

    full_data_risk = pd.concat([risk_yields, risk_stock], axis=1).dropna()
    full_data_risk.index = pd.to_datetime(full_data_risk.index)

    test_start_ts = pd.to_datetime(new_start_date)
    test_data = full_data_risk.loc[full_data_risk.index >= test_start_ts].head(n_test).copy()

    var_p, var_h, var_m = [], [], []
    valid_dates = []
    total = len(test_data)

    for i, current_date in enumerate(test_data.index):
        historical_slice = full_data_risk.loc[full_data_risk.index <= current_date]
        rolling_window = historical_slice.tail(window_size)

        if len(rolling_window) < window_size:
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        mean = rolling_window.mean()
        cov = rolling_window.cov()

        _, _, vp = calculate_parametric_var(horizon, adj_w, mean, cov, confidence)
        vh = historical_var(rolling_window, result["final_output"], horizon, confidence)
        vm = monte_carlo_var(rolling_window, result["final_output"], horizon,
                              n_simulations=mc_simulations, confidence=confidence,
                              random_state=random_state)

        var_p.append(vp)
        var_h.append(vh)
        var_m.append(vm)
        valid_dates.append(current_date)

        if progress_callback:
            progress_callback(i + 1, total)

    output_df = pd.DataFrame({
        "return_port": test_data.loc[valid_dates].dot(adj_w),
        "var_parametric": var_p,
        "var_historical": var_h,
        "var_mc": var_m,
    }, index=valid_dates)

    return output_df

def kupiec_test(model_var, actual_return, confidence):
    """
    Kupiec Proportion-of-Failures (POF) backtest: checks whether the
    observed VaR breach rate matches the theoretical rate (1 -
    confidence) via a likelihood-ratio test against chi-squared(1).

    Returns a dict: n_obs, n_breaches, breach_rate, lr_stat,
    critical_value, p_value, passed (bool).
    """
    result_test = pd.DataFrame({"Model VaR": model_var, "Actual Return": actual_return})

    x = int(np.sum(result_test["Actual Return"] < result_test["Model VaR"]))
    N = len(model_var)
    p = 1 - confidence          # theoretical exception rate (alpha)
    p_hat = x / N if N else 0.0  # observed exception rate

    # Note: at x=0 or x=N, Python's 0**0 == 1 convention keeps this finite
    # (no log(0) domain error) — verified, so no special-casing needed.
    lr_pof = -2 * np.log(
        (((1 - p) ** (N - x)) * (p ** x)) / (((1 - p_hat) ** (N - x)) * (p_hat ** x))
    )

    alpha = p
    crit = chi2.ppf(1 - alpha, df=1)
    p_value = 1 - chi2.cdf(lr_pof, df=1)

    return {
        "n_obs": N,
        "n_breaches": x,
        "breach_rate": p_hat,
        "expected_rate": p,
        "lr_stat": lr_pof,
        "critical_value": crit,
        "p_value": p_value,
        "passed": bool(p_value > alpha),
    }


def print_kupiec_test(model_var, actual_return, confidence):
    """Console-friendly wrapper around kupiec_test() — prints the same summary the original script did."""
    r = kupiec_test(model_var, actual_return, confidence)
    print(f"LR: {r['lr_stat']:.4f}")
    print(f"Critical Value ({r['expected_rate'] * 100:.2f}%): {r['critical_value']:.4f}")
    print(f"p-value: {r['p_value']:.4f}")
    print("result test: Fail to reject H0 (Pass)" if r["passed"] else "result test: reject H0 (Fail)")
    return r
