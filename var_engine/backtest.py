# -*- coding: utf-8 -*-
"""Backtesting: rolling VaR re-estimation + Kupiec Proportion-of-Failures test."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .assembly import build_portfolio_risk_matrix, build_stock_risk, build_yield_risk
from .market_data import get_fred_yield_curve
from .var_models import calculate_parametric_var, historical_var, monte_carlo_var


def time_series_var(result, target_capital, new_start_date, N_test, confidence=0.99, window_size=250):
    """
    Roll a `window_size`-day estimation window forward one day at a
    time over `N_test` out-of-sample days (starting at `new_start_date`).

    Unlike the original implementation, the portfolio is treated as a
    buy-and-hold investment: asset values drift with daily market moves,
    so the effective adjusted risk weights are recalculated each day.

    The existing VaR engines and all other functions are unchanged.
    """
    horizon = 1

    # Original risk-factor ordering and initial sensitivities.
    output_num, output_str, final_output = build_portfolio_risk_matrix(
        result["risk_matrices"], target_capital
    )
    risk_columns = list(final_output["risk"])
    asset_values_0 = pd.Series(result["value_asset"], dtype=float)
    asset_names = list(asset_values_0.index)

    # Keep each asset's risk sensitivities so that its market value can be
    # marked to market using the same linear risk-factor framework already
    # used elsewhere in the model.
    asset_risk = {}
    for asset_name in asset_names:
        rm = result["risk_matrices"][asset_name].copy()
        asset_risk[asset_name] = rm.groupby("risk", as_index=False)["adj_cf"].sum()

    start_dt = pd.to_datetime(new_start_date) - timedelta(days=366)
    fetch_start_str = start_dt.strftime("%Y-%m-%d")

    full_yield_data, x_known = get_fred_yield_curve(fetch_start_str, None)
    _, risk_yields = build_yield_risk(full_yield_data, x_known, output_num["risk"])
    risk_stock = build_stock_risk(output_str["risk"], fetch_start_str, None)

    full_data_risk, _, _ = build_risk_data(risk_yields, risk_stock)
    full_data_risk.index = pd.to_datetime(full_data_risk.index)
    full_data_risk = full_data_risk.reindex(columns=risk_columns)

    test_start_ts = pd.to_datetime(new_start_date)
    test_data = full_data_risk.loc[full_data_risk.index >= test_start_ts].head(N_test).copy()

    # The first test-day return is measured from the last available
    # pre-test observation, so the weight used for that day's VaR is the
    # actual weight held at the start of the day.
    pre_test_dates = full_data_risk.index[full_data_risk.index < test_start_ts]
    if len(pre_test_dates) == 0:
        raise ValueError("No market-risk observation exists before new_start_date.")

    previous_date = pre_test_dates[-1]
    current_asset_values = asset_values_0.copy()

    var_p, var_h, var_m = [], [], []
    realized_returns, valid_dates = [], []
    adjusted_weight_history = []

    for current_date in test_data.index:
        # VaR must use information available before the realized return.
        historical_slice = full_data_risk.loc[full_data_risk.index < current_date]
        rolling_window = historical_slice.tail(window_size)

        if len(rolling_window) < window_size:
            continue

        # Recalculate today's adjusted risk weights from the portfolio
        # value held at the start of the day.
        current_portfolio_value = float(current_asset_values.sum())
        if not np.isfinite(current_portfolio_value) or current_portfolio_value <= 0:
            continue

        daily_adj_w = np.zeros(len(risk_columns), dtype=float)

        for asset_name in asset_names:
            initial_value = float(asset_values_0[asset_name])
            if initial_value == 0:
                continue

            value_scale = float(current_asset_values[asset_name]) / initial_value
            rm = asset_risk[asset_name]

            for _, row in rm.iterrows():
                matches = [
                    i for i, risk in enumerate(risk_columns)
                    if risk == row["risk"]
                ]
                for i in matches:
                    daily_adj_w[i] += (
                        float(row["adj_cf"])
                        * value_scale
                        / current_portfolio_value
                    )

        mean = rolling_window.mean()
        cov = rolling_window.cov()

        _, _, vp = calculate_parametric_var(
            horizon, daily_adj_w, mean, cov, confidence
        )

        daily_final_output = final_output.copy()
        daily_final_output["adj_weight"] = daily_adj_w

        vh = historical_var(
            rolling_window, daily_final_output, horizon, confidence
        )
        vm = monte_carlo_var(
            rolling_window,
            daily_final_output,
            horizon,
            n_simulations=10_000,
            confidence=confidence,
        )

        # Mark each asset to market using the same risk sensitivities used
        # by the original portfolio risk matrix.
        daily_change = (
            full_data_risk.loc[current_date]
            - full_data_risk.loc[previous_date]
        )

        pnl_by_asset = {}
        for asset_name in asset_names:
            pnl = 0.0
            for _, row in asset_risk[asset_name].iterrows():
                risk = row["risk"]
                if risk in daily_change.index and pd.notna(daily_change[risk]):
                    pnl += float(row["adj_cf"]) * float(daily_change[risk])
            pnl_by_asset[asset_name] = pnl

        total_pnl = float(sum(pnl_by_asset.values()))
        realized_return = total_pnl / current_portfolio_value

        # Buy-and-hold: today's end-of-day values become tomorrow's
        # starting values, so the adjusted weights naturally drift.
        for asset_name, pnl in pnl_by_asset.items():
            current_asset_values[asset_name] += pnl

        var_p.append(vp)
        var_h.append(vh)
        var_m.append(vm)
        realized_returns.append(realized_return)
        adjusted_weight_history.append(daily_adj_w.copy())
        valid_dates.append(current_date)

        previous_date = current_date

    output_df = pd.DataFrame({
        "return_port": realized_returns,
        "var_parametric": var_p,
        "var_historical": var_h,
        "var_mc": var_m,
    }, index=valid_dates)

    # Preserve the existing output columns while making the daily adjusted
    # risk weights available for inspection.
    if adjusted_weight_history:
        output_df.attrs["adjusted_weights"] = pd.DataFrame(
            adjusted_weight_history,
            index=valid_dates,
            columns=risk_columns,
        )

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
