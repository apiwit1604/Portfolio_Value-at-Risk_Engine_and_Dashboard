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


def time_series_var(
    result, 
    target_capital, 
    new_start_date, 
    n_test, 
    confidence=0.99, 
    window_size=250, 
    mc_simulations=10_000, 
    random_state=None, 
    progress_callback=None
):
    horizon = 1

    # 1. จัดโครงสร้างพอร์ตและดึง Risk Factors ดั้งเดิม
    output_num, output_str, final_output = build_portfolio_risk_matrix(
        result["risk_matrices"], target_capital
    )
    risk_columns = list(final_output["risk"])
    asset_values_0 = pd.Series(result["value_asset"], dtype=float)
    asset_names = list(asset_values_0.index)
    n_assets = len(asset_names)
    n_risks = len(risk_columns)

    # 2. แปลง Asset Sensitivity เป็น Matrix [Assets x Risks] เพื่อ Vectorize ลด O(N^2)
    # mapping index ของ risk_columns เพื่อความรวดเร็ว
    risk_idx_map = {r: i for i, r in enumerate(risk_columns)}
    asset_risk_matrix = np.zeros((n_assets, n_risks), dtype=float)

    for a_idx, asset_name in enumerate(asset_names):
        rm = result["risk_matrices"][asset_name]
        grouped_cf = rm.groupby("risk", as_index=False)["adj_cf"].sum()
        for _, row in grouped_cf.iterrows():
            r_name = row["risk"]
            if r_name in risk_idx_map:
                r_idx = risk_idx_map[r_name]
                asset_risk_matrix[a_idx, r_idx] += float(row["adj_cf"])

    # 3. ดึงข้อมูลย้อนหลัง (เพิ่มเป็น 550 Calendar Days เพื่อการันตี 250 Trading Days + Buffer)
    start_dt = pd.to_datetime(new_start_date) - timedelta(days=550)
    fetch_start_str = start_dt.strftime("%Y-%m-%d")

    full_yield_data, x_known = get_fred_yield_curve(fetch_start_str, None)
    _, risk_yields = build_yield_risk(full_yield_data, x_known, output_num["risk"])
    risk_stock = build_stock_risk(output_str["risk"], fetch_start_str, None)

    full_data_risk, _, _ = build_risk_data(risk_yields, risk_stock)
    full_data_risk.index = pd.to_datetime(full_data_risk.index)
    full_data_risk = full_data_risk.reindex(columns=risk_columns)

    # 4. กำหนดขอบเขตข้อมูลสำหรับ Testing
    test_start_ts = pd.to_datetime(new_start_date)
    test_data = full_data_risk.loc[full_data_risk.index >= test_start_ts].head(n_test).copy()

    pre_test_dates = full_data_risk.index[full_data_risk.index < test_start_ts]
    if len(pre_test_dates) == 0:
        raise ValueError("ไม่มีข้อมูล Market Risk ก่อน new_start_date")

    previous_date = pre_test_dates[-1]
    current_asset_values = asset_values_0.copy().values # ใช้ Numpy Array เพื่อความเร็ว
    initial_asset_values = asset_values_0.values

    var_p, var_h, var_m = [], [], []
    realized_returns, valid_dates = [], []
    adjusted_weight_history = []

    total_steps = len(test_data.index)

    # 5. Main Simulation Loop
    for step_idx, current_date in enumerate(test_data.index):
        if progress_callback:
            progress_callback(step_idx, total_steps)

        historical_slice = full_data_risk.loc[full_data_risk.index < current_date]
        rolling_window = historical_slice.tail(window_size)

        if len(rolling_window) < window_size:
            continue

        current_portfolio_value = float(np.sum(current_asset_values))
        if not np.isfinite(current_portfolio_value) or current_portfolio_value <= 0:
            continue

        # คำนวณ Value Scale ของแต่ละสินทรัพย์ [Assets,]
        # หากมูลค่าเริ่มต้นเป็น 0 ให้ scale เป็น 0
        with np.errstate(divide='ignore', invalid='ignore'):
            value_scales = np.where(initial_asset_values != 0, current_asset_values / initial_asset_values, 0.0)

        # Vectorized Adjustment Weight Calculation:
        # Scale Sensitivities ของแต่ละสินทรัพย์ -> Sum รวมทั้งพอร์ต -> หารด้วย Port Value
        scaled_asset_risk = asset_risk_matrix * value_scales[:, np.newaxis] # [Assets x Risks]
        daily_adj_w = scaled_asset_risk.sum(axis=0) / current_portfolio_value # [Risks,]

        # Calculation Metrics
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
        
        # ส่งผ่าน mc_simulations และ random_state ตาม Parameter ใหม่
        vm = monte_carlo_var(
            rolling_window,
            daily_final_output,
            horizon,
            n_simulations=mc_simulations,
            confidence=confidence,
            random_state=random_state
        )

        # 6. Mark-to-Market PnL Calculation (แก้ไข Scaling Bug)
        daily_change = (
            full_data_risk.loc[current_date] - full_data_risk.loc[previous_date]
        ).values # [Risks,]

        # PnL ของแต่ละ Asset = Scaled Sensitivities dot Daily Change
        # ใช้ scaled_asset_risk เพื่อให้สอดคล้องกับ Mark-to-Market จริง ณ มูลค่าปัจจุบัน
        pnl_by_asset = np.nan_to_num(scaled_asset_risk @ daily_change) # [Assets,]

        total_pnl = float(np.sum(pnl_by_asset))
        realized_return = total_pnl / current_portfolio_value

        # อัปเดตมูลค่าพอร์ต Buy-and-hold สำหรับวันถัดไป
        current_asset_values += pnl_by_asset

        # บันทึกผลลัพธ์
        var_p.append(vp)
        var_h.append(vh)
        var_m.append(vm)
        realized_returns.append(realized_return)
        adjusted_weight_history.append(daily_adj_w.copy())
        valid_dates.append(current_date)

        previous_date = current_date

    if progress_callback:
        progress_callback(total_steps, total_steps)

    output_df = pd.DataFrame({
        "return_port": realized_returns,
        "var_parametric": var_p,
        "var_historical": var_h,
        "var_mc": var_m,
    }, index=valid_dates)

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
