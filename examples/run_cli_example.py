# -*- coding: utf-8 -*-
"""
Console example — runs all 5 weighting strategies + backtests each one,
same as the original notebook's bottom cell, using the refactored
var_engine package.

Run from the repo root:
    python examples/run_cli_example.py

This talks to Yahoo Finance and FRED live, so it needs internet access
and will take a few minutes (the "Compare all strategies" combination
is the slow path described in the README).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import var_engine`

from var_engine import build_portfolio, get_fred_yield_curve, kupiec_test, time_series_var
from var_engine.display import display_portfolio_result

# ---- Same example portfolio as the dashboard's default ----
portfolio = [
    {"name": "S1", "type": "STK", "stock_name": "NVDA", "weight": 0.70, "esg_score": 75},
    {"name": "E1", "type": "EPO", "stock_name": "NVDA", "weight": 0.20, "esg_score": 80, "K": 250, "T": 1},
    {"name": "G", "type": "FX", "fx_name": "THB=X", "weight": 0.10, "esg_score": 90},
]

start_date = "2025-01-01"
end_date = "2025-12-31"
target_capital = 100_000
investment_horizon = 252
confidence = 0.99
esg_target = 80

data, x_known = get_fred_yield_curve(start_date, end_date)

runs = [
    ("BASE CASE — Portfolio at Stated Weights", "given", None),
    ("MINIMUM RISK PORTFOLIO", "min_risk", None),
    ("MINIMUM VaR PORTFOLIO", "min_var", None),
    ("MAXIMUM SHARPE RATIO PORTFOLIO", "max_sharpe", None),
    (f"MAXIMUM SHARPE RATIO PORTFOLIO (ESG >= {esg_target})", "max_sharpe", esg_target),
]

results = {}
for label, strategy, esg_target_run in runs:
    print("\n" + "=" * 65)
    print(f" {label}")
    print("=" * 65)

    res = build_portfolio(
        portfolio, investment_horizon, target_capital, data, x_known,
        start_date, end_date, strategy=strategy,
        min_weight=0, max_weight=1,
        confidence=confidence, esg_target=esg_target_run,
        random_state=42,  # reproducible Monte Carlo VaR; drop for fresh draws each run
    )
    display_portfolio_result(res)
    results[label] = res

# ---- Backtest each strategy's resulting weights ----
N_test = 60
new_start_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

for label, res in results.items():
    print("\n" + "#" * 65)
    print(f"# BACKTEST RESULTS — {label}")
    print("#" * 65)

    output = time_series_var(res, target_capital, new_start_date, N_test, confidence=confidence,
                              random_state=42)
    if output.empty:
        print("No valid out-of-sample days (not enough history after end_date). Skipping.")
        continue

    for m_label, col in [("Parametric VaR", "var_parametric"),
                          ("Historical VaR", "var_historical"),
                          ("Monte Carlo VaR", "var_mc")]:
        print("-" * 40)
        print(f"kupiec_test of {m_label}")
        print("-" * 40)
        r = kupiec_test(output[col], output["return_port"], confidence)
        print(f"LR: {r['lr_stat']:.4f}")
        print(f"Critical Value ({r['expected_rate'] * 100:.2f}%): {r['critical_value']:.4f}")
        print(f"p-value: {r['p_value']:.4f}")
        print("result test: Fail to reject H0 (Pass)" if r["passed"] else "result test: reject H0 (Fail)")
        print()
