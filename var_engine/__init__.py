# -*- coding: utf-8 -*-
"""
var_engine — Multi-asset Portfolio Value-at-Risk engine.

Builds a multi-asset portfolio (bonds, stocks, FX, and European
options/forwards), estimates its Value-at-Risk with three methods
(Parametric / Historical / Monte Carlo), optimizes the portfolio's
weights under several objectives, and backtests the resulting VaR
estimates with the Kupiec Proportion-of-Failures test.

See the repository README for the full field reference and a worked
example; see examples/run_cli_example.py for a script version of the
walkthrough below.

Quick start
-----------
    from var_engine import get_fred_yield_curve, build_portfolio, validate_portfolio

    portfolio = [
        {"name": "S1", "type": "STK", "stock_name": "NVDA", "weight": 0.70, "esg_score": 75},
        {"name": "E1", "type": "EPO", "stock_name": "NVDA", "weight": 0.20, "esg_score": 80, "K": 250, "T": 1},
        {"name": "G",  "type": "FX",  "fx_name": "THB=X",   "weight": 0.10, "esg_score": 90},
    ]
    validate_portfolio(portfolio)

    data, x_known = get_fred_yield_curve("2025-01-01", "2025-12-31")
    result = build_portfolio(
        portfolio, investment_horizon=252, target_capital=100_000,
        data=data, x_known=x_known, start_date="2025-01-01", end_date="2025-12-31",
        strategy="given", confidence=0.99,
    )
    print(result["portfolio_var_parametric"], result["portfolio_var_historical"], result["portfolio_var_mc"])
"""

from .assembly import (
    build_asset_data,
    build_portfolio_risk_matrix,
    build_risk_data,
    build_stock_risk,
    build_yield_risk,
)
from .backtest import kupiec_test, print_kupiec_test, time_series_var
from .cache import clear_cache
from .market_data import DEFAULT_FRED_SERIES, get_fred_yield_curve, interpolate_yield_curve
from .optimization import build_portfolio
from .pricing import (
    REQUIRED_FIELDS,
    get_coupon_bond,
    get_european_option,
    get_stock,
    get_zero_bond,
    validate_portfolio,
)
from .var_models import calculate_parametric_var, esg_score_invest, historical_var, monte_carlo_var, sharpratio

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_FRED_SERIES",
    "REQUIRED_FIELDS",
    "build_asset_data",
    "build_portfolio",
    "build_portfolio_risk_matrix",
    "build_risk_data",
    "build_stock_risk",
    "build_yield_risk",
    "calculate_parametric_var",
    "clear_cache",
    "esg_score_invest",
    "get_coupon_bond",
    "get_european_option",
    "get_fred_yield_curve",
    "get_stock",
    "get_zero_bond",
    "historical_var",
    "interpolate_yield_curve",
    "kupiec_test",
    "monte_carlo_var",
    "print_kupiec_test",
    "sharpratio",
    "time_series_var",
    "validate_portfolio",
]
