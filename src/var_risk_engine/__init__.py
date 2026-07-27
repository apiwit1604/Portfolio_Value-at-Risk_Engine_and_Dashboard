"""
var_risk_engine -- Portfolio Value at Risk (VaR) engine.

Computes portfolio VaR via three methods (Parametric, Historical, Monte
Carlo), finds risk/loss-minimizing portfolio weights via constrained
optimization, and runs a rolling-window Kupiec POF backtest.

Public API re-exported here so callers can do:

    from var_risk_engine import performance_portfolio, historical_var, ...

instead of reaching into individual submodules.
"""

from . import config
from .backtest import (
    build_rolling_windows,
    calculate_var_metrics,
    compute_actual_return,
    kupiec_test,
)
from .data_loader import compute_returns, load_close_prices
from .optimization import find_min_loss_portfolio, find_min_risk_portfolio
from .portfolio_stats import scale_cov, scale_mu
from .reporting import plot_rolling_var, show_portfolio, show_var_comparison
from .var_models import historical_var, monte_carlo_var, performance_portfolio

__all__ = [  # noqa: RUF022 -- intentionally ordered by pipeline stage, not alphabetically
    "config",
    "load_close_prices",
    "compute_returns",
    "scale_mu",
    "scale_cov",
    "performance_portfolio",
    "historical_var",
    "monte_carlo_var",
    "find_min_risk_portfolio",
    "find_min_loss_portfolio",
    "build_rolling_windows",
    "compute_actual_return",
    "calculate_var_metrics",
    "kupiec_test",
    "show_portfolio",
    "show_var_comparison",
    "plot_rolling_var",
]

__version__ = "0.1.0"
