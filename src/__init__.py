"""
var_risk_engine.src
====================
Portfolio Value at Risk (VaR) engine -- Parametric, Historical, and Monte
Carlo methods, portfolio optimization, and rolling-window backtesting with
the Kupiec POF test.

Modules:
  config          -- all tunable constants
  data            -- price fetching and return computation
  portfolio_stats -- horizon scaling for mu/covariance
  var_models      -- performance_portfolio, historical_var, monte_carlo_var
  optimize        -- find_min_risk_portfolio, find_min_loss_portfolio
  backtest        -- rolling window construction, calculate_var_metrics, kupiec_test
  report          -- print/display helpers
"""
