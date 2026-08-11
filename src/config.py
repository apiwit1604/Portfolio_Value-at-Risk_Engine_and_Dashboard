"""
Config
======
All constants defined here rather than scattered as magic numbers, so a
single parameter change propagates everywhere it's used.

Scaling convention (used consistently throughout): volatility scales with
sqrt(horizon), variance/covariance scales with horizon -- consistent with
the i.i.d. daily returns assumption.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Universe & data window
# ---------------------------------------------------------------------------
TICKERS = ["NVDA", "AAL", "META", "TSLA", "ASML"]
START_DATE = "2015-12-31"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Initial portfolio weights (must be in the same order as TICKERS)
# ---------------------------------------------------------------------------
n = len(TICKERS)
W_PORT_INITIAL = np.ones(n) / n

# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
INVESTMENT_HORIZON = 5      # in trading days, used by the static VaR section
CONFIDENCE = 0.95           # e.g. 0.95 -> 95% VaR

# ---------------------------------------------------------------------------
# Rolling window backtest
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 252        # trading days used to estimate mu/sigma per window
N_ROLLING_WINDOWS = 252     # number of windows to compute (stepped by 1 day)

# ---------------------------------------------------------------------------
# Optimization bounds (used by find_min_risk_portfolio / find_min_loss_portfolio)
# ---------------------------------------------------------------------------
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Monte Carlo simulation sizes
# ---------------------------------------------------------------------------
MC_SIMS_STATIC = 300_000    # static (section 10) Monte Carlo VaR
MC_SIMS_ROLLING = 50_000    # rolling backtest (section 11) Monte Carlo VaR
