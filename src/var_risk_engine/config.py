"""
Configuration for the Portfolio VaR Engine.

All constants live here instead of being scattered as "magic numbers"
throughout the codebase, so a single parameter change propagates everywhere
it is used.

Scaling convention (used consistently across the whole project):
    - volatility scales with sqrt(horizon)
    - variance / covariance scales with horizon
This follows directly from the i.i.d. daily returns assumption.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Universe & data window
# ---------------------------------------------------------------------------
TICKERS = ["NVDA", "AAL", "META", "TSLA", "ASML"]
START_DATE = "2015-12-31"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
# Initial portfolio weights (must be in the same order as TICKERS)
W_PORT_INITIAL = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])

# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
INVESTMENT_HORIZON = 5  # in trading days, used by the static VaR section
CONFIDENCE = 0.95  # e.g. 0.95 -> 95% VaR

# ---------------------------------------------------------------------------
# Rolling window backtest
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 252  # trading days used to estimate mu/sigma per window
N_ROLLING_WINDOWS = 252  # number of windows to compute (stepped by 1 day)

# ---------------------------------------------------------------------------
# Monte Carlo simulation counts
# ---------------------------------------------------------------------------
MC_SIMS_STATIC = 300_000  # simulations for the static (single-horizon) section
MC_SIMS_ROLLING = 50_000  # simulations per window in the rolling backtest
