# Portfolio Value at Risk (VaR) Engine

*[อ่านภาษาไทย](README.th.md)*

Computes portfolio VaR using three methods -- **Parametric
(Variance-Covariance)**, **Historical**, and **Monte Carlo** -- finds the
portfolio weights that minimize risk or minimize tail loss via constrained
optimization, and runs a rolling 1-year backtest (Kupiec POF test) to
evaluate how each model performs over time.

## Structure

```
var_risk_engine/
├── src/
│   ├── config.py            # all tunable constants (tickers, dates, weights, horizon, etc.)
│   ├── data.py               # load_close_prices, compute_returns
│   ├── portfolio_stats.py    # scale_mu, scale_cov (horizon scaling)
│   ├── var_models.py         # performance_portfolio, historical_var, monte_carlo_var
│   ├── optimize.py           # find_min_risk_portfolio, find_min_loss_portfolio
│   ├── backtest.py           # rolling windows, calculate_var_metrics, kupiec_test
│   └── report.py             # show_portfolio, show_var_comparison
├── notebooks/
│   └── var_risk_engine.ipynb # narrative notebook that imports from src/
├── requirements.txt
├── README.md                 # this file (English)
└── README.th.md              # Thai version
```

Calculation logic lives in `src/`; the notebook is the narrative/display
layer. This split means the logic can be unit-tested or reused in a script
independently of the write-up, and the notebook stays readable instead of
mixing implementation details with explanation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/var_risk_engine.ipynb
```

## Known limitation -- read before trusting the backtest numbers

`src/backtest.py` has a **documented look-ahead bias**: the rolling window
used to *estimate* VaR for a given day currently includes that same day's
return, instead of stopping the day before it. This means the model has
partially "seen" the outcome it's being scored against in the Kupiec test,
which likely makes all three models look more accurate than they actually
are.

This has **not been fixed** in the current version — see the docstring at
the top of `src/backtest.py` for the exact indexing mechanism and the
one-line fix that resolves it. Treat Section 11-12 results in the notebook
as illustrative of the workflow, not as validated model performance, until
this is addressed.

A second, lower-severity caveat: with `N_ROLLING_WINDOWS = 252` and 95%
confidence, the expected number of VaR exceptions is only ~12-13. The
Kupiec test has limited statistical power at this sample size — a "Pass"
result is weak evidence of model adequacy, not proof.

## Method summary

| Method | Assumption | Approach |
|---|---|---|
| Parametric | Returns are normally distributed | `return + z * risk`, z from the normal quantile |
| Historical | No distributional assumption | Empirical quantile of realized portfolio returns |
| Monte Carlo | GBM with correlated shocks | Cholesky-correlated simulation, quantile of simulated portfolio |

VaR is stored as a **negative** number throughout (more negative = worse
loss) — this affects how `find_min_loss_portfolio` is read: it *maximizes*
VaR (pushes it toward 0), which is the same as minimizing the magnitude of
the tail loss.
