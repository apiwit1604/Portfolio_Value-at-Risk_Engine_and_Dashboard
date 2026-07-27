# Portfolio VaR Engine

Computes portfolio Value at Risk (VaR) using three independent methods —
**Parametric (Variance-Covariance)**, **Historical**, and **Monte Carlo** —
finds portfolio weights that minimize risk or minimize tail loss via
constrained optimization, and validates each VaR model with a rolling
252-day **Kupiec Proportion-of-Failures (POF) backtest**.

## Project layout

```
var_risk_engine/
├── pyproject.toml            # package metadata, dependencies, pytest/ruff config
├── README.md
├── src/
│   └── var_risk_engine/      # the installable package
│       ├── __init__.py       # public API re-exports
│       ├── config.py         # all constants (tickers, dates, weights, horizons, sim counts)
│       ├── data_loader.py    # price fetching (yfinance) + return computation
│       ├── portfolio_stats.py# horizon scaling (scale_mu, scale_cov)
│       ├── var_models.py     # Parametric / Historical / Monte Carlo VaR
│       ├── optimization.py   # min-risk and min-loss (max-VaR) portfolio search
│       ├── backtest.py       # rolling windows, realized returns, Kupiec POF test
│       └── reporting.py      # print/plot display helpers
├── notebooks/
│   └── var_risk_engine.ipynb # orchestration notebook — imports the package, no embedded logic
└── tests/
    ├── conftest.py            # shared synthetic-data fixtures (seeded, offline)
    ├── test_data_loader.py
    ├── test_var_models.py
    ├── test_optimization.py
    ├── test_backtest.py
    └── test_reporting.py
```

**Why `src/var_risk_engine/` and not just `var_risk_engine/` at the repo root?**
This is the standard ["src layout"](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
It forces the package to be *installed* (`pip install -e .`) rather than
accidentally importable via an ambient `sys.path` entry pointing at the repo
root. Without the `src/` indirection, running tests or scripts from the repo
root can silently import the working-tree copy even when the installed
version is stale or broken — the `src/` layout makes that failure mode
structurally impossible.

## Installation

```bash
git clone <repo-url>
cd var_risk_engine
pip install -e ".[dev]"
```

This installs the package in editable mode plus development dependencies
(`pytest`, `pytest-cov`, `jupyter`, `nbconvert`, `ruff`).

## Usage

```python
from var_risk_engine import config
from var_risk_engine.data_loader import load_close_prices, compute_returns
from var_risk_engine.var_models import performance_portfolio

close_prices = load_close_prices(config.TICKERS, config.START_DATE, config.END_DATE)
returns_daily, returns_annual = compute_returns(close_prices)
```

Or open `notebooks/var_risk_engine.ipynb` for the full walkthrough (data
loading → VaR models → optimization → rolling backtest → Kupiec validation).

## Running tests

```bash
pytest
```

Runs the full suite (43 tests) with coverage reporting (currently 100% line
coverage across all modules). Tests use seeded synthetic data — no network
access is required or used.

## Known limitations

These are documented in code as `# LIMITATION:` comments at the relevant
function, not silently fixed, because fixing them changes the numerical
output of the backtest:

1. **`kupiec_test` key-alignment fragility** (`backtest.py`): the function
   combines two dicts via `pd.DataFrame({...})`, which aligns entries by key.
   This is only safe if both dicts contain exactly the same key set — if
   they ever diverge, pandas introduces `NaN`s silently rather than raising.
2. **Portfolio-weight look-ahead bias** (`backtest.py`, `compute_actual_return`):
   the backtest applies a single fixed weight vector across the entire
   rolling-window test. If that vector was derived from optimizing over the
   full sample (as the notebook does), the "realized" returns used to score
   each window's VaR are not strictly point-in-time, even though the
   mu/sigma estimates that feed each window's VaR are.

## Sign convention

VaR is stored as a (typically negative) **return**, not a positive loss
magnitude, throughout this codebase. A more negative VaR means a larger
expected loss at the given confidence level. This affects how the
optimization objectives are framed — see the docstring in `optimization.py`
for the historical naming issue this caused (`find_min_var_portfolio` →
renamed to `find_min_loss_portfolio`).
