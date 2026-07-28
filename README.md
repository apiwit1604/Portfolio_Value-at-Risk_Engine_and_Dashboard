# Portfolio VaR Engine

Computes portfolio Value at Risk (VaR) using three independent methods —
**Parametric (Variance-Covariance)**, **Historical**, and **Monte Carlo** —
finds portfolio weights that minimize risk or minimize tail loss via
constrained optimization, and validates each VaR model with a rolling
252-day **Kupiec Proportion-of-Failures (POF) backtest**.

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
optimization objectives are framed.
