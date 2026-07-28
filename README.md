# Portfolio VaR Risk Engine

A single-notebook implementation of three Value-at-Risk (VaR) methodologies — **Parametric (Variance-Covariance)**, **Historical**, and **Monte Carlo** — applied to a 5-stock equity portfolio, with constrained mean-variance optimization and a rolling 252-day out-of-sample backtest validated against the **Kupiec Proportion-of-Failures (POF) test**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tor-apiwit/var_risk_engine/blob/main/var_risk_engine.ipynb)

---

## What this project does

Given a portfolio of `NVDA, AAL, META, TSLA, ASML` (2015–2025 daily data via `yfinance`), the notebook:

1. Computes daily log returns and scales mean/covariance to a target horizon consistently (`mu * h`, `cov * h`, under the i.i.d. daily-returns assumption).
2. Estimates VaR three independent ways:
   - **Parametric** — closed-form `return + z * risk` under a normality assumption.
   - **Historical** — empirical quantile of actual realized portfolio returns, distribution-free.
   - **Monte Carlo** — 300,000 correlated GBM simulations via Cholesky decomposition of the correlation matrix.
3. Solves two constrained optimization problems (`scipy.optimize.minimize`, SLSQP) to find portfolio weights that:
   - minimize portfolio variance (**Global Minimum-Variance Portfolio**), or
   - minimize tail loss, i.e. maximize parametric VaR (**Minimum-Loss Portfolio**).
4. Re-runs all three VaR methods on a rolling 252-day window (252 steps, one trading day at a time) to see how each model's risk estimate evolves — and whether estimates widen during high-volatility regimes.
5. Backtests each model's rolling VaR against realized returns using the **Kupiec POF test** (likelihood-ratio statistic vs. chi-square critical value), to check whether each model's exception rate is statistically consistent with its stated confidence level.

## Why three VaR methods, not one

Each method fails differently, and the notebook is explicit about that instead of picking one and hiding the tradeoff:

| Method | Assumes | Breaks when |
|---|---|---|
| Parametric | Returns are normally distributed | Fat tails / skew — equities are famously not normal |
| Historical | The future resembles the sampled past | Regime shifts; no history of the "next" crash |
| Monte Carlo | Correlation structure is stable, GBM is a valid return generator | Correlations spike in a crisis (the moment VaR matters most) |

Running all three and backtesting them against each other is the point — a single VaR number in isolation is not a risk estimate you should trust.

## Key design decisions

- **Consistent horizon scaling.** `scale_mu` / `scale_cov` are the single place mean and covariance are scaled from daily to horizon (`mu * horizon`, `cov * horizon`) — enforced everywhere instead of `* investment_horizon` scattered across cells.
- **VaR sign convention.** VaR is stored as a negative number throughout (more negative = worse loss). `find_min_loss_portfolio` (deliberately renamed from an earlier, misleading `find_min_var_portfolio`) maximizes VaR — i.e. pushes it toward zero — which minimizes the magnitude of tail loss. This is *not* the same as "minimizing VaR," which would drive the estimate toward `-∞`.
- **No look-ahead bias in the rolling backtest.** Window `-i` is built as `returns_daily.iloc[end - window - i : end - i]` — Python's exclusive slice stop means the statistics used to forecast day `-i`'s VaR never see day `-i`'s actual return.
- **Portfolio-level quantiles, not asset-level.** Both Historical and Monte Carlo VaR take the quantile of the *combined weighted portfolio return series*, not a weighted sum of each asset's individual quantile — VaR is not linear across marginals, so the two are not interchangeable.

## Repo structure

```
var_risk_engine.ipynb   # Full pipeline: data → VaR (x3) → optimization → rolling backtest → Kupiec test
```

This is intentionally a single notebook — sections are numbered and documented inline (config → data/returns → portfolio stats → parametric VaR → optimization → historical VaR → Monte Carlo VaR → rolling re-estimation → Kupiec backtest) so the whole pipeline can be read top to bottom in one sitting.

## Running it

Open in Colab (badge above) or locally:

```bash
pip install numpy pandas matplotlib yfinance scipy
jupyter notebook var_risk_engine.ipynb
```

All tunable parameters (tickers, date range, horizon, confidence level, rolling window length) live in the **Config** section near the top of the notebook — change them there rather than in the function bodies.

## Known limitation

The rolling-window Monte Carlo re-estimation (`calculate_var_metrics`) hardcodes a 1-day horizon (`t=1`) internally. The drift/vol scaling by `t` is only correct at `t=1`; calling it with a different horizon would require re-deriving the per-window drift term. Flagged here rather than silently — this is a "know its limits" note, not a bug that affects the current results.

## Next in this portfolio

Part of a broader quant-finance project series: options pricer, ALM toolkit, credit scorecard, and a Streamlit dashboard tying them together.
