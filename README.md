# Portfolio Value-at-Risk Engine & Dashboard


[[App Screenshot](./images/dashboard_images.png)](https://portfolio-value-at-risk-engine-and-dashboard.streamlit.app/)

Dashboard: [Enjoy Now!](https://portfolio-value-at-risk-engine-and-dashboard.streamlit.app/)

A multi-asset **Value-at-Risk (VaR)** engine — bonds, stocks, FX, and European
options/forwards — with three VaR methodologies (Parametric, Historical,
Monte Carlo), four portfolio-optimization strategies, and Kupiec
Proportion-of-Failures backtesting. Ships as both an importable Python
package (`var_engine/`) and an interactive **Streamlit dashboard**
(`app.py`) where you can change assets, tickers, dates, and the investment
horizon without touching code.

Built on live market data: **[Yahoo Finance](https://finance.yahoo.com)**
(via `yfinance`) for stock/FX/option-underlying prices, and
**[FRED](https://fred.stlouisfed.org)** for the U.S. Treasury yield curve.

> Originally prototyped in a Colab notebook, then refactored into this
> package + dashboard. See [Known limitations](#known-limitations--modeling-assumptions)
> below for an honest list of what this does and doesn't do well —
> read that before using any number here for a real decision.

---

## Contents

- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Using the dashboard](#using-the-dashboard)
- [Asset type reference](#asset-type-reference) — how to configure every instrument type
- [VaR methodology](#var-methodology)
- [Optimization strategies](#optimization-strategies)
- [Backtesting](#backtesting--kupiec-proportion-of-failures-test)
- [Using the engine as a library](#using-the-engine-as-a-library)
- [Performance notes & caching](#performance-notes--caching)
- [Known limitations & modeling assumptions](#known-limitations--modeling-assumptions)
- [Deploying the dashboard](#deploying-the-dashboard)

---

## Quick start

```bash
git clone https://github.com/<your-username>/var-portfolio-engine.git
cd var-portfolio-engine
pip install -r requirements.txt

streamlit run app.py
```

Opens at `http://localhost:8501`. The dashboard loads with a working example
portfolio (70% NVDA stock / 20% NVDA put option / 10% THB FX) so you see a
real result immediately — edit the table to build your own.

No dashboard needed? Run the console version instead:

```bash
python examples/run_cli_example.py
```

---

## Repository structure

```
var-portfolio-engine/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── .streamlit/config.toml     # dashboard theme
├── var_engine/                 # the importable engine — no Streamlit dependency here
│   ├── __init__.py             # public API (see docstring for a code example)
│   ├── cache.py                 # caches Yahoo Finance / FRED fetches (see Performance notes)
│   ├── market_data.py            # FRED yield curve download + cubic-spline interpolation
│   ├── pricing.py                 # per-instrument pricing (ZCB, CB, STK, FX, options/forwards)
│   ├── assembly.py                 # combines priced assets into one risk matrix
│   ├── var_models.py                # Parametric / Historical / Monte Carlo VaR, ESG, Sharpe
│   ├── optimization.py               # build_portfolio(): the 4 weighting strategies
│   ├── backtest.py                    # rolling VaR backtest + Kupiec test
│   ├── display.py                      # console pretty-printer + breach plot (CLI use only)
│   └── ui_helpers.py                    # dashboard table <-> portfolio-dict conversion (unit-testable, no Streamlit import)
└── examples/
    └── run_cli_example.py       # console walkthrough: all 5 strategies + backtests, no dashboard
```

The original single 1,000-line notebook export is split so that (a) the
math has no Streamlit dependency and can be unit-tested or reused in a
notebook, and (b) `app.py` stays UI-only glue code.

---

## Using the dashboard

### 1. Build the portfolio

Edit the table directly — add rows with the **+** button, delete with the
row checkbox + delete key. Every row needs `name`, `type`, `weight`, and
`esg_score`; the remaining columns depend on `type` (see the reference
table below — irrelevant columns for a given type can be left blank).

**Ticker column accepts any valid [Yahoo Finance](https://finance.yahoo.com) symbol** —
not just the ones in the example:

| Market | Example tickers |
|---|---|
| US stocks | `AAPL`, `NVDA`, `MSFT`, `TSLA` |
| Thai SET stocks | `PTT.BK`, `AOT.BK`, `SCB.BK`, `KBANK.BK` |
| FX spot | `THB=X`, `EURUSD=X`, `JPY=X` |
| Crypto | `BTC-USD`, `ETH-USD` |
| Indices | `^GSPC` (S&P 500), `^SET.BK` |

Weights must sum to **1.0000** — the app shows a live running total and a
**Normalize weights** button that rescales whatever you've entered.

### 2. Set dates, horizon, and confidence (sidebar)

- **Historical estimation window** (start/end date) — the price history
  used to estimate current prices, expected returns, and the
  covariance matrix. This is "**วันที่ในการลงทุน**" in the sense of "what
  period's data defines the risk model" — it is *not* a future date;
  VaR is always estimated from historical data as of `end_date`.
- **Investment horizon** — how many trading days ahead the VaR number
  covers (this is "**ระยะเวลาการลงทุน**" — the holding period). Presets from
  1 day to 1 year, or type a custom number of trading days.
- **Confidence level** — 90–99.5%, or custom.

### 3. Pick a strategy and run

Choose one of the five weighting strategies (see
[Optimization strategies](#optimization-strategies)), or check
**Compare all 5 strategies** to run all of them and get a side-by-side
table. The compare-all option is meaningfully slower — see
[Performance notes](#performance-notes--caching) for why.

### 4. Backtest (optional, on demand)

A separate **Run backtest** button in section 3 of the dashboard rolls the
estimation window forward day-by-day over an out-of-sample period and runs
the Kupiec test. This is the most expensive operation in the app (it
re-computes Monte Carlo VaR for every out-of-sample day), so it's kept
behind its own button rather than running automatically.

---

## Asset type reference

Every position in the portfolio table is one instrument. `name`, `type`,
`weight`, and `esg_score` are always required; the table below is the
complete field reference for the rest.

| `type` | Meaning | Extra required fields | Notes |
|---|---|---|---|
| `ZCB` | Zero-coupon bond | `face_value`, `years` | Priced off the FRED yield curve interpolated at `years`. |
| `CB` | Coupon bond | `face_value`, `coupon_rate`, `freq`, `years` | `coupon_rate` is annual (e.g. `0.03` = 3%); `freq` = payments/year (`2` = semiannual). Each coupon date gets its own point on the yield curve. |
| `STK` | Stock | `ticker` | Any Yahoo Finance equity ticker. Priced at the last close in the estimation window. |
| `FX` | FX spot | `ticker` | Any Yahoo Finance FX ticker, e.g. `THB=X` (USD/THB), `EURUSD=X`. |
| `ECO` | European call option | `ticker`, `K`, `T` | `K` = strike price, `T` = years to expiry. Priced with Black-Scholes using historical volatility from the estimation window. |
| `EPO` | European put option | `ticker`, `K`, `T` | Same as `ECO`, put payoff. |
| `FC` | Forward contract | `ticker`, `K`, `T` | `K` = agreed forward price, synthesized via put-call parity (long call + short put at the same `K`/`T`). |

**`weight`** is this position's target share of "Target capital" (sidebar) —
all weights in the table must sum to 1.0. **`esg_score`** (0–100) is a
weighted-average score reported in the results and can be used as a hard
floor for the "Max Sharpe + ESG floor" strategy.

### Example rows

Zero-coupon bond, 5-year, $1,000 face value, 15% of a $100,000 portfolio:

| name | type | ticker | weight | esg_score | face_value | years |
|---|---|---|---|---|---|---|
| B1 | ZCB | | 0.15 | 60 | 1000 | 5 |

3% semiannual coupon bond, 10-year, $1,000 face value:

| name | type | ticker | weight | esg_score | face_value | coupon_rate | freq | years |
|---|---|---|---|---|---|---|---|---|
| B2 | CB | | 0.20 | 65 | 1000 | 0.03 | 2 | 10 |

NVDA put option, strike $250, expiring in 1 year:

| name | type | ticker | weight | esg_score | K | T |
|---|---|---|---|---|---|---|
| E1 | EPO | NVDA | 0.20 | 80 | 250 | 1 |

### How risk is measured per instrument

Each instrument maps its dollar sensitivity onto one or more **risk
buckets** — either a point on the yield curve (in years) or a ticker —
which is what gets combined across the whole portfolio into the
covariance-based risk model:

- **Bonds (`ZCB`/`CB`)**: sensitivity to the yield at each cash-flow date,
  approximated as `-tenor × position_value` per bucket (a
  duration-style linear approximation — see limitations below).
- **Stocks/FX (`STK`/`FX`)**: sensitivity ≈ 1 to that ticker's own daily
  log return (i.e. `d(Value) ≈ position_value × d(log price)`).
- **Options/forwards (`ECO`/`EPO`/`FC`)**: split into two buckets —
  delta-equivalent sensitivity to the underlying's log return, and
  rho-equivalent sensitivity to the `T`-year point on the yield curve.

---

## VaR methodology

All three methods estimate the same quantity — the loss level not expected
to be exceeded over the investment horizon, at the chosen confidence — from
different assumptions:

- **Parametric (delta-normal)**: assumes risk-factor changes are jointly
  normal. Fast, closed-form, but understates tail risk if returns are
  fat-tailed (they usually are).
- **Historical**: applies today's position sensitivities to the *actual*
  historical risk-factor changes in the estimation window, then takes the
  empirical quantile. No distributional assumption, but only as good as
  the history you give it — a calm estimation window will understate risk
  for a volatile future period, and vice versa.
- **Monte Carlo**: fits a multivariate normal to the same historical
  risk factors, simulates many correlated draws, and takes the empirical
  quantile of simulated portfolio returns. Same normality assumption as
  Parametric underneath, but captures option convexity/non-linearity that
  the Parametric method's linear sensitivities miss.

Because all three read the same underlying risk factors, expect them to
mostly agree for a stock/FX-heavy portfolio and diverge more once options
are added (Parametric's linear approximation is weakest there).

---

## Optimization strategies

| Strategy | Objective |
|---|---|
| Given weights | No optimization — uses the weights exactly as entered. |
| Minimum risk | Minimize portfolio volatility (`portfolio_risk`). |
| Minimum VaR | Minimize the size of the Parametric VaR loss. |
| Maximum Sharpe | Maximize `(return - risk_free) / volatility`. Risk-free rate is read from the FRED 3-month yield (`0.25`-year point) at the end of the estimation window. |
| Max Sharpe + ESG floor | Same as above, plus a constraint that the weighted ESG score ≥ your chosen floor. |

All four optimized strategies use `scipy.optimize.minimize` (SLSQP) with
weights bounded by the sidebar's min/max-weight sliders and constrained to
sum to 1. If the optimizer doesn't converge (e.g. an infeasible
min/max-weight + ESG-floor combination), the dashboard surfaces the
optimizer's own message rather than silently returning a bad result.

---

## Backtesting — Kupiec Proportion-of-Failures test

For each out-of-sample day, the engine rolls a fixed-size estimation window
forward, recomputes VaR with that day's data, and checks whether the
*realized* return breached VaR. Over N days at (1 − confidence) VaR, you'd
expect roughly `N × (1 − confidence)` breaches; the Kupiec test is a
likelihood-ratio test of whether the *observed* breach count is
statistically consistent with that expectation — "Pass" means the model's
breach rate isn't distinguishable from the theoretical rate; "Fail" means
it is (too many or too few breaches).

---

## Using the engine as a library

```python
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
```

See `examples/run_cli_example.py` for the full walkthrough (all 5
strategies + backtests) as a script.

---

## Performance notes & caching

Worth knowing before you run **"Compare all 5 strategies"** or a long
backtest:

- Optimized strategies (`min_risk`/`min_var`/`max_sharpe`) call
  `scipy.optimize.minimize`, which evaluates the objective — a full
  re-pricing of every asset — many times (SLSQP: typically 10–50+
  evaluations for a handful of assets). **Without caching, that means
  10–50+ live Yahoo Finance/FRED calls per optimization run**, which is
  both slow and prone to Yahoo Finance rate-limiting.
- `var_engine/cache.py` fixes this by wrapping the raw data fetches in
  `functools.lru_cache`, keyed on `(ticker, start, end)` — so every
  distinct fetch happens at most once per process, regardless of how many
  optimizer iterations or dashboard re-runs ask for it. This changes
  nothing about the math (same inputs, same outputs), only how often the
  network gets hit.
- Backtesting is the slowest operation by design: it recomputes Monte
  Carlo VaR for every out-of-sample day. The dashboard defaults to fewer
  simulations for the backtest (5,000/day) than the main run (50,000) and
  shows a progress bar — increase either if you want more precision and
  can wait longer.
- The sidebar's **"Refresh cached market data"** button clears the cache
  if you need fresh prices without restarting the app.

---

## Known limitations & modeling assumptions

Being upfront about what this tool doesn't do, so results aren't
over-trusted:

- **Bond risk uses a linear (duration-style) approximation**
  (`-tenor × position_value`), not full repricing under a shifted curve —
  it will understate risk for large yield moves or long-dated bonds.
- **Options are priced with Black-Scholes using a single historical
  volatility estimate** from the estimation window, with no dividend
  yield and European exercise assumed. Real listed options usually trade
  at a different (and skewed) implied volatility — treat these numbers as
  illustrative, not tradeable.
- **The risk-free rate for Sharpe ratio is hardcoded to the FRED 3-month
  tenor** (the `0.25`-year key in the yield-curve dict). If you were to
  customize the FRED tenor set in `market_data.py` and drop the 3-month
  point, this would break — the dashboard doesn't expose yield-curve tenor
  customization in the UI for this reason.
- **Historical and Monte Carlo VaR are both fully dependent on the
  estimation window you choose** — a calm window will understate risk for
  a volatile future, and there's no automated check for regime changes.
- **Monte Carlo VaR is not seeded by default** in the underlying function
  (pass `random_state=` for reproducibility, which the dashboard and CLI
  example both do) — without it, re-running gives slightly different
  numbers each time, which is expected simulation noise, not a bug.
- This is an educational/portfolio-demonstration tool, not a production
  risk system — it has no position limits, no intraday risk, no
  multi-currency base-currency conversion, and assumes European exercise
  throughout.

---

## Deploying the dashboard

**[Streamlit Community Cloud](https://share.streamlit.io)** (free): push
this repo to GitHub, connect it on share.streamlit.io, point it at `app.py`,
and it builds from `requirements.txt` automatically. No secrets are
required — both Yahoo Finance and FRED are called with public,
unauthenticated endpoints.

---

## License

MIT — see [LICENSE](LICENSE).
