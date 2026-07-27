"""
Rolling 1-year re-estimation and Kupiec's Proportion-of-Failures (POF)
backtest.

Rolling window re-estimation recomputes all three VaR models on a rolling
252-trading-day window (~1 trading year), stepped by 1 day at a time, to see
how VaR changes over time (e.g. VaR should worsen during high-volatility
periods).

Index convention: key 0 = most recent window (today, looking back 252 days),
key -i = the window shifted back i days from today.

Kupiec's POF test evaluates how well each of the three VaR models
(Parametric, Historical, Monte Carlo) predicted risk, using a Likelihood
Ratio test statistic based on the binomial distribution.

Statistical background
-----------------------
The test checks whether the number of times the actual loss was worse than
the VaR the model predicted (exceptions, x) is consistent with the stated
confidence level (c).

    H0 (null): the model is accurate -- the observed exception rate (p_hat)
               equals the theoretical rate (p = 1 - c).
    H1 (alt):  the model is inaccurate (it may underestimate or
               overestimate risk).

Likelihood ratio statistic:

    LR_pof = -2 * ln[ ((1-p)^(N-x) * p^x) / ((1-p_hat)^(N-x) * p_hat^x) ]

Decision rule: compare the p-value against the significance level alpha
(NOT against confidence) so the result maps directly onto standard
statistical language without requiring the reader to infer anything:

    p-value >= alpha  (equivalently LR_pof <= critical value)
        -> Fail to reject H0 -- the model passes: the number of exceptions
           is within a statistically acceptable range.
    p-value <  alpha  (equivalently LR_pof >  critical value)
        -> Reject H0 -- the model fails: it may be underestimating risk
           (if x is too high) or overestimating risk (if x is too low).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def build_rolling_windows(returns_daily: pd.DataFrame, rolling_window: int, n_windows: int) -> dict:
    """Build the rolling 1-year window dataset.

    This steps one TRADING DAY at a time, not one year at a time.
        key 0  = most recent window (today, looking back `rolling_window` days)
        key -i = the window shifted back i days from today

    Look-ahead bias prevention: window[-i] = returns_daily.iloc[end-rolling_window-i : end-i]
    covers data only up through "the day before -i" (Python slicing excludes
    the stop index), so the mu/sigma used to forecast VaR for day -i never
    see day -i's actual return -- consistent with correct backtest
    methodology for the window statistics themselves.

    LIMITATION: this addresses look-ahead bias in the per-window mu/sigma
    estimates only. It does NOT address portfolio-weight look-ahead bias --
    see the note on compute_actual_return below.
    """
    end = len(returns_daily)
    returns_daily_1y = {}
    for i in range(n_windows):
        returns_daily_1y[-i] = returns_daily.iloc[end - rolling_window - i : end - i]
    return returns_daily_1y


def compute_actual_return(returns_daily: pd.DataFrame, w_port: pd.Series, n_windows: int) -> dict:
    """Compute the portfolio's realized daily return, walking backward from
    the end of the table, with keys 0 (most recent day), -1, -2, ... up to
    n_windows.

    LIMITATION -- portfolio-weight look-ahead bias: `w_port` is a single
    fixed weight vector applied across the entire backtest. If that weight
    vector was itself derived from an optimization over the FULL sample
    (including data from after each rolling window's cutoff date), then the
    realized returns compared against each window's VaR are not truly
    point-in-time: the weights "know" about the future relative to earlier
    windows in the backtest, even though the mu/sigma inputs to each
    window's VaR do not. This is a portfolio-level look-ahead bias distinct
    from the window-level protection described in build_rolling_windows,
    and should be treated as a caveat on the backtest's validity rather
    than a solved problem.
    """
    n_rows = len(returns_daily)
    actual_return_dict = {}

    for i in range(n_windows):
        row_index = n_rows - 1 - i
        actual_return_dict[-i] = returns_daily.iloc[row_index] @ w_port

    return {"actual_return": actual_return_dict}


def calculate_var_metrics(
    window_data: dict,
    w_port: np.ndarray,
    n_rolling: int,
    confidence: float,
    t: int,
    mc_sims: int = 50_000,
) -> dict:
    """Recompute Parametric / Historical / Monte Carlo VaR for each rolling
    window.
    """
    sigma_port = {}
    mu_port = {}
    sigma_asset = {}
    mu_asset = {}
    var_para = {}
    var_hist = {}
    var_monte = {}

    z_score = norm.ppf(1 - confidence)
    n_assets = len(w_port)

    for i in range(n_rolling):
        cov_matrix = window_data[-i].cov()

        # Per-asset mu/sigma
        mu_asset[-i] = window_data[-i].mean()
        sigma_asset[-i] = window_data[-i].std()

        # Parametric VaR
        sigma_port[-i] = np.sqrt(w_port @ cov_matrix @ w_port)
        mu_port[-i] = mu_asset[-i] @ w_port
        var_para[-i] = mu_port[-i] + z_score * sigma_port[-i]

        # Historical VaR
        portfolio_return = window_data[-i] @ w_port
        var_hist[-i] = portfolio_return.quantile(1 - confidence)

        # Monte Carlo simulation (Cholesky recomputed per window)
        L = np.linalg.cholesky(cov_matrix)
        z_independent = np.random.standard_normal((mc_sims, n_assets))
        z_correlated = z_independent @ L.T

        log_returns = np.zeros((mc_sims, n_assets))
        for k in range(n_assets):
            log_returns[:, k] = (
                (mu_asset[-i].iloc[k] - 0.5 * sigma_asset[-i].iloc[k] ** 2) * t
                + np.sqrt(t) * z_correlated[:, k]
            )

        portfolio_sim_returns = log_returns @ w_port
        var_monte[-i] = np.quantile(portfolio_sim_returns, 1 - confidence)

    return {
        "Parametric VaR": var_para,
        "Historical VaR": var_hist,
        "Monte VaR": var_monte,
    }


def kupiec_test(model_var: dict, actual_return: dict, confidence: float) -> None:
    """Kupiec Proportion-of-Failures (POF) backtest.

    LIMITATION: model_var and actual_return are combined via
    pd.DataFrame({...}), which aligns entries by dict key. This is only
    safe as long as both dicts contain exactly the same set of keys (e.g.
    both cover keys 0..-(N-1) with no gaps). There is no explicit check
    that the key sets match -- if they ever diverge, pandas will silently
    align on the intersection (or introduce NaNs) rather than raising an
    error, which would misalign the exception count without warning.
    """

    result_test = pd.DataFrame({"Model VaR": model_var, "Actual Return": actual_return})

    x = np.sum(result_test["Actual Return"] < result_test["Model VaR"])
    N = len(model_var)
    p = 1 - confidence  # theoretical exception rate (alpha)
    p_hat = x / N  # observed exception rate

    lr_pof = -2 * np.log((((1 - p) ** (N - x)) * (p**x)) / (((1 - p_hat) ** (N - x)) * (p_hat**x)))
    alpha = p
    crit = chi2.ppf(1 - alpha, df=1)
    p_value = 1 - chi2.cdf(lr_pof, df=1)

    print(f"LR: {lr_pof:.4f}")
    print(f"Critical Value ({alpha * 100:.2f}%): {crit:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value > alpha:
        print("result test: Fail to reject H0 (Pass)")
    else:
        print("result test: reject H0 (Fail)")
