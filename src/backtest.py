"""
Rolling 1-Year Re-estimation & Backtest
========================================
Recomputes all three VaR methods on a rolling 252-day window (~1 trading
year), stepped by 1 day at a time, to see how VaR changes over time (e.g.
VaR should worsen during periods of high market volatility).

Index convention: key 0 = the most recent window (today, looking back 252
days), key -i = the window shifted back i days from today.

*** KNOWN ISSUE -- NOT FIXED HERE, PORTED AS-IS FROM THE ORIGINAL SCRIPT ***
build_rolling_windows() constructs window[-i] as
    returns_daily.iloc[end - ROLLING_WINDOW - i : end - i]
whose LAST included row has index (end - i - 1).

compute_actual_return() computes the "actual" return at key -i from row
    n_rows - 1 - i  ==  end - i - 1
i.e. the SAME row that is the last row of window[-i].

That means the mu/sigma used to forecast VaR for key -i are estimated
using a sample that INCLUDES the very day whose realized return is being
compared against it. This is look-ahead bias: the model has "seen" the
outcome it's being scored on for that one day. It very likely inflates
apparent model accuracy in the Kupiec test.

Fix (not yet applied): either compute the window as
    returns_daily.iloc[end - ROLLING_WINDOW - i : end - i - 1]
(exclude the target day itself), or compute actual_return at row
    n_rows - i  (one day after the window's last row)
so the window and the actual-return day never overlap. Left unchanged
here at the user's request -- flagging clearly rather than silently
"fixing" behavior that may be intentional or still being iterated on.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def build_rolling_windows(
    returns_daily: pd.DataFrame, rolling_window: int, n_windows: int
) -> dict[int, pd.DataFrame]:
    """
    Build the rolling 1-year window dataset.

    This steps one TRADING DAY at a time, not one year at a time.
      key 0  = most recent window (today, looking back `rolling_window` days)
      key -i = the window shifted back i days from today

    See module docstring for a known look-ahead-bias caveat in how this
    aligns with compute_actual_return().
    """
    end = len(returns_daily)
    windows = {}
    for i in range(n_windows):
        windows[-i] = returns_daily.iloc[end - rolling_window - i : end - i]
    return windows


def compute_actual_return(returns_daily: pd.DataFrame, w_port: pd.Series, n_windows: int) -> dict:
    """
    Compute the portfolio's realized daily return, walking backward from
    the end of the table, with keys 0 (most recent day), -1, -2, ... up to
    n_windows.
    """
    n_rows = len(returns_daily)
    actual_return_dict = {}

    for i in range(n_windows):
        row_index = n_rows - 1 - i
        actual_return_dict[-i] = returns_daily.iloc[row_index] @ w_port

    return {"actual_return": actual_return_dict}


def calculate_var_metrics(
    window_data: dict[int, pd.DataFrame],
    w_port: np.ndarray,
    n_rolling: int,
    confidence: float,
    t: int,
    mc_sims: int = 50_000,
) -> dict:
    """
    Recompute Parametric / Historical / Monte Carlo VaR for each rolling
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

        # Mu/Sigma asset
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
        Z_independent = np.random.standard_normal((mc_sims, n_assets))
        Z_correlated = Z_independent @ L.T

        log_returns = np.zeros((mc_sims, n_assets))
        for k in range(n_assets):
            log_returns[:, k] = (
                (mu_asset[-i].iloc[k] - 0.5 * sigma_asset[-i].iloc[k] ** 2) * t
                + np.sqrt(t) * Z_correlated[:, k]
            )

        portfolio_sim_returns = log_returns @ w_port
        var_monte[-i] = np.quantile(portfolio_sim_returns, 1 - confidence)

    return {
        "Parametric VaR": var_para,
        "Historical VaR": var_hist,
        "Monte VaR": var_monte,
    }


def kupiec_test(model_var: dict, actual_return: dict, confidence: float) -> dict:
    """
    Kupiec Proportion-of-Failures (POF) backtest.

    Tests whether the number of times the actual loss was worse than the
    model's predicted VaR (exceptions, x) is consistent, in a statistically
    significant sense, with the stated confidence level (c).

    H0: the model is accurate -- observed exception rate (p_hat) equals
        the theoretical rate (p = 1 - c)
    H1: the model is inaccurate (may underestimate or overestimate risk)

    LR_pof = -2 * ln[ ((1-p)^(N-x) * p^x) / ((1-p_hat)^(N-x) * p_hat^x) ]

    Decision rule compares the p-value against the significance level
    alpha (not against confidence), so the result maps directly onto
    standard statistical language:
      p-value >= alpha  ->  fail to reject H0  ->  model passes
      p-value <  alpha  ->  reject H0          ->  model fails

    Returns a dict of the computed statistics (in addition to printing a
    human-readable summary) so callers can build comparison tables.
    """
    result_test = pd.DataFrame({"Model VaR": model_var, "Actual Return": actual_return})

    x = np.sum(result_test["Actual Return"] < result_test["Model VaR"])
    N = len(model_var)
    p = 1 - confidence          # theoretical exception rate (alpha)
    p_hat = x / N               # observed exception rate

    lr_pof = -2 * np.log((((1 - p) ** (N - x)) * (p**x)) / (((1 - p_hat) ** (N - x)) * (p_hat**x)))
    alpha = p
    crit = chi2.ppf(1 - alpha, df=1)
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    passed = p_value > alpha

    print(f"LR: {lr_pof:.4f}")
    print(f"Critical Value ({alpha * 100:.2f}%): {crit:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("result test: Fail to reject H0 (Pass)" if passed else "result test: reject H0 (Fail)")

    return {
        "x": x,
        "N": N,
        "p_hat": p_hat,
        "LR_pof": lr_pof,
        "critical_value": crit,
        "p_value": p_value,
        "passed": passed,
    }
