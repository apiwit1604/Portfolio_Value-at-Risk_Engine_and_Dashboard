# -*- coding: utf-8 -*-
"""
Asset pricing — one function per instrument type, plus portfolio-config
validation.

Every pricing function returns the same shape:
    {"units": float, "value": float, "pricing": float,
     "result": DataFrame, "risk_matrix": DataFrame}

`risk_matrix` maps the position's dollar sensitivity ("adj_cf") onto one
or more risk-factor buckets (a tenor in years, or a ticker) — that's what
`assembly.build_portfolio_risk_matrix()` later sums across assets.

Asset types
-----------
    type   meaning                 extra required keys
    -----  ----------------------  --------------------------------
    ZCB    Zero-coupon bond        face_value, years
    CB     Coupon bond             face_value, coupon_rate, freq, years
    STK    Stock                   stock_name  (Yahoo ticker, e.g. "AAPL")
    FX     FX spot                 fx_name     (Yahoo FX ticker, e.g. "THB=X")
    ECO    European call option    stock_name, K (strike), T (years to expiry)
    EPO    European put option     stock_name, K, T
    FC     Forward contract        stock_name, K (forward price), T

`weight`, `name`, `type`, and `esg_score` are required on every position;
`weight` is the position's target share of `target_capital` (weights across
the whole portfolio list must sum to 1.0 — validate_portfolio checks this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .cache import fetch_stock_history
from .market_data import interpolate_yield_curve

REQUIRED_FIELDS = {
    "ZCB": {"face_value", "years"},
    "CB": {"face_value", "coupon_rate", "freq", "years"},
    "STK": {"stock_name"},
    "FX": {"fx_name"},
    "ECO": {"stock_name", "K", "T"},
    "EPO": {"stock_name", "K", "T"},
    "FC": {"stock_name", "K", "T"},
}


def validate_portfolio(portfolio: list[dict]) -> None:
    """Raise ValueError early if `portfolio` is misconfigured."""
    if not portfolio:
        raise ValueError("Portfolio is empty — add at least one asset.")

    total_weight = sum(a["weight"] for a in portfolio)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.4f}")

    for asset in portfolio:
        name, asset_type = asset.get("name"), asset.get("type")
        if asset_type not in REQUIRED_FIELDS:
            raise ValueError(
                f"Asset '{name}': unknown type '{asset_type}'. "
                f"Must be one of {sorted(REQUIRED_FIELDS)}."
            )
        missing = REQUIRED_FIELDS[asset_type] - asset.keys()
        if missing:
            raise ValueError(
                f"Asset '{name}' (type={asset_type}) is missing required field(s): {sorted(missing)}"
            )
        if "esg_score" not in asset:
            raise ValueError(f"Asset '{name}' is missing 'esg_score'.")


# ---------------------------------------------------------------------
# Fixed income
# ---------------------------------------------------------------------

def get_zero_bond(face_value, years, weight, target_capital, data, x_known):
    """
    Price a zero-coupon bond and map its position risk to a single
    maturity bucket (`years`).

    Sensitivity note: `adj_cashflow = -tenor * position_value` is a
    duration-style *linear approximation* of dP/dy (not an exact
    repricing under a shifted curve) — standard for delta-normal VaR,
    but it will understate risk for large yield moves.
    """
    tenor = float(years)
    yield_rate = float(interpolate_yield_curve(data.iloc[-1], x_known, [tenor]).iloc[0])

    price = face_value / (1 + yield_rate) ** tenor
    asset_value = weight * target_capital
    asset_units = asset_value / price
    position_value = asset_units * price

    adj_cashflow = -tenor * position_value

    result = pd.DataFrame({"risk": [tenor], "yield": [yield_rate], "pv_cf": [position_value]})
    risk_matrix = pd.DataFrame({"risk": [tenor], "adj_cf": [adj_cashflow]})

    return {
        "units": float(asset_units),
        "value": float(position_value),
        "pricing": float(price),
        "result": result,
        "risk_matrix": risk_matrix,
    }


def get_coupon_bond(face_value, coupon_rate, freq, years, weight, target_capital, data, x_known):
    """
    Price a fixed coupon bond and map each coupon date to its own
    maturity risk bucket.
    """
    paytimes = np.arange(years, 0, -1 / freq)
    coupon = face_value * coupon_rate / freq

    yields = interpolate_yield_curve(data.iloc[-1], x_known, paytimes).values

    cash_flows = np.where(np.isclose(paytimes, years), face_value + coupon, coupon)
    pv_cashflows = cash_flows / (1 + yields) ** paytimes

    price = float(pv_cashflows.sum())
    asset_value = weight * target_capital
    asset_units = float(asset_value / price)
    position_value = float(asset_units * price)

    position_pv_cashflows = pv_cashflows * asset_units
    adj_cashflow = -paytimes * position_pv_cashflows

    result = pd.DataFrame({"risk": paytimes, "yield": yields, "pv_cf": position_pv_cashflows})
    risk_matrix = pd.DataFrame({"risk": paytimes, "adj_cf": adj_cashflow})

    return {
        "units": asset_units,
        "value": position_value,
        "pricing": price,
        "result": result,
        "risk_matrix": risk_matrix,
    }


# ---------------------------------------------------------------------
# Equity / FX
# ---------------------------------------------------------------------

def get_stock(stock_name, weight, target_capital, start_date, end_date):
    """
    Fetch the latest close price for a stock/FX ticker (any valid Yahoo
    Finance symbol) and map it to a ticker-named risk bucket
    (sensitivity ~= 1 to that ticker's own log return, i.e.
    dValue ~= position_value * d(log price)).
    """
    close_prices = fetch_stock_history(stock_name, start_date, end_date)
    price = float(close_prices.iloc[-1])

    asset_value = weight * target_capital
    asset_units = asset_value / price
    position_value = asset_units * price  # == asset_value up to float rounding

    result = pd.DataFrame({"risk": [stock_name], "pv_cf": [position_value]})
    risk_matrix = pd.DataFrame({"risk": [stock_name], "adj_cf": [position_value]})

    return {
        "units": float(asset_units),
        "value": float(position_value),
        "pricing": price,
        "result": result,
        "risk_matrix": risk_matrix,
    }


# ---------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------

def get_european_option(K, T, option_type, stock_name, weight, target_capital,
                         start_date, end_date, data, x_known):
    """
    Price a European call, put, or synthetic forward (call - put) with
    Black-Scholes, and map its risk to both the underlying stock
    (delta) and the T-year point on the yield curve (rho).

    Assumptions worth knowing before trusting these numbers: constant
    volatility estimated from the `start_date`-`end_date` sample
    (annualized, 252 trading days), no dividend yield, and European
    (not American) exercise. Real single-name equity/FX options are
    rarely well described by a single historical-vol Black-Scholes
    number — treat this as a teaching-grade approximation, not a
    trading model.

    `option_type` is set by the caller: "ECO" -> "call", "EPO" -> "put",
    "FC" -> "forward" (see assembly.build_asset_data).
    """
    close_prices = fetch_stock_history(stock_name, start_date, end_date)
    return_stock = np.log(close_prices / close_prices.shift(1)).dropna()

    sigma = return_stock.std() * np.sqrt(252)
    S = float(close_prices.iloc[-1])
    r = float(interpolate_yield_curve(data.iloc[-1], x_known, T).iloc[0])

    if sigma == 0 or np.isnan(sigma):
        raise ValueError(
            f"Estimated volatility for '{stock_name}' is zero/undefined over "
            f"{start_date}..{end_date} — widen the estimation window."
        )

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call
    delta_c = float(norm.cdf(d1))
    rho_c = float(K * T * np.exp(-r * T) * norm.cdf(d2))
    price_c = float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    # Put
    delta_p = float(norm.cdf(d1) - 1)
    rho_p = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
    price_p = float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    # Synthetic forward, by put-call parity
    price_f = float(price_c - price_p)

    asset_value = weight * target_capital

    if option_type == "call":
        asset_units = asset_value / price_c
        position_value = asset_units * price_c
        cf_stock = delta_c * S * asset_units
        cf_bond = rho_c * asset_units
        price = price_c
    elif option_type == "put":
        asset_units = asset_value / price_p
        position_value = asset_units * price_p
        cf_stock = delta_p * S * asset_units
        cf_bond = rho_p * asset_units
        price = price_p
    elif option_type == "forward":
        asset_units = asset_value / price_f
        position_value = asset_units * price_f
        cf_stock = (delta_c - delta_p) * S * asset_units
        cf_bond = (rho_c - rho_p) * asset_units
        price = price_f
    else:
        raise ValueError("option_type must be 'call', 'put', or 'forward'")

    if price <= 0:
        raise ValueError(
            f"Non-positive price ({price:.4f}) for {option_type} on '{stock_name}' "
            f"(K={K}, T={T}) — check the strike/maturity are realistic for this underlying."
        )

    result = pd.DataFrame({"risk": [stock_name, T], "pv_cf": [cf_stock, cf_bond]})
    risk_matrix = pd.DataFrame({"risk": [stock_name, T], "adj_cf": [cf_stock, cf_bond]})

    return {
        "units": float(asset_units),
        "value": float(position_value),
        "pricing": price,
        "result": result,
        "risk_matrix": risk_matrix,
    }
