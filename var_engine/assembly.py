# -*- coding: utf-8 -*-
"""Portfolio assembly — price every position and build one aligned risk matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import fetch_stock_history
from .market_data import interpolate_yield_curve
from .pricing import get_coupon_bond, get_european_option, get_stock, get_zero_bond


def build_asset_data(portfolio, weight_list, target_capital, data, x_known, start_date, end_date):
    """Price every asset in `portfolio` and collect pricing/units/value/risk data."""
    pricing_asset, data_asset, risk_matrices = {}, {}, {}
    asset_values, asset_units = {}, {}

    for weight, asset in zip(weight_list, portfolio):
        name, asset_type = asset["name"], asset["type"]

        if asset_type == "ZCB":
            result = get_zero_bond(
                face_value=asset["face_value"], years=asset["years"],
                weight=weight, target_capital=target_capital,
                data=data, x_known=x_known,
            )
        elif asset_type == "CB":
            result = get_coupon_bond(
                face_value=asset["face_value"], coupon_rate=asset["coupon_rate"],
                freq=asset["freq"], years=asset["years"],
                weight=weight, target_capital=target_capital,
                data=data, x_known=x_known,
            )
        elif asset_type == "STK":
            result = get_stock(
                stock_name=asset["stock_name"], weight=weight, target_capital=target_capital,
                start_date=start_date, end_date=end_date,
            )
        elif asset_type == "FX":
            result = get_stock(
                stock_name=asset["fx_name"], weight=weight, target_capital=target_capital,
                start_date=start_date, end_date=end_date,
            )
        elif asset_type == "ECO":
            result = get_european_option(
                K=asset["K"], T=asset["T"], option_type="call",
                stock_name=asset["stock_name"], weight=weight, target_capital=target_capital,
                start_date=start_date, end_date=end_date, data=data, x_known=x_known,
            )
        elif asset_type == "EPO":
            result = get_european_option(
                K=asset["K"], T=asset["T"], option_type="put",
                stock_name=asset["stock_name"], weight=weight, target_capital=target_capital,
                start_date=start_date, end_date=end_date, data=data, x_known=x_known,
            )
        elif asset_type == "FC":
            result = get_european_option(
                K=asset["K"], T=asset["T"], option_type="forward",
                stock_name=asset["stock_name"], weight=weight, target_capital=target_capital,
                start_date=start_date, end_date=end_date, data=data, x_known=x_known,
            )
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

        pricing_asset[name] = result["pricing"]
        data_asset[name] = result["result"]
        risk_matrices[name] = result["risk_matrix"]
        asset_values[name] = result["value"]
        asset_units[name] = result["units"]

    return pricing_asset, data_asset, risk_matrices, asset_values, asset_units


def build_portfolio_risk_matrix(risk_matrices, portfolio_value):
    """Combine every asset's risk_matrix into one table of adj_cf / adj_weight by risk bucket."""
    all_data = pd.concat(risk_matrices, ignore_index=True)
    is_numeric = pd.to_numeric(all_data["risk"], errors="coerce").notna()

    df_num = all_data[is_numeric].copy()
    df_num["risk"] = df_num["risk"].astype(float)
    output_num = df_num.groupby("risk", as_index=False)["adj_cf"].sum()

    df_str = all_data[~is_numeric].copy()
    output_str = df_str.groupby("risk", as_index=False)["adj_cf"].sum()

    final_output = pd.concat([output_num, output_str], ignore_index=True)
    final_output["adj_weight"] = final_output["adj_cf"] / portfolio_value

    return output_num, output_str, final_output


def build_yield_risk(data, x_known, risk_tenors):
    """Interpolate the historical yield curve at each risk tenor, then take day-over-day changes."""
    data_yields = data.apply(lambda row: interpolate_yield_curve(row, x_known, risk_tenors), axis=1)
    data_yields = data_yields.rename(columns=risk_tenors)
    data_yields = data_yields.rename_axis("Date")
    data_yields.index = data_yields.index.strftime("%Y-%m-%d")

    risk_yields = (data_yields - data_yields.shift(1)).dropna()
    return data_yields, risk_yields


def build_stock_risk(stock_tickers, start_date, end_date):
    """Fetch daily closes for each ticker (cached) and return their log returns."""
    close_prices = []
    for ticker in stock_tickers:
        # fetch_stock_history already returns a tz-naive index (see cache.py)
        s = fetch_stock_history(ticker, start_date, end_date).copy()
        s.index = s.index.normalize()
        s.name = ticker
        close_prices.append(s)

    if not close_prices:
        return pd.DataFrame()

    close_df = pd.concat(close_prices, axis=1)
    close_df.index = close_df.index.strftime("%Y-%m-%d")

    risk_stock = np.log(close_df / close_df.shift(1)).dropna()
    return risk_stock


def build_risk_data(risk_yields, risk_stock):
    """Combine yield and stock risk factors into one aligned covariance / mean estimate."""
    data_risk = pd.concat([risk_yields, risk_stock], axis=1).dropna()
    data_risk_cov = data_risk.cov()
    data_risk_mean = data_risk.mean()
    return data_risk, data_risk_cov, data_risk_mean
