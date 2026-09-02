# -*- coding: utf-8 -*-
"""
UI-agnostic helpers for the Streamlit dashboard.

Kept separate from app.py (which imports Streamlit) so this conversion
logic — the part most worth unit-testing — can be tested with plain
pandas DataFrames, no Streamlit runtime required.
"""

from __future__ import annotations

import pandas as pd

# Columns shown in the dashboard's editable portfolio table. One flat
# table covers all 7 asset types; irrelevant fields for a given row's
# `type` are simply left blank (mirrors how the original portfolio list
# already used different dict keys per asset type).
TABLE_COLUMNS = [
    "name", "type", "ticker", "weight", "esg_score",
    "face_value", "years", "coupon_rate", "freq", "K", "T",
]

ASSET_TYPES = ["STK", "FX", "ZCB", "CB", "ECO", "EPO", "FC"]

# Which extra (non-name/type/weight/esg_score) columns each type needs,
# and which engine dict key the "ticker" column maps to.
TYPE_FIELD_MAP = {
    "STK": {"required": ["ticker"], "ticker_key": "stock_name"},
    "FX": {"required": ["ticker"], "ticker_key": "fx_name"},
    "ZCB": {"required": ["face_value", "years"], "ticker_key": None},
    "CB": {"required": ["face_value", "coupon_rate", "freq", "years"], "ticker_key": None},
    "ECO": {"required": ["ticker", "K", "T"], "ticker_key": "stock_name"},
    "EPO": {"required": ["ticker", "K", "T"], "ticker_key": "stock_name"},
    "FC": {"required": ["ticker", "K", "T"], "ticker_key": "stock_name"},
}

# Default rows: the same example portfolio as the original script
# (70% NVDA stock / 20% NVDA put / 10% THB FX), so the dashboard shows
# a working result on first load instead of an empty table.
DEFAULT_PORTFOLIO_ROWS = [
    {"name": "S1", "type": "STK", "ticker": "NVDA", "weight": 0.70, "esg_score": 75,
     "face_value": None, "years": None, "coupon_rate": None, "freq": None, "K": None, "T": None},
    {"name": "E1", "type": "EPO", "ticker": "NVDA", "weight": 0.20, "esg_score": 80,
     "face_value": None, "years": None, "coupon_rate": None, "freq": None, "K": 250.0, "T": 1.0},
    {"name": "G", "type": "FX", "ticker": "THB=X", "weight": 0.10, "esg_score": 90,
     "face_value": None, "years": None, "coupon_rate": None, "freq": None, "K": None, "T": None},
]

HORIZON_PRESETS = {
    "1 day": 1,
    "1 week (5 trading days)": 5,
    "1 month (21 trading days)": 21,
    "3 months (63 trading days)": 63,
    "6 months (126 trading days)": 126,
    "1 year (252 trading days)": 252,
    "Custom": None,
}

CONFIDENCE_PRESETS = {
    "90%": 0.90,
    "95%": 0.95,
    "97.5%": 0.975,
    "99%": 0.99,
    "99.5%": 0.995,
    "Custom": None,
}

STRATEGY_LABELS = {
    "Given weights (no optimization)": ("given", False),
    "Minimum risk": ("min_risk", False),
    "Minimum VaR": ("min_var", False),
    "Maximum Sharpe ratio": ("max_sharpe", False),
    "Maximum Sharpe ratio + ESG floor": ("max_sharpe", True),
}


def _to_float(value, field_name, row_label):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"Asset '{row_label}': '{field_name}' is required for this asset type.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Asset '{row_label}': '{field_name}' must be a number, got {value!r}.")


def dataframe_to_portfolio(df: pd.DataFrame) -> list[dict]:
    """
    Convert the dashboard's editable table (one flat DataFrame with
    TABLE_COLUMNS) into the list-of-dicts shape var_engine expects.

    Raises ValueError with a specific, user-facing message on the first
    problem found (missing field, bad type, duplicate name, ...) —
    the Streamlit layer just needs to catch ValueError and st.error() it.
    """
    if df is None or df.empty:
        raise ValueError("The portfolio table is empty — add at least one asset.")

    portfolio = []
    seen_names = set()

    for i, row in df.reset_index(drop=True).iterrows():
        row_label = str(row.get("name") or f"row {i + 1}")

        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError(f"Row {i + 1}: 'name' is required.")
        if name in seen_names:
            raise ValueError(f"Asset name '{name}' is used more than once — names must be unique.")
        seen_names.add(name)

        asset_type = str(row.get("type") or "").strip().upper()
        if asset_type not in TYPE_FIELD_MAP:
            raise ValueError(
                f"Asset '{name}': type must be one of {ASSET_TYPES}, got {row.get('type')!r}."
            )

        weight = _to_float(row.get("weight"), "weight", row_label)
        esg_score = _to_float(row.get("esg_score"), "esg_score", row_label)

        asset = {"name": name, "type": asset_type, "weight": weight, "esg_score": esg_score}
        spec = TYPE_FIELD_MAP[asset_type]

        if spec["ticker_key"] is not None:
            ticker = str(row.get("ticker") or "").strip()
            if not ticker:
                raise ValueError(f"Asset '{name}' (type={asset_type}): a Yahoo Finance ticker is required.")
            asset[spec["ticker_key"]] = ticker

        for field in spec["required"]:
            if field == "ticker":
                continue
            asset[field] = _to_float(row.get(field), field, row_label)

        if asset_type == "CB":
            asset["freq"] = int(asset["freq"])  # payments/year must be an integer

        portfolio.append(asset)

    total_weight = sum(a["weight"] for a in portfolio)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0 — currently sum to {total_weight:.4f}. "
            "Use the 'Normalize weights' button or adjust the table."
        )

    return portfolio


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale the 'weight' column so it sums to 1.0 (no-op if already empty/zero-sum)."""
    df = df.copy()
    weights = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    total = weights.sum()
    if total > 0:
        df["weight"] = weights / total
    return df
