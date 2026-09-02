# -*- coding: utf-8 -*-
"""
Streamlit dashboard for var_engine — Portfolio Value-at-Risk.

Run locally:
    streamlit run app.py

Everything users are expected to change lives in the sidebar (assets,
dates, horizon, confidence, strategy) or the portfolio table — no code
editing required. See README.md for the full field reference.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from var_engine import (
    build_portfolio,
    clear_cache,
    get_fred_yield_curve,
    kupiec_test,
    time_series_var,
)
from var_engine.display import plot_portfolio_var_breaches
from var_engine.ui_helpers import (
    CONFIDENCE_PRESETS,
    DEFAULT_PORTFOLIO_ROWS,
    HORIZON_PRESETS,
    STRATEGY_LABELS,
    TABLE_COLUMNS,
    dataframe_to_portfolio,
    normalize_weights,
)

st.set_page_config(page_title="Portfolio VaR Dashboard", layout="wide", page_icon="📉")

# ---------------------------------------------------------------------
# Cached wrapper around the FRED yield-curve fetch (on top of var_engine's
# own lru_cache on the raw series — this caches the assembled DataFrame
# across Streamlit re-runs within a session too).
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_yield_curve(start_date: str, end_date: str):
    return get_fred_yield_curve(start_date, end_date)


st.title("📉 Portfolio Value-at-Risk Dashboard")
st.caption(
    "Multi-asset VaR (Parametric / Historical / Monte Carlo), portfolio optimization, "
    "and Kupiec backtesting — built on live Yahoo Finance & FRED data."
)

# =======================================================================
# SIDEBAR — everything that isn't the asset table itself
# =======================================================================
with st.sidebar:
    st.header("Portfolio settings")

    target_capital = st.number_input(
        "Target capital (USD)", min_value=1_000.0, value=100_000.0, step=1_000.0,
        help="Total portfolio size. Each asset's dollar allocation = weight x this value.",
    )

    st.subheader("Historical estimation window")
    st.caption("Prices, means, and covariances for VaR are estimated from this date range.")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start date", value=pd.Timestamp("2025-01-01"))
    with col_b:
        end_date = st.date_input("End date", value=pd.Timestamp("2025-12-31"))

    st.subheader("Investment horizon")
    horizon_choice = st.selectbox(
        "Holding period for the VaR estimate", list(HORIZON_PRESETS.keys()), index=5,
    )
    if HORIZON_PRESETS[horizon_choice] is None:
        investment_horizon = st.number_input("Custom horizon (trading days)", min_value=1, value=252, step=1)
    else:
        investment_horizon = HORIZON_PRESETS[horizon_choice]

    st.subheader("Confidence level")
    conf_choice = st.selectbox("VaR confidence level", list(CONFIDENCE_PRESETS.keys()), index=3)
    if CONFIDENCE_PRESETS[conf_choice] is None:
        confidence = st.slider("Custom confidence", min_value=0.80, max_value=0.999, value=0.99, step=0.001)
    else:
        confidence = CONFIDENCE_PRESETS[conf_choice]

    st.subheader("Optimization strategy")
    strategy_choice = st.radio("Weighting strategy", list(STRATEGY_LABELS.keys()), index=0)
    strategy, needs_esg = STRATEGY_LABELS[strategy_choice]

    esg_target = None
    if needs_esg:
        esg_target = st.slider("Minimum weighted ESG score", 0, 100, 80)

    if strategy != "given":
        col_c, col_d = st.columns(2)
        with col_c:
            min_weight = st.number_input("Min weight/asset", 0.0, 1.0, 0.0, 0.05)
        with col_d:
            max_weight = st.number_input("Max weight/asset", 0.0, 1.0, 1.0, 0.05)
    else:
        min_weight, max_weight = 0.0, 1.0

    compare_all = st.checkbox(
        "Compare all 5 strategies", value=False,
        help="Runs given / min-risk / min-VaR / max-Sharpe / max-Sharpe+ESG back to back. "
             "Slower — each optimizer run re-prices the portfolio 10-50+ times.",
    )

    with st.expander("Advanced: Monte Carlo & cache"):
        mc_simulations = st.number_input(
            "Monte Carlo simulations", min_value=1_000, max_value=500_000, value=50_000, step=1_000,
            help="Lower = faster, noisier. 50,000 is a reasonable interactive default; "
                 "the original notebook used 100,000.",
        )
        random_seed = st.number_input(
            "Random seed (Monte Carlo)", min_value=0, value=42, step=1,
            help="Fixes Monte Carlo VaR so re-running with the same inputs gives the same number.",
        )
        if st.button("🔄 Refresh cached market data"):
            clear_cache()
            _cached_yield_curve.clear()
            st.success("Cache cleared — next run re-fetches from Yahoo Finance / FRED.")

# =======================================================================
# MAIN — portfolio table
# =======================================================================
st.subheader("1. Portfolio")
st.caption(
    "Any ticker on [Yahoo Finance](https://finance.yahoo.com) works in the **ticker** column — "
    "US stocks (`AAPL`, `NVDA`), Thai SET stocks (`PTT.BK`, `AOT.BK`), FX (`THB=X`, `EURUSD=X`), "
    "crypto (`BTC-USD`), and more. Add/remove rows directly in the table."
)

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(DEFAULT_PORTFOLIO_ROWS, columns=TABLE_COLUMNS)

edited_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    width='stretch',
    key="portfolio_editor",
    column_config={
        "name": st.column_config.TextColumn("Name", help="Short unique label, e.g. 'S1'.", required=True),
        "type": st.column_config.SelectboxColumn(
            "Type", options=["STK", "FX", "ZCB", "CB", "ECO", "EPO", "FC"], required=True,
            help="STK=Stock, FX=FX spot, ZCB=Zero-coupon bond, CB=Coupon bond, "
                 "ECO=European call, EPO=European put, FC=Forward.",
        ),
        "ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol (STK/FX/ECO/EPO/FC only)."),
        "weight": st.column_config.NumberColumn(
            "Weight", min_value=0.0, max_value=1.0, step=0.01, format="%.4f",
            help="Share of target capital. All weights must sum to 1.0.",
        ),
        "esg_score": st.column_config.NumberColumn("ESG score", min_value=0, max_value=100, step=1),
        "face_value": st.column_config.NumberColumn("Face value", help="ZCB/CB only."),
        "years": st.column_config.NumberColumn("Years to maturity", help="ZCB/CB only."),
        "coupon_rate": st.column_config.NumberColumn("Coupon rate", help="CB only, e.g. 0.03 = 3%.", format="%.4f"),
        "freq": st.column_config.NumberColumn("Coupon freq/yr", help="CB only, e.g. 2 = semiannual.", step=1),
        "K": st.column_config.NumberColumn("Strike / forward price (K)", help="ECO/EPO/FC only."),
        "T": st.column_config.NumberColumn("Years to expiry (T)", help="ECO/EPO/FC only."),
    },
)
st.session_state.portfolio_df = edited_df

weight_sum = pd.to_numeric(edited_df["weight"], errors="coerce").fillna(0).sum()
col_w1, col_w2 = st.columns([3, 1])
with col_w1:
    if abs(weight_sum - 1.0) < 1e-6:
        st.success(f"Weights sum to {weight_sum:.4f} ✓")
    else:
        st.warning(f"Weights currently sum to {weight_sum:.4f} — must equal 1.0000 before running.")
with col_w2:
    if st.button("Normalize weights"):
        st.session_state.portfolio_df = normalize_weights(edited_df)
        st.rerun()

run_clicked = st.button("▶ Run VaR analysis", type="primary")

# =======================================================================
# RUN
# =======================================================================
if run_clicked:
    try:
        portfolio = dataframe_to_portfolio(edited_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    try:
        with st.spinner("Fetching FRED yield curve..."):
            data, x_known = _cached_yield_curve(start_str, end_str)
    except Exception as e:
        st.error(f"Failed to fetch the FRED yield curve: {e}")
        st.stop()

    def run_one(strat, esg_t):
        return build_portfolio(
            portfolio, investment_horizon, target_capital, data, x_known,
            start_str, end_str, strategy=strat,
            min_weight=min_weight, max_weight=max_weight,
            confidence=confidence, esg_target=esg_t,
            mc_simulations=mc_simulations, random_state=int(random_seed),
        )

    try:
        if compare_all:
            esg_floor = esg_target or 80
            runs = [
                ("Given weights", "given", None),
                ("Minimum risk", "min_risk", None),
                ("Minimum VaR", "min_var", None),
                ("Maximum Sharpe", "max_sharpe", None),
                (f"Max Sharpe + ESG>={esg_floor}", "max_sharpe", esg_floor),
            ]
            # Maps the sidebar's (strategy, needs_esg) choice to the matching
            # label above, so "main_result" below reflects what the user
            # actually selected, not just the first strategy computed.
            strategy_to_label = {
                ("given", False): "Given weights",
                ("min_risk", False): "Minimum risk",
                ("min_var", False): "Minimum VaR",
                ("max_sharpe", False): "Maximum Sharpe",
                ("max_sharpe", True): f"Max Sharpe + ESG>={esg_floor}",
            }
            results = {}
            progress = st.progress(0.0, text="Running strategies...")
            for i, (label, strat, esg_t) in enumerate(runs):
                progress.progress(i / len(runs), text=f"Running: {label}")
                results[label] = run_one(strat, esg_t)
            progress.progress(1.0, text="Done.")
            time.sleep(0.2)
            progress.empty()
            main_result = results[strategy_to_label[(strategy, needs_esg)]]
        else:
            with st.spinner(f"Pricing portfolio and running '{strategy_choice}'..."):
                main_result = run_one(strategy, esg_target)
            results = {strategy_choice: main_result}
    except Exception as e:
        st.error(f"Portfolio calculation failed: {e}")
        st.stop()

    st.session_state.last_result = main_result
    st.session_state.last_results_all = results
    st.session_state.last_portfolio = portfolio
    st.session_state.last_dates = (start_str, end_str)

# =======================================================================
# RESULTS
# =======================================================================
if "last_result" in st.session_state:
    result = st.session_state.last_result
    results_all = st.session_state.last_results_all

    st.subheader("2. Results")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio value", f"${result['portfolio_value']:,.0f}")
    m2.metric("Expected return", f"{result['portfolio_return'] * 100:.2f}%")
    m3.metric("Volatility (risk)", f"{result['portfolio_risk'] * 100:.2f}%")
    m4.metric("Sharpe ratio", f"{result['sharp']:.2f}")
    m5.metric("ESG score", f"{result['esg_score']:.1f}")

    st.markdown(f"**Value at Risk** — {confidence * 100:.1f}% confidence, "
                f"{investment_horizon}-trading-day horizon:")
    v1, v2, v3 = st.columns(3)
    for col, key, label in [
        (v1, "portfolio_var_parametric", "Parametric VaR"),
        (v2, "portfolio_var_historical", "Historical VaR"),
        (v3, "portfolio_var_mc", "Monte Carlo VaR"),
    ]:
        var_pct = result[key]
        var_dollar = var_pct * result["portfolio_value"]
        col.metric(label, f"${var_dollar:,.0f}", f"{var_pct * 100:.2f}% of capital")

    with st.expander("Asset-level detail (weights, pricing, units, value)"):
        assets_df = pd.DataFrame({
            "Weight": {n: w for n, w in zip([a["name"] for a in st.session_state.last_portfolio],
                                             result["weight_asset"])},
            "Price": result["pricing_asset"],
            "Units": result["units_asset"],
            "Value ($)": result["value_asset"],
        })
        st.dataframe(assets_df.style.format({"Weight": "{:.4f}", "Price": "{:.4f}",
                                              "Units": "{:.4f}", "Value ($)": "{:,.2f}"}))

    with st.expander("Risk-factor sensitivity (adjusted cash flow by risk bucket)"):
        risk_df = result["final_output"].copy()
        risk_df["risk"] = risk_df["risk"].astype(str)
        st.bar_chart(risk_df.set_index("risk")["adj_weight"])
        st.dataframe(risk_df.style.format({"adj_cf": "{:,.2f}", "adj_weight": "{:.6f}"}))

    if len(results_all) > 1:
        st.markdown("**Strategy comparison**")
        comp_rows = []
        for label, r in results_all.items():
            comp_rows.append({
                "Strategy": label,
                "Return %": r["portfolio_return"] * 100,
                "Risk %": r["portfolio_risk"] * 100,
                "Sharpe": r["sharp"],
                "ESG": r["esg_score"],
                "VaR Parametric": r["portfolio_var_parametric"] * r["portfolio_value"],
                "VaR Historical": r["portfolio_var_historical"] * r["portfolio_value"],
                "VaR Monte Carlo": r["portfolio_var_mc"] * r["portfolio_value"],
            })
        comp_df = pd.DataFrame(comp_rows).set_index("Strategy")
        st.dataframe(comp_df.style.format("{:,.2f}"))

    # -------------------------------------------------------------
    # 3. Backtest (separate, on-demand — this is the slow one)
    # -------------------------------------------------------------
    st.subheader("3. Backtest (Kupiec Proportion-of-Failures test)")
    st.caption(
        "Rolls the estimation window forward day by day over an out-of-sample period and checks "
        "whether realized returns breached VaR about as often as the confidence level implies. "
        "This re-runs Monte Carlo on every test day, so it's the slowest part of the app — "
        "run it on demand, not automatically."
    )

    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        n_test = st.slider("Out-of-sample days to test", 10, 250, 60)
    with bt2:
        window_size = st.slider("Rolling estimation window (days)", 60, 500, 250)
    with bt3:
        bt_mc_sims = st.number_input("MC sims per day (backtest)", 500, 50_000, 5_000, 500)

    if st.button("▶ Run backtest"):
        start_str, end_str = st.session_state.last_dates
        new_start_date = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        progress_bar = st.progress(0.0, text="Backtesting...")

        def _cb(done, total):
            progress_bar.progress(done / max(total, 1), text=f"Backtesting day {done}/{total}")

        try:
            bt_output = time_series_var(
                result, target_capital, new_start_date, n_test,
                confidence=confidence, window_size=window_size,
                mc_simulations=int(bt_mc_sims), random_state=int(random_seed),
                progress_callback=_cb,
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"Backtest failed: {e}")
            st.stop()

        progress_bar.empty()

        if bt_output.empty:
            st.warning(
                "No valid out-of-sample days were produced — this usually means there isn't "
                f"{window_size} days of history available after {end_str}, or the date range "
                "has no trading days yet. Try a smaller rolling window or an earlier end date."
            )
        else:
            fig = plot_portfolio_var_breaches(bt_output)
            st.pyplot(fig)

            st.markdown("**Kupiec Proportion-of-Failures test**")
            kupiec_rows = []
            for label, col in [
                ("Parametric VaR", "var_parametric"),
                ("Historical VaR", "var_historical"),
                ("Monte Carlo VaR", "var_mc"),
            ]:
                r = kupiec_test(bt_output[col], bt_output["return_port"], confidence)
                kupiec_rows.append({
                    "Method": label,
                    "Breaches": f"{r['n_breaches']} / {r['n_obs']}",
                    "Breach rate": f"{r['breach_rate'] * 100:.2f}%",
                    "Expected rate": f"{r['expected_rate'] * 100:.2f}%",
                    "LR stat": f"{r['lr_stat']:.3f}",
                    "p-value": f"{r['p_value']:.4f}",
                    "Result": "✅ Pass" if r["passed"] else "❌ Fail",
                })
            st.dataframe(pd.DataFrame(kupiec_rows), width='stretch', hide_index=True)
else:
    st.info("Configure the portfolio above and click **Run VaR analysis** to see results.")
