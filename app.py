# -*- coding: utf-8 -*-
"""
Portfolio VaR Dashboard — Streamlit Interactive Web Application
Built on top of the `var_engine` module.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import functions from var_engine
from var_engine import (
    get_fred_yield_curve,
    build_portfolio,
    time_series_var,
    kupiec_test,
    clear_cache
)
from var_engine.assembly import build_risk_data
from var_engine.ui_helpers import (
    TABLE_COLUMNS,
    ASSET_TYPES,
    DEFAULT_PORTFOLIO_ROWS,
    dataframe_to_portfolio,
    normalize_weights,
    HORIZON_PRESETS,
    CONFIDENCE_PRESETS,
    STRATEGY_LABELS
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio VaR & Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Sidebar Settings & User Inputs
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Portfolio Settings")

st.sidebar.subheader("1. Capital & Horizon")
target_capital = st.sidebar.number_input(
    "Target Capital ($)",
    min_value=1_000,
    max_value=100_000_000,
    value=100_000,
    step=10_000,
    format="%d"
)

# Horizon & Confidence presets
horizon_label = st.sidebar.selectbox("Investment Horizon", list(HORIZON_PRESETS.keys()), index=5)
investment_horizon = HORIZON_PRESETS[horizon_label] if HORIZON_PRESETS[horizon_label] is not None else st.sidebar.number_input("Custom Days", 1, 500, 252)

confidence_label = st.sidebar.selectbox("Confidence Level", list(CONFIDENCE_PRESETS.keys()), index=3)
confidence = CONFIDENCE_PRESETS[confidence_label] if CONFIDENCE_PRESETS[confidence_label] is not None else st.sidebar.slider("Custom Confidence", 0.80, 0.999, 0.99, 0.005)

# Market Data Range
st.sidebar.subheader("2. Market Data Date Range")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Start Date", datetime(2025, 1, 1)).strftime("%Y-%m-%d")
with col_d2:
    end_date = st.date_input("End Date", datetime(2025, 12, 31)).strftime("%Y-%m-%d")

# Optimization Strategy
st.sidebar.subheader("3. Portfolio Strategy")
strategy_choice = st.sidebar.selectbox("Strategy", list(STRATEGY_LABELS.keys()))
strategy_name, use_esg = STRATEGY_LABELS[strategy_choice]

esg_target = None
if use_esg:
    esg_target = st.sidebar.slider("Minimum ESG Target Score", 0, 100, 75)

if st.sidebar.button("🧹 Clear Market Data Cache"):
    clear_cache()
    st.sidebar.success("Market data cache cleared successfully!")

# -----------------------------------------------------------------------------
# 3. Main Header & Portfolio Editor Table
# -----------------------------------------------------------------------------
st.title("📊 Multi-Asset Portfolio Value-at-Risk (VaR) Dashboard")
st.caption("Risk assessment & management dashboard for multi-asset portfolios (Stocks, Derivatives, FX, Bonds)")

with st.expander("📝 Portfolio Asset Allocation", expanded=True):
    st.markdown("Specify assets, types (`STK`, `FX`, `ZCB`, `CB`, `EPO`, `ECO`, `FC`), and target weights (weights must sum to 1.0).")
    
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame(DEFAULT_PORTFOLIO_ROWS)

    col_btn1, col_btn2 = st.columns([1.5, 5])
    with col_btn1:
        if st.button("⚖️ Normalize Weights to 1.0"):
            st.session_state.portfolio_df = normalize_weights(st.session_state.portfolio_df)
            st.rerun()

    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        column_config={
            "type": st.column_config.SelectboxColumn("Type", options=ASSET_TYPES, required=True),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
            "esg_score": st.column_config.NumberColumn("ESG Score", min_value=0, max_value=100),
        },
        use_container_width=True,
        key="portfolio_editor"
    )

# -----------------------------------------------------------------------------
# 4. Engine Processing
# -----------------------------------------------------------------------------
try:
    portfolio_config = dataframe_to_portfolio(edited_df)
    
    with st.spinner("⏳ Fetching FRED interest rates & historical market data..."):
        yield_data, x_known = get_fred_yield_curve(start_date, end_date)
        res = build_portfolio(
            portfolio=portfolio_config,
            investment_horizon=investment_horizon,
            target_capital=target_capital,
            data=yield_data,
            x_known=x_known,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy_name,
            confidence=confidence,
            esg_target=esg_target
        )

    # -------------------------------------------------------------------------
    # 5. Display Key Financial Metrics
    # -------------------------------------------------------------------------
    st.subheader("💡 Key Risk & Return Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric("Parametric VaR", f"${abs(res['portfolio_var_parametric'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_parametric']*100):.2f}%", delta_color="inverse")
    with m2:
        st.metric("Historical VaR", f"${abs(res['portfolio_var_historical'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_historical']*100):.2f}%", delta_color="inverse")
    with m3:
        st.metric("Monte Carlo VaR", f"${abs(res['portfolio_var_mc'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_mc']*100):.2f}%", delta_color="inverse")
    with m4:
        st.metric("Expected Return", f"{(res['portfolio_return']*100):.2f}%")
    with m5:
        st.metric("ESG / Sharpe Ratio", f"{res['esg_score']:.1f} Pt", f"Sharpe: {res['sharp']:.2f}")

    st.info(f"📌 **Beginner Note:** At a **{(confidence*100):.1f}%** confidence level, there is only a **{(100 - confidence*100):.1f}%** chance that this portfolio will lose more than the VaR values shown above over an **{investment_horizon}** trading day horizon.")

    # -------------------------------------------------------------------------
    # 6. Interactive Charts Section
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📈 Portfolio Allocation & Sensitivity", "🎲 VaR Method Comparison", "🔄 Backtesting & Kupiec Test"])

    # --- TAB 1: Allocation & Sensitivity ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🥧 Portfolio Asset Allocation")
            asset_names = [a['name'] for a in portfolio_config]
            weights = res['weight_asset']
            values = [res['value_asset'][name] for name in asset_names]
            
            fig_pie = px.pie(
                names=asset_names,
                values=weights,
                hover_data=[values],
                labels={"value": "Value ($)"},
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Value: $%{customdata[0]:,.2f}")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("##### 🎯 Risk Sensitivity by Risk Factor")
            final_out = res['final_output']
            fig_bar = px.bar(
                final_out,
                x='risk',
                y='adj_weight',
                text_auto='.2%',
                title="Risk Sensitivity (Adjusted Weight)",
                labels={'risk': 'Risk Factor (Tenor / Ticker)', 'adj_weight': 'Adjusted Weight'},
                color='adj_weight',
                color_continuous_scale='Reds'
            )
            fig_bar.update_traces(hovertemplate="<b>Risk Factor: %{x}</b><br>Risk Weight: %{y:.2%}")
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: VaR Comparison ---
    with tab2:
        st.markdown("##### 📊 Value-at-Risk (VaR) Comparison Across Models")
        var_data = pd.DataFrame({
            "Estimation Method": ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo Simulation"],
            "VaR (%)": [res['portfolio_var_parametric'], res['portfolio_var_historical'], res['portfolio_var_mc']],
            "VaR Value ($)": [abs(res['portfolio_var_parametric'] * target_capital), 
                              abs(res['portfolio_var_historical'] * target_capital), 
                              abs(res['portfolio_var_mc'] * target_capital)]
        })

        fig_var = px.bar(
            var_data,
            x="Estimation Method",
            y="VaR Value ($)",
            color="Estimation Method",
            text_auto="$,.2f",
            title=f"Maximum Expected Loss Estimation at {confidence:.1%} Confidence"
        )
        fig_var.update_traces(hovertemplate="<b>%{x}</b><br>Max Loss Estimate: $%{y:,.2f}")
        st.plotly_chart(fig_var, use_container_width=True)

    # --- TAB 3: Backtesting & Kupiec Test ---
    with tab3:
        st.markdown("##### 🔍 Historical Backtesting & Kupiec POF Test")
        
        # Calculate Next Day automatically based on market data End Date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        next_day = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        col_bt1, col_bt2 = st.columns([1, 3])
        with col_bt1:
            st.info(f"📅 **Backtest Start Date:** `{next_day}`\n\n*(Automatically set to the next day following End Date)*")
            test_days = st.number_input("Out-of-Sample Test Window (Days)", 30, 500, 100)
            run_bt = st.button("🚀 Run Backtest Analysis")

        if run_bt:
            with st.spinner("Executing rolling backtest and calculating Kupiec POF test..."):
                bt_df = time_series_var(
                    result=res,
                    target_capital=target_capital,
                    new_start_date=next_day,
                    N_test=test_days,
                    confidence=confidence
                )

                # Interactive Plotly Chart
                fig_bt = go.Figure()

                # Actual Return
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['return_port'],
                    mode='lines', name='Actual Return',
                    line=dict(color='black', width=1.5)
                ))

                # VaR lines
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['var_parametric'],
                    mode='lines', name='Parametric VaR', line=dict(dash='dash', color='#1f77b4')
                ))
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['var_historical'],
                    mode='lines', name='Historical VaR', line=dict(dash='dash', color='#2ca02c')
                ))
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['var_mc'],
                    mode='lines', name='Monte Carlo VaR', line=dict(dash='dash', color='#ff7f0e')
                ))

                # Highlight Breaches
                breaches = bt_df[bt_df['return_port'] < bt_df['var_parametric']]
                fig_bt.add_trace(go.Scatter(
                    x=breaches.index, y=breaches['return_port'],
                    mode='markers', name='VaR Breaches',
                    marker=dict(color='red', size=8, symbol='x')
                ))

                fig_bt.update_layout(
                    title="Realized Portfolio Returns vs. Rolling VaR Estimates",
                    xaxis_title="Date",
                    yaxis_title="Return / VaR Level",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                # Kupiec POF Test Summary
                kupiec_res = kupiec_test(bt_df['var_parametric'], bt_df['return_port'], confidence)
                
                st.markdown("### 📋 Kupiec POF Test Statistical Results")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Observations (N)", f"{kupiec_res['n_obs']} Days")
                k2.metric("Total Breaches (x)", f"{kupiec_res['n_breaches']} Times")
                k3.metric("Observed / Expected Rate", f"{kupiec_res['breach_rate']:.2%} / {kupiec_res['expected_rate']:.2%}")
                
                status_color = "green" if kupiec_res['passed'] else "red"
                status_text = "PASSED (Model is Accurate)" if kupiec_res['passed'] else "FAILED (Model Risk Unreliable)"
                k4.markdown(f"**Model Status:**<br><span style='color:{status_color}; font-size: 18px; font-weight:bold;'>{status_text}</span>", unsafe_unsafe_html=True if 'unsafe_unsafe_html' in locals() else True)

except Exception as e:
    st.error(f"⚠️ Calculation Error: {str(e)}")
    st.info("💡 Tip: Verify ticker symbols and ensure portfolio weights sum to 1.0.")