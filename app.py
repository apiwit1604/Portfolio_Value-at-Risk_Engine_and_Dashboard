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

# นำเข้าฟังก์ชันจากโฟลเดอร์ var_engine
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

# แต่งสไตล์เบื้องต้นให้ดูสะอาดตา
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
st.sidebar.title("⚙️ ตั้งค่าพอร์ตและการวิเคราะห์")

st.sidebar.subheader("1. กำหนดเงินทุนและระยะเวลา")
target_capital = st.sidebar.number_input(
    "เงินลงทุนรวม (Target Capital $)",
    min_value=1_000,
    max_value=100_000_000,
    value=100_000,
    step=10_000,
    format="%d"
)

# ตัวเลือก Horizon & Confidence
horizon_label = st.sidebar.selectbox("ระยะเวลาถือครอง (Horizon)", list(HORIZON_PRESETS.keys()), index=5)
investment_horizon = HORIZON_PRESETS[horizon_label] if HORIZON_PRESETS[horizon_label] is not None else st.sidebar.number_input("จำนวนวัน (Custom)", 1, 500, 252)

confidence_label = st.sidebar.selectbox("ระดับความเชื่อมั่น (Confidence Level)", list(CONFIDENCE_PRESETS.keys()), index=3)
confidence = CONFIDENCE_PRESETS[confidence_label] if CONFIDENCE_PRESETS[confidence_label] is not None else st.sidebar.slider("Custom Confidence", 0.80, 0.999, 0.99, 0.005)

# ย้อนหลังข้อมูลตลาด
st.sidebar.subheader("2. ช่วงเวลาข้อมูลตลาด (Market Data Range)")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("เริ่มวันที่", datetime(2025, 1, 1)).strftime("%Y-%m-%d")
with col_d2:
    end_date = st.date_input("สิ้นสุดวันที่", datetime(2025, 12, 31)).strftime("%Y-%m-%d")

# กลยุทธ์การจัดพอร์ต
st.sidebar.subheader("3. กลยุทธ์พอร์ตการลงทุน")
strategy_choice = st.sidebar.selectbox("เลือกกลยุทธ์ (Strategy)", list(STRATEGY_LABELS.keys()))
strategy_name, use_esg = STRATEGY_LABELS[strategy_choice]

esg_target = None
if use_esg:
    esg_target = st.sidebar.slider("ESG Score ขั้นต่ำที่ต้องการ", 0, 100, 75)

if st.sidebar.button("🧹 ล้าง Cache ข้อมูลตลาด"):
    clear_cache()
    st.sidebar.success("ล้างข้อมูล Cache เรียบร้อยแล้ว!")

# -----------------------------------------------------------------------------
# 3. Main Header & Portfolio Editor Table
# -----------------------------------------------------------------------------
st.title("📊 Multi-Asset Portfolio Value-at-Risk (VaR) Dashboard")
st.caption("แผงควบคุมประเมินความเสี่ยงพอร์ตการลงทุนหลายสินทรัพย์ (หุ้น, สัญญาอนุพันธ์, FX, พันธบัตร)")

with st.expander("📝 ปรับแต่งสัดส่วนสินทรัพย์ในพอร์ต (Portfolio Allocation)", expanded=True):
    st.markdown("ระบุสินทรัพย์ ประเภท (STK, FX, ZCB, CB, EPO, ECO, FC) และสัดส่วนน้ำหนัก (Weight รวมกันต้องได้ 1.0)")
    
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame(DEFAULT_PORTFOLIO_ROWS)

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if st.button("⚖️ ปรับ Weight ให้รวมได้ 1.0 (Normalize)"):
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
    # แปลงข้อมูลจากตารางไปเป็นโครงสร้างที่ var_engine ต้องการ
    portfolio_config = dataframe_to_portfolio(edited_df)
    
    with st.spinner("⏳ กำลังดึงข้อมูลอัตราดอกเบี้ย FRED และราคาตลาดย้อนหลัง..."):
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
    st.subheader("💡 สรุปตัวเลขความเสี่ยงและผลตอบแทน (Key Metrics)")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric("มูลค่าความเสี่ยง Parametric VaR", f"${abs(res['portfolio_var_parametric'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_parametric']*100):.2f}%", delta_color="inverse")
    with m2:
        st.metric("มูลค่าความเสี่ยง Historical VaR", f"${abs(res['portfolio_var_historical'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_historical']*100):.2f}%", delta_color="inverse")
    with m3:
        st.metric("มูลค่าความเสี่ยง Monte Carlo VaR", f"${abs(res['portfolio_var_mc'] * target_capital):,.2f}", 
                  f"{(res['portfolio_var_mc']*100):.2f}%", delta_color="inverse")
    with m4:
        st.metric("ผลตอบแทนคาดหวัง (Expected Return)", f"{(res['portfolio_return']*100):.2f}%")
    with m5:
        st.metric("คะแนน ESG / Sharpe Ratio", f"{res['esg_score']:.1f} Pt", f"Sharpe: {res['sharp']:.2f}")

    st.info(f"📌 **คำอธิบายสำหรับผู้เริ่มต้น:** ค่า VaR (Value at Risk) ที่ระดับความเชื่อมั่น **{(confidence*100):.1f}%** หมายถึง มีโอกาสเพียง **{(100 - confidence*100):.1f}%** ที่พอร์ตนี้จะขาดทุนเกินกว่าจำนวนเงินข้างต้น ในช่วงเวลา **{investment_horizon}** วันทำการ")

    # -------------------------------------------------------------------------
    # 6. Interactive Charts Section
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📈 สัดส่วนการลงทุน & ความเสี่ยง", "🎲 การเปรียบเทียบ VaR 3 โมเดล", "🔄 Backtesting (ย้อนรอยความเสี่ยง)"])

    # --- TAB 1: Allocation & Sensitivity ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🥧 สัดส่วนน้ำหนักลงทุนแยกตามสินทรัพย์ (Weight Allocation)")
            asset_names = [a['name'] for a in portfolio_config]
            weights = res['weight_asset']
            values = [res['value_asset'][name] for name in asset_names]
            
            fig_pie = px.pie(
                names=asset_names,
                values=weights,
                hover_data=[values],
                labels={"value":"มูลค่า ($)"},
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate="<b>%{label}</b><br>น้ำหนัก: %{percent}<br>มูลค่า: $%{customdata[0]:,.2f}")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("##### 🎯 สัดส่วนความเสี่ยงตามปัจจัยเสี่ยง (Risk Factor Weight)")
            final_out = res['final_output']
            fig_bar = px.bar(
                final_out,
                x='risk',
                y='adj_weight',
                text_auto='.2%',
                title="Risk Sensitivity (Adjusted Weight)",
                labels={'risk': 'ปัจจัยเสี่ยง (Risk Bucket / Ticker)', 'adj_weight': 'สัดส่วนความเสี่ยง'},
                color='adj_weight',
                color_continuous_scale='Reds'
            )
            fig_bar.update_traces(hovertemplate="<b>ปัจจัยเสี่ยง: %{x}</b><br>สัดส่วนความเสี่ยง: %{y:.2%}")
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: VaR Comparison ---
    with tab2:
        st.markdown("##### 📊 เปรียบเทียบผลลัพธ์ VaR จาก 3 วิธีคำนวณ")
        var_data = pd.DataFrame({
            "วิธีการคำนวณ (Method)": ["Parametric (Delta-Normal)", "Historical Simulation", "Monte Carlo Simulation"],
            "VaR (เปอร์เซ็นต์)": [res['portfolio_var_parametric'], res['portfolio_var_historical'], res['portfolio_var_mc']],
            "VaR (จำนวนเงิน $)": [abs(res['portfolio_var_parametric'] * target_capital), 
                                 abs(res['portfolio_var_historical'] * target_capital), 
                                 abs(res['portfolio_var_mc'] * target_capital)]
        })

        fig_var = px.bar(
            var_data,
            x="วิธีการคำนวณ (Method)",
            y="VaR (จำนวนเงิน $)",
            color="วิธีการคำนวณ (Method)",
            text_auto="$,.2f",
            title=f"การคาดการณ์การขาดทุนสูงสุด (Max Loss) ที่ confidence {confidence:.1%}"
        )
        fig_var.update_traces(hovertemplate="<b>%{x}</b><br>ประเมินผลขาดทุนสูงสุด: $%{y:,.2f}")
        st.plotly_chart(fig_var, use_container_width=True)

    # --- TAB 3: Backtesting & Kupiec Test ---
    with tab3:
        st.markdown("##### 🔍 การทดสอบประสิทธิภาพโมเดลย้อนหลัง (Backtesting & Kupiec POF Test)")
        
        col_bt1, col_bt2 = st.columns([1, 3])
        with col_bt1:
            test_days = st.number_input("จำนวนวันที่ต้องการทดสอบย้อนหลัง (Days)", 30, 500, 100)
            bt_start = st.date_input("วันเริ่มต้นการทดสอบ (Test Start)", datetime(2025, 6, 1)).strftime("%Y-%m-%d")
            run_bt = st.button("🚀 เริ่มการทดสอบ Backtest")

        if run_bt:
            with st.spinner("กำลังทำการ Rolling Backtest และคำนวณ Kupiec Test..."):
                bt_df = time_series_var(
                    result=res,
                    target_capital=target_capital,
                    new_start_date=bt_start,
                    N_test=test_days,
                    confidence=confidence
                )

                # วาดกราฟ Backtesting แบบ Interactive
                fig_bt = go.Figure()

                # เส้น ผลตอบแทนจริง
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['return_port'],
                    mode='lines', name='ผลตอบแทนจริง (Actual Return)',
                    line=dict(color='black', width=1.5)
                ))

                # เส้น VaR แต่ละประเภท
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

                # จุดที่เกิด Breach (ขาดทุนเกิน VaR)
                breaches = bt_df[bt_df['return_port'] < bt_df['var_parametric']]
                fig_bt.add_trace(go.Scatter(
                    x=breaches.index, y=breaches['return_port'],
                    mode='markers', name='จุดที่ทะลุ VaR (Breaches)',
                    marker=dict(color='red', size=8, symbol='x')
                ))

                fig_bt.update_layout(
                    title="ผลตอบแทนจริงเทียบกับเส้นขอบเขต VaR (Breaches Analysis)",
                    xaxis_title="วันที่",
                    yaxis_title="อัตราผลตอบแทน / VaR",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                # แสดงผล Kupiec POF Test
                kupiec_res = kupiec_test(bt_df['var_parametric'], bt_df['return_port'], confidence)
                
                st.markdown("### 📋 ผลการทดสอบทางสถิติ (Kupiec Test Results)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("จำนวนวันที่ทดสอบ", f"{kupiec_res['n_obs']} วัน")
                k2.metric("จำนวนครั้งที่ขาดทุนเกิน VaR", f"{kupiec_res['n_breaches']} ครั้ง")
                k3.metric("อัตราการทะลุจริง / คาดหวัง", f"{kupiec_res['breach_rate']:.2%} / {kupiec_res['expected_rate']:.2%}")
                
                status_color = "green" if kupiec_res['passed'] else "red"
                status_text = "PASSED (โมเดลมีความแม่นยำ)" if kupiec_res['passed'] else "FAILED (โมเดลประเมินความเสี่ยงผิดพลาด)"
                k4.markdown(f"**สถานะโมเดล:**<br><span style='color:{status_color}; font-size: 18px; font-weight:bold;'>{status_text}</span>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
    st.info("💡 คำแนะนำ: โปรดตรวจสอบว่าระบุ ชื่อ Ticker หุ้น/FX ถูกต้อง และตรวจสอบสัดส่วน Weight รวมให้เท่ากับ 1.0")
