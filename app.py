# -*- coding: utf-8 -*-
"""Page 1 — Introduction. Entry point of the Streamlit multipage app."""
from __future__ import annotations

import streamlit as st

from src.state import init_state

# Config หน้าจอหลัก
st.set_page_config(
    page_title="Credit Risk Dashboard — Introduction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_state()

# ==========================================
# HIDE STREAMLIT DEFAULT NAVIGATION
# ==========================================
st.markdown(
    """
    <style>
        /* ซ่อน Default Multi-page Navigation ด้านบน Sidebar */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# CUSTOM SIDEBAR DESIGN
# ==========================================
with st.sidebar:
    st.markdown("## 🛡️ Risk Analytics")
    st.caption("Portfolio Credit Risk Suite v1.0")
    st.divider()

    # หน้าหลัก
    st.page_link("app.py", label="Overview & Introduction", icon="🏠")
    
    st.divider()

    # Section 1: Core Models
    st.caption("MODELING FRAMEWORKS")
    st.page_link("pages/2_Merton_KMV_CVaR.py", label="Merton–KMV CVaR", icon="📈")
    st.page_link("pages/3_CreditMetrics_CVaR.py", label="CreditMetrics CVaR", icon="🎲")
    st.page_link("pages/4_Basel_Single_Factor_CVaR.py", label="Basel Single Factor Credit VaR", icon="🏛️")

    st.divider()

    # Section 2: Analytics & Calibration
    st.caption("ANALYTICS & PARAMETERS")
    st.page_link("pages/5_Model_Comparison.py", label="Model Comparison", icon="⚖️")
    st.page_link("pages/6_Probability_of_Default.py", label="Probability of Default", icon="🎯")

    st.divider()

    # Section 3: Configuration
    st.caption("SYSTEM CONFIGURATION")
    st.page_link("pages/7_Settings.py", label="Settings", icon="⚙️")

    # Bottom Widget: Summary Panel
    st.divider()
    with st.container():
        st.caption("PORTFOLIO STATE")
        st.markdown("**Shared State:** Active ✅")
        st.caption("Changes in portfolio data or settings will sync dynamically across all pages.")


# ==========================================
# MAIN PAGE CONTENT
# ==========================================
st.title("Portfolio Credit Risk Dashboard")
st.caption("Credit Value-at-Risk & Probability of Default, across three independent modeling frameworks")

st.markdown(
    """
This dashboard turns a set of standalone credit-risk models into one
interactive tool: edit a sample loan/bond portfolio once, and see how its
**Credit Value-at-Risk (Credit VaR)** and **Economic Capital** look under
three very different modeling philosophies — plus four separate ways of
estimating a single **Probability of Default (PD)**.

**Objective.** Show, side by side, *why* three textbook-standard Credit VaR
approaches (a structural Monte Carlo model, a rating-migration Monte Carlo
model, and a regulatory closed-form formula) can price the same portfolio
differently — and let you test that with your own numbers, not just the
demo data.
"""
)

st.divider()
st.subheader("How the pages fit together")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Merton–KMV")
    st.markdown(
        "Structural, asset-value Monte Carlo model. A firm defaults when its "
        "simulated asset value falls below its EAD at the risk horizon."
    )
with col2:
    st.markdown("#### CreditMetrics")
    st.markdown(
        "Rating-migration Monte Carlo model. Firms are revalued under every "
        "possible future rating, weighted by simulated migration outcomes."
    )
with col3:
    st.markdown("#### Basel Single-Factor")
    st.markdown(
        "Regulatory Capital Calculation via the Closed-Form Basel II/III ASRF Formula."
    )

col4, col5 = st.columns(2)
with col4:
    st.markdown("#### Model Comparison")
    st.markdown("Runs the same portfolio through all three models above and lines up the results.")
with col5:
    st.markdown("#### Probability of Default")
    st.markdown(
        "Four independent PD estimation methods: Merton's structural model, "
        "Jarrow–Turnbull (flat and term-structure hazard rates), and a "
        "model-free bootstrap from credit spreads."
    )

st.markdown("#### Settings")
st.markdown(
    "The market/model inputs that are hard to eyeball — the rating scale, the "
    "1-year transition matrix, the risk-free curve, and the credit-spread "
    "curve. Change them here and Pages (Merton KMV CVaR, CreditMetrics CVaR, Basel Single-Factor CVaR, Model Comparison and Probability of Default) recompute against the new inputs."
)

st.divider()

st.caption(
    "Built with Streamlit, NumPy, pandas, SciPy and Plotly. "
    "Use the sidebar to navigate between pages — the portfolio you edit on "
    "any of Pages (Merton KMV CVaR, CreditMetrics CVaR, Basel Single-Factor CVaR and Model Comparison) is shared across all of them."
)
