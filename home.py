"""Home page — overview of the Quant Toolkit."""

import streamlit as st

st.title("Quant Toolkit")
st.caption("A quantitative analysis workbench for equity research and portfolio analysis.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Analysis**")
    st.write(
        "Single-security price history, return distribution, risk metrics, "
        "and GARCH(1,1) volatility forecasting with bootstrap simulation."
    )
    st.markdown("Stock Analysis  \nVol Forecast")

with col2:
    st.markdown("**Portfolio**")
    st.write(
        "Mean-variance and risk-parity optimization, Value at Risk, "
        "factor exposure, stress testing, and walk-forward backtesting."
    )
    st.markdown("Portfolio Optimizer  \nRisk Model  \nBacktester")

with col3:
    st.markdown("**Research**")
    st.write(
        "Cross-sectional momentum and trend screening across a custom watchlist "
        "or the S&P 500, plus calendar anomaly testing with multiple-testing correction."
    )
    st.markdown("Screener  \nAnomaly Lab")

st.markdown("---")

st.markdown(
    "**Data:** Yahoo Finance (primary, no API key required) · "
    "Alpha Vantage (fundamentals fallback, 25 requests/day free tier)."
)
st.markdown(
    "_This tool is provided for educational and personal research purposes only. "
    "Nothing on this platform constitutes investment advice, a solicitation, or a "
    "recommendation to buy or sell any security. All analysis is based on historical "
    "data; past performance does not guarantee future results._"
)
