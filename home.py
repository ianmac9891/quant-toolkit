"""Home page — overview of the Quant Toolkit."""

import streamlit as st

st.title("Quant Toolkit")
st.caption("A quantitative analysis workbench for equity research and portfolio analysis.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Analysis**")
    st.write(
        "Single-equity work: characterise a stock's risk profile, forecast its volatility, "
        "measure how it reacts to events, and plan options trades."
    )
    st.markdown("Stock Analysis  \nVol Forecast  \nEvent Study  \nOptions Trade Planner")

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

st.markdown("---")

st.markdown("**Disclaimer**")
st.caption(
    "This application is provided for educational and personal research purposes only. "
    "The author is not a licensed financial advisor, registered investment advisor, "
    "broker-dealer, or financial planner in any jurisdiction. Nothing on this platform "
    "constitutes investment advice, a solicitation, an offer to buy or sell any security, "
    "or a recommendation of any specific investment strategy.\n\n"
    "All analysis is based on historical data obtained from third-party providers, which "
    "may contain errors, omissions, delays, or inaccuracies. Past performance does not "
    "guarantee future results. All calculations, models, outputs, and visualizations are "
    "illustrative only and should not be relied upon for any trading, investment, tax, or "
    "financial decision.\n\n"
    "The author makes no representations or warranties, express or implied, regarding the "
    "accuracy, completeness, reliability, or fitness for any particular purpose of any "
    "information produced by this application. To the fullest extent permitted by "
    "applicable law, the author disclaims all liability for any direct, indirect, "
    "incidental, consequential, special, or other losses or damages arising from or "
    "related to the use of this application or reliance on its outputs.\n\n"
    "Users should consult a qualified, licensed financial professional before making "
    "any investment decision."
)
