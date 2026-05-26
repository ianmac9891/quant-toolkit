import traceback

try:
    import streamlit as st
    st.write("App is starting...")
except Exception as e:
    print(traceback.format_exc())
    raise e
"""Streamlit entry point. Run with: streamlit run app.py"""

import streamlit as st

st.set_page_config(
    page_title="Quant Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Quant Toolkit")
st.caption("A quantitative analysis workbench for equity research and portfolio analysis.")

st.markdown(
    """
    ### Tools
    - **Stock Analysis** — Historical price, return distribution, and risk metrics for a single equity.
    - **Portfolio Optimizer** — Mean-variance and risk-parity optimization across a basket of tickers.
    - **Risk Model** — Value at Risk, factor exposures, Monte Carlo simulation, and historical stress tests.
    - **Backtester** — In-sample / out-of-sample strategy backtesting with walk-forward optimization.
    - **Screener** — Momentum and trend ranking across a custom watchlist or the S&P 500 / 1500 universe.
    - **Anomaly Lab** — Calendar anomaly testing with Benjamini-Hochberg multiple-testing correction.
    - **Vol Forecast** — GARCH(1,1) volatility model with bootstrap probability cone and terminal price distribution.

    Use the sidebar to navigate between tools.

    ### Data
    - **Primary source:** Yahoo Finance (no API key required; daily adjusted bars for most listed equities).
    - **Fundamentals fallback:** Alpha Vantage (25 requests/day on the free tier; used sparingly).

    Price data is cached locally in the `cache/` directory. Removing a ticker's cache file forces a fresh download on the next run.

    ---
    *This tool is provided for educational and personal research purposes only. Nothing on this platform constitutes investment advice, a solicitation, or a recommendation to buy or sell any security. All analysis is based on historical data; past performance does not guarantee future results.*
    """
)
