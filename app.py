"""Streamlit entry point — navigation wrapper. Run with: streamlit run app.py"""

import streamlit as st

st.set_page_config(
    page_title="Quant Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("**Quant Toolkit**")
st.sidebar.caption("Quantitative analysis tools")

pg = st.navigation(
    {
        "": [
            st.Page("home.py", title="Home", icon=":material/home:"),
        ],
        "Analysis": [
            st.Page("pages/stock_analysis.py",  title="Stock Analysis",       icon=":material/show_chart:"),
            st.Page("pages/vol_forecast.py",    title="Vol Forecast",         icon=":material/trending_up:"),
            st.Page("pages/event_study.py",     title="Event Study",          icon=":material/query_stats:"),
            st.Page("pages/options_planner.py", title="Options Trade Planner", icon=":material/candlestick_chart:"),
        ],
        "Portfolio": [
            st.Page("pages/portfolio_optimizer.py", title="Portfolio Optimizer", icon=":material/tune:"),
            st.Page("pages/risk_model.py",          title="Risk Model",          icon=":material/shield:"),
            st.Page("pages/backtester.py",          title="Backtester",          icon=":material/replay:"),
        ],
        "Research": [
            st.Page("pages/screener.py",    title="Screener",     icon=":material/search:"),
            st.Page("pages/anomaly_lab.py", title="Anomaly Lab",  icon=":material/science:"),
        ],
    }
)
pg.run()
