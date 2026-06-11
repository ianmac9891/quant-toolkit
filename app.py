"""Quant Research Terminal — entry point. Run with: streamlit run app.py

Navigation is hidden (no sidebar): the home page is the terminal's front door
and every tool page carries a top bar that routes back to it. Pages stay
addressable at stable URL paths so cards and deep links work.
"""

import streamlit as st

import ui

st.set_page_config(
    page_title=ui.APP_NAME,
    layout="wide",
    initial_sidebar_state="collapsed",
)

ui.inject_design_system()
ui.inject_pwa()

pg = st.navigation(
    [
        st.Page("home.py", title="Home", url_path="", default=True),
        # Equity Research
        st.Page("pages/security_analytics.py",   title="Security Analytics",     url_path="security-analytics"),
        st.Page("pages/volatility_analytics.py", title="Volatility Analytics",   url_path="volatility-analytics"),
        st.Page("pages/event_study.py",          title="Event Study",            url_path="event-study"),
        st.Page("pages/derivatives_workbench.py", title="Derivatives Workbench", url_path="derivatives-workbench"),
        st.Page("pages/options_chain.py",          title="Options Chain Explorer", url_path="options-chain"),
        st.Page("pages/earnings_analysis.py",      title="Earnings Move Analysis", url_path="earnings-analysis"),
        # Portfolio & Risk
        st.Page("pages/portfolio_construction.py", title="Portfolio Construction", url_path="portfolio-construction"),
        st.Page("pages/risk_analytics.py",          title="Risk Analytics",         url_path="risk-analytics"),
        st.Page("pages/strategy_simulation.py",     title="Strategy Simulation",    url_path="strategy-simulation"),
        st.Page("pages/correlation_analytics.py",   title="Correlation Analytics",  url_path="correlation-analytics"),
        # Systematic Research
        st.Page("pages/equity_screening.py",      title="Equity Screening",      url_path="equity-screening"),
        st.Page("pages/seasonality_research.py",  title="Seasonality Research",  url_path="seasonality-research"),
        st.Page("pages/relative_value.py",        title="Relative Value Analysis", url_path="relative-value"),
    ],
    position="hidden",
)
pg.run()
