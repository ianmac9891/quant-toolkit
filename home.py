"""Quant Research Terminal — landing page. Tools are selected from here."""

import streamlit as st

import ui

st.markdown(
    f"""
    <div style="margin: 0.5rem 0 0.4rem 0;
                font-family: var(--font-mono); font-size: 13px; font-weight: 600;
                letter-spacing: 0.16em; color: var(--accent);">{ui.APP_WORDMARK}</div>
    <div style="font-size: 30px; font-weight: 650; letter-spacing: -0.015em;
                color: var(--text); margin-bottom: 0.35rem;">{ui.APP_NAME}</div>
    <div style="font-size: 13.5px; color: var(--text-muted); max-width: 720px;
                line-height: 1.55; margin-bottom: 2rem;">
        Institutional-style analytics for equity research, portfolio construction,
        risk measurement, and systematic strategy validation. All analytics operate
        on daily historical market data; nothing on this platform is investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Equity Research ───────────────────────────────────────────────────────────

ui.section("Equity Research")
c1, c2 = st.columns(2)
with c1:
    st.html(ui.nav_card(
        "security-analytics", "Security Analytics",
        "Single-instrument diagnostics: total-return profile, drawdown history, "
        "return distribution, and rolling risk statistics against a benchmark.",
        ["Performance and risk summary", "Drawdown analysis", "Distribution diagnostics",
         "Rolling volatility and Sharpe"],
    ))
    st.html(ui.nav_card(
        "event-study", "Event Study",
        "Market-model abnormal returns around announcement dates, with "
        "Brown-Warner, Patell, and BMP test statistics and multi-event aggregation.",
        ["Cumulative abnormal return", "Patell and BMP inference", "Multi-event aggregation"],
    ))
with c2:
    st.html(ui.nav_card(
        "volatility-analytics", "Volatility Analytics",
        "GARCH and GJR-GARCH conditional volatility forecasting with "
        "bootstrap-simulated price distributions and regime classification.",
        ["Conditional volatility projection", "Simulated price intervals",
         "Reference-level probabilities", "Volatility regime"],
    ))
    st.html(ui.nav_card(
        "derivatives-workbench", "Derivatives Workbench",
        "Multi-leg option position modeling under Black-Scholes-Merton: payoff "
        "profiles, position Greeks, probability of profit, and implied volatility.",
        ["Payoff profile", "Position Greeks", "Probability of profit",
         "Implied volatility solver"],
    ))

# ── Portfolio & Risk ──────────────────────────────────────────────────────────

ui.section("Portfolio & Risk")
c3, c4, c5 = st.columns(3)
with c3:
    st.html(ui.nav_card(
        "portfolio-construction", "Portfolio Construction",
        "Mean-variance and risk-parity allocation with shrinkage estimators, "
        "resampling, and the efficient frontier.",
        ["Maximum Sharpe / minimum variance", "Risk parity", "Ledoit-Wolf and OAS shrinkage",
         "Michaud resampling"],
    ))
with c4:
    st.html(ui.nav_card(
        "risk-analytics", "Risk Analytics",
        "Value at Risk and expected shortfall, forward wealth simulation, "
        "Fama-French factor exposure, and historical stress replay.",
        ["VaR / CVaR at 95% and 99%", "Monte Carlo wealth paths", "FF3 factor regression",
         "Stress scenarios"],
    ))
with c5:
    st.html(ui.nav_card(
        "strategy-simulation", "Strategy Simulation",
        "Historical simulation of allocation strategies with transaction costs, "
        "no-lookahead discipline, and estimation/validation segmentation.",
        ["Momentum and trend strategies", "Walk-forward optimization",
         "Estimation vs validation statistics", "Rebalance ledger"],
    ))

# ── Systematic Research ───────────────────────────────────────────────────────

ui.section("Systematic Research")
c6, c7 = st.columns(2)
with c6:
    st.html(ui.nav_card(
        "equity-screening", "Equity Screening",
        "Cross-sectional momentum and trend-quality ranking across a custom "
        "watchlist or the S&P 500 / S&P 1500, with data-integrity exclusions.",
        ["Composite rank score", "Trend regression diagnostics", "Extension filter",
         "Market-cap gating"],
    ))
with c7:
    st.html(ui.nav_card(
        "seasonality-research", "Seasonality Research",
        "Calendar-effect hypothesis testing with false-discovery-rate control, "
        "out-of-sample replication, and post-cost economics.",
        ["20 calendar hypotheses", "Benjamini-Hochberg FDR", "In/out-of-sample replication",
         "Net-of-cost assessment"],
    ))

# ── Data sources ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="margin-top: 1.6rem; font-size: 12px; color: var(--text-faint);">
    <span style="font-weight:600; letter-spacing:0.07em; text-transform:uppercase;">Data</span>
    &nbsp;&nbsp;Yahoo Finance (primary, daily OHLCV) · Alpha Vantage (fundamentals fallback)
    · Ken French Data Library (factor returns). Prices are cached locally in Parquet.
    </div>
    """,
    unsafe_allow_html=True,
)

ui.footer_disclaimer()
