"""Quant Research Terminal — landing page. Tools are selected from here."""

import streamlit as st

import ui

st.markdown(
    f"""
    <div style="margin: 0.6rem 0 0.5rem 0;
                font-family: var(--font-mono); font-size: 14px; font-weight: 600;
                letter-spacing: 0.18em; color: var(--accent);">{ui.APP_WORDMARK}</div>
    <div style="font-family: var(--font-display); font-size: 44px; font-weight: 550;
                letter-spacing: 0; line-height: 1.1;
                color: var(--text); margin-bottom: 0.6rem;">{ui.APP_NAME}</div>
    <div style="font-size: 15px; color: var(--text-muted); max-width: 760px;
                line-height: 1.6; margin-bottom: 2.4rem;">
        Institutional-style analytics for equity research, portfolio construction,
        risk measurement, and systematic strategy validation. All analytics operate
        on daily historical market data; nothing on this platform is investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Equity Research ───────────────────────────────────────────────────────────

ui.section("Equity & Derivatives Research")
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
    st.html(ui.nav_card(
        "earnings-analysis", "Earnings Move Analysis",
        "Historical earnings-day reactions with the current option-implied "
        "move shown against the realized distribution.",
        ["Realized reaction history", "Implied vs trailing realized move",
         "Distribution of absolute moves"],
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
    st.html(ui.nav_card(
        "options-chain", "Options Chain Explorer",
        "Listed option chains from delayed market data: the implied volatility "
        "smile and term structure, straddle-implied move, and positioning.",
        ["IV smile and term structure", "Implied expected move",
         "Open interest and volume profile", "Quote table with model deltas"],
    ))

# ── Portfolio & Risk ──────────────────────────────────────────────────────────

ui.section("Portfolio & Risk")
c3, c4 = st.columns(2)
with c3:
    st.html(ui.nav_card(
        "portfolio-construction", "Portfolio Construction",
        "Mean-variance and risk-parity allocation with shrinkage estimators, "
        "resampling, and the efficient frontier.",
        ["Maximum Sharpe / minimum variance", "Risk parity", "Ledoit-Wolf and OAS shrinkage",
         "Michaud resampling"],
    ))
    st.html(ui.nav_card(
        "strategy-simulation", "Strategy Simulation",
        "Historical simulation of allocation strategies with transaction costs, "
        "no-lookahead discipline, and estimation/validation segmentation.",
        ["Momentum and trend strategies", "Walk-forward optimization",
         "Estimation vs validation statistics", "Calendar-year returns"],
    ))
with c4:
    st.html(ui.nav_card(
        "risk-analytics", "Risk Analytics",
        "Value at Risk and expected shortfall, forward wealth simulation, "
        "Fama-French factor exposure, and historical stress replay.",
        ["VaR / CVaR in dollars at selectable horizons", "Monte Carlo wealth paths",
         "FF3 factor regression", "Stress scenarios"],
    ))
    st.html(ui.nav_card(
        "correlation-analytics", "Correlation Analytics",
        "Cross-asset correlation structure, the rolling diversification pulse, "
        "and concentration gauges for a custom universe.",
        ["Correlation matrix", "Rolling average pairwise correlation",
         "PC1 variance share", "Diversification ratio"],
    ))

# ── Systematic Research ───────────────────────────────────────────────────────

ui.section("Systematic Research")
c6, c7, c8 = st.columns(3)
with c6:
    st.html(ui.nav_card(
        "equity-screening", "Equity Screening",
        "Cross-sectional momentum and trend-quality ranking across the S&P 500 "
        "or S&P 1500 with GICS sector filters and data-integrity exclusions.",
        ["Composite rank score", "GICS sector filter", "Trend regression diagnostics",
         "Extension filter"],
    ))
with c7:
    st.html(ui.nav_card(
        "seasonality-research", "Seasonality Research",
        "Calendar-effect hypothesis testing with false-discovery-rate control, "
        "out-of-sample replication, and post-cost economics.",
        ["20 calendar hypotheses", "Benjamini-Hochberg FDR", "Out-of-sample replication",
         "Net-of-cost assessment"],
    ))
with c8:
    st.html(ui.nav_card(
        "relative-value", "Relative Value Analysis",
        "Pair cointegration diagnostics: hedge ratio, Engle-Granger test, "
        "spread half-life, and the current z-score.",
        ["OLS hedge ratio", "Engle-Granger cointegration", "Half-life of mean reversion",
         "Spread z-score bands"],
    ))

# ── Macro & Rates ─────────────────────────────────────────────────────────────

ui.section("Macro & Rates")
c9, c10, c11 = st.columns(3)
with c9:
    st.html(ui.nav_card(
        "yield-curve", "Yield Curve Monitor",
        "The Treasury curve from bills to bonds: current shape, history by "
        "tenor, and recession-watch spread inversions.",
        ["Current curve snapshot", "Yield history by tenor",
         "10y minus 13w and 30y minus 5y spreads", "App-wide risk-free default"],
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
