"""Volatility forecast — headline-first, GARCH(1,1) with bootstrap simulation."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data as dt
from src import volforecast as vf
from src.theme import (
    PRIMARY, BENCHMARK, NEUTRAL,
    PRIMARY_10, PRIMARY_18, PRIMARY_28, PRIMARY_80,
    REFLINE,
    apply_chart_theme,
)

st.set_page_config(page_title="Vol Forecast", layout="wide")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_ms(ts) -> int:
    """Integer milliseconds for Plotly add_vline / add_hline with date axes."""
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _months(trading_days: int) -> str:
    m = round(trading_days / 21)
    return f"{m} month{'s' if m != 1 else ''}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()

today = date.today()
fit_start = st.sidebar.date_input(
    "Fit window start",
    value=date(today.year - 3, today.month, today.day),
    min_value=date(2000, 1, 1),
    max_value=today - timedelta(days=252),
    help="GARCH is fit on the price history from this date to today.",
)
fit_end = today

horizon = st.sidebar.slider(
    "Forecast horizon (trading days)",
    min_value=21, max_value=252, value=126, step=21,
    help="21 ≈ 1 month · 63 ≈ 3 months · 126 ≈ 6 months · 252 ≈ 1 year",
)

drift_pct = st.sidebar.number_input(
    "Annual drift assumption (%)",
    min_value=-50.0, max_value=100.0, value=0.0, step=0.5,
    format="%.1f",
    help=(
        "An optional directional return assumption, expressed as an annualized rate. "
        "At zero, the forecast cone is centered symmetrically on the current price. "
        "Non-zero values tilt the distribution in the direction of the assumed drift."
    ),
)
drift_annual = drift_pct / 100.0

target_placeholder = st.sidebar.empty()


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return dt.get_prices(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _run(
    prices: pd.Series,
    horizon: int,
    drift_annual: float,
    n_sim: int = 10_000,
) -> vf.VolForecast:
    """Fit GARCH + bootstrap simulation. target_price NOT in key — computed live."""
    fit     = vf.fit_garch(prices)
    current = float(prices.iloc[-1])
    return vf.simulate_paths(fit, current, horizon, drift_annual, n_sim)


# ── Fetch prices ──────────────────────────────────────────────────────────────

with st.spinner(f"Loading {ticker}…"):
    try:
        price_df = _fetch(ticker, fit_start, fit_end)
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        st.stop()

if price_df.empty or "adj_close" not in price_df.columns:
    st.error(f"No price data for **{ticker}**. Check the ticker symbol.")
    st.stop()

prices       = price_df["adj_close"].dropna()
current_price = float(prices.iloc[-1])
last_date     = prices.index[-1]
n_fit_days    = len(prices)

target_price = float(
    target_placeholder.number_input(
        "Target price ($)",
        min_value=0.01,
        value=float(round(current_price, 2)),
        step=float(round(current_price * 0.01, 2)),
        format="%.2f",
        help="The probability that the simulated terminal price exceeds this level is displayed in the metrics below the chart.",
    )
)

# ── Run forecast ──────────────────────────────────────────────────────────────

with st.spinner("Fitting GARCH model and simulating paths…"):
    try:
        forecast = _run(prices, horizon, drift_annual)
    except ValueError as e:
        st.error(
            f"**Volatility model error:** {e}  \n"
            "Extending the fit window may resolve this. Tickers with limited price "
            "history — such as recent IPOs or spinoffs — may not yield a stationary fit."
        )
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during model fitting: {e}")
        st.stop()

fit = forecast.fit

# ── Live probability (outside cache, O(1)) ────────────────────────────────────

p_target  = vf.p_above(forecast, target_price)
p_current = vf.p_above(forecast, current_price)

terminal_end_date = pd.bdate_range(start=last_date, periods=horizon + 1)[-1]
p25_term  = float(forecast.p25[-1])
p75_term  = float(forecast.p75[-1])

# ── Page header ───────────────────────────────────────────────────────────────

st.title(f"Volatility Forecast: {ticker}")
st.markdown(
    f"GARCH(1,1) fit on **{n_fit_days:,} trading days** "
    f"({fit_start} – {fit_end})  ·  "
    f"Forecast horizon: **{horizon} days ({_months(horizon)})**  ·  "
    f"Current price: **${current_price:,.2f}**"
)

# ── 1. Headline ───────────────────────────────────────────────────────────────

drift_note = (
    ""
    if drift_annual == 0.0
    else f" (tilted by a {drift_pct:+.1f}% annual drift assumption)"
)

st.info(
    f"Over the next **{_months(horizon)}**, {ticker} is most likely to trade between "
    f"**${p25_term:,.2f}** and **${p75_term:,.2f}** — a 50% probability range{drift_note}.  \n"
    f"**{p_target:.0%}** chance it finishes above the target of **${target_price:,.2f}** "
    f"by {terminal_end_date.date()}.  \n"
    f"**{p_current:.0%}** chance it finishes above today's price of **${current_price:,.2f}**."
)

# ── 2. Probability cone chart ─────────────────────────────────────────────────

fwd_dates = pd.bdate_range(start=last_date, periods=horizon + 1)
hist_tail  = prices.iloc[-63:]

fig = go.Figure()

# Historical price (light line, visible on dark background)
fig.add_trace(go.Scatter(
    x=hist_tail.index, y=hist_tail.values,
    mode="lines", line=dict(color="#E5E7EB", width=2),
    name="Price history", showlegend=True,
))

# Forward bands — outer to inner so inner fills on top
bands = [
    (forecast.p2_5,  forecast.p97_5, "95% band", PRIMARY_10),
    (forecast.p10,   forecast.p90,   "80% band", PRIMARY_18),
    (forecast.p25,   forecast.p75,   "50% band", PRIMARY_28),
]
for lo, hi, name, fill_color in bands:
    fig.add_trace(go.Scatter(
        x=fwd_dates, y=hi,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fwd_dates, y=lo,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=fill_color,
        name=name, showlegend=True,
    ))

# Median path
fig.add_trace(go.Scatter(
    x=fwd_dates, y=forecast.p50,
    mode="lines", line=dict(color=PRIMARY_80, width=2, dash="dash"),
    name="Median", showlegend=True,
))

# Current price reference line
fig.add_hline(
    y=current_price, line_dash="dot", line_color=NEUTRAL, line_width=1,
    annotation_text=f"Today ${current_price:,.2f}",
    annotation_position="top left", annotation_font_size=11,
)

# Target price line (only if meaningfully different from current)
if abs(target_price - current_price) / current_price > 0.001:
    fig.add_hline(
        y=target_price, line_dash="dot", line_color=BENCHMARK, line_width=1.5,
        annotation_text=f"Target ${target_price:,.2f}",
        annotation_position="top right", annotation_font_size=11,
    )

# Divider at today
fig.add_vline(
    x=_ts_ms(last_date), line_dash="solid", line_color=REFLINE, line_width=1,
)
fig.add_annotation(
    x=_ts_ms(last_date), y=1.02, xref="x", yref="paper",
    text="Today", showarrow=False, font=dict(size=10, color=NEUTRAL),
)

all_y = np.concatenate([hist_tail.values, forecast.p2_5, forecast.p97_5])
y_lo = float(np.nanmin(all_y)) * 0.97
y_hi = float(np.nanmax(all_y)) * 1.03

fig.update_layout(
    xaxis_title=None,
    yaxis_title=f"{ticker} price ($)",
    yaxis=dict(range=[y_lo, y_hi]),
    height=480,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    hovermode="x unified",
)
apply_chart_theme(fig)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Bands show the **50%, 80%, and 95% simulation intervals** driven by {ticker}'s "
    "volatility — not a price prediction. "
    + (
        "Centered on zero drift (no directional view)."
        if drift_annual == 0.0
        else f"Tilted by a {drift_pct:+.1f}%/year drift assumption."
    )
)

# ── 3. P(> target) ────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
c1.metric(
    label=f"P({ticker} > ${target_price:,.2f} by {terminal_end_date.date()})",
    value=f"{p_target:.1%}",
    help="Computed from 10,000 bootstrap-simulated price paths.",
)
c2.metric(
    label=f"P({ticker} > today's ${current_price:,.2f})",
    value=f"{p_current:.1%}",
    help="Probability the price finishes above the current level by the horizon date.",
)

# ── 4. Volatility context ─────────────────────────────────────────────────────

st.subheader("Volatility context")

vol_pct_int = int(round(fit.vol_percentile * 100))
regime_color = {"elevated": "#dc2626", "normal": "#16a34a", "compressed": "#2563eb"}[fit.vol_regime]
regime_html = (
    f'<span style="color:{regime_color};font-weight:bold">'
    f'{fit.vol_regime.upper()} ({vol_pct_int}th percentile)</span>'
)

c1, c2, c3 = st.columns(3)
c1.metric("Current vol (annualized)", f"{fit.current_ann_vol:.1%}",
          help="Conditional volatility at the last observation, annualized.")
c2.metric("Long-run vol", f"{fit.longrun_ann_vol:.1%}",
          help="The GARCH(1,1) unconditional volatility (omega / (1 − α − β)), annualized.")
c3.metric("Persistence (α + β)", f"{fit.persistence:.3f}",
          help="Fraction of a volatility shock that carries over to the next day. "
               "Closer to 1 = shocks decay slowly.")

st.markdown(
    f"Regime: {regime_html}",
    unsafe_allow_html=True,
)
if fit.vol_regime == "elevated":
    st.caption(
        "Elevated volatility indicates options are relatively expensive — "
        "implied volatility tends to track realized volatility, so option sellers "
        "collect higher premiums while option buyers pay more for protection."
    )
elif fit.vol_regime == "compressed":
    st.caption(
        "Compressed volatility indicates options are relatively inexpensive — "
        "protection costs less in this environment, though low-volatility regimes "
        "can reverse abruptly."
    )
else:
    st.caption(
        "Volatility is within the normal historical range for this ticker."
    )

# ── 5. Model details (collapsed) ─────────────────────────────────────────────

with st.expander("Model details"):
    st.markdown("**GARCH(1,1) parameters**")

    param_df = pd.DataFrame({
        "Parameter": ["ω (omega)", "α (alpha)", "β (beta)", "α + β (persistence)",
                      "Long-run vol", "Current vol", "Log-likelihood", "AIC"],
        "Value": [
            f"{fit.omega:.6f}",
            f"{fit.alpha:.4f}",
            f"{fit.beta:.4f}",
            f"{fit.persistence:.4f}",
            f"{fit.longrun_ann_vol:.2%}",
            f"{fit.current_ann_vol:.2%}",
            f"{fit.loglikelihood:.1f}",
            f"{fit.aic:.1f}",
        ],
    })
    st.dataframe(param_df, hide_index=True, use_container_width=False)

    st.markdown("**Forecast volatility path (analytic)**")
    vol_path     = vf.analytic_vol_path(fit, horizon)
    vol_path_pct = np.concatenate([[fit.current_ann_vol * 100], vol_path * 100])
    vol_days     = np.arange(horizon + 1)

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
        x=vol_days, y=vol_path_pct,
        mode="lines", line=dict(color=PRIMARY, width=2),
        name="Forecast vol",
    ))
    vol_fig.add_hline(
        y=fit.longrun_ann_vol * 100,
        line_dash="dash", line_color=NEUTRAL, line_width=1.5,
        annotation_text=f"Long-run {fit.longrun_ann_vol:.1%}",
        annotation_position="top right",
    )
    vol_fig.update_layout(
        xaxis_title="Forward trading days",
        yaxis_title="Annualized vol (%)",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    apply_chart_theme(vol_fig)
    st.plotly_chart(vol_fig, use_container_width=True)

    st.caption(
        "The GARCH(1,1) analytic variance forecast mean-reverts toward the long-run level "
        f"at a rate set by persistence = {fit.persistence:.3f}. "
        "At long horizons, the forecast converges to the unconditional volatility, so the "
        "outer edges of a multi-year cone essentially reflect the long-run vol rather than "
        "any near-term elevation or compression.  \n"
        "The simulation uses bootstrap resampling of empirical standardized residuals — "
        "not normal draws — so fat tails and asymmetry in historical returns are preserved."
    )
