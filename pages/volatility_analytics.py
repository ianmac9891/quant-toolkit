"""Volatility Analytics — GARCH/GJR-GARCH conditional volatility forecasting."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import data as dt
from src import volforecast as vf
from src.theme import (
    PRIMARY, BENCHMARK, NEUTRAL, NEGATIVE, POSITIVE, TEXT,
    PRIMARY_10, PRIMARY_18, PRIMARY_28, PRIMARY_80,
    REFLINE, CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Equity Research", "Volatility Analytics",
    "Conditional volatility modeling with GARCH(1,1) or GJR-GARCH(1,1,1), "
    "bootstrap-simulated price distributions, and probability estimates for "
    "user-defined reference levels.",
)


def _ts_ms(ts) -> int:
    """Integer milliseconds for Plotly add_vline with date axes."""
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _months(trading_days: int) -> str:
    m = round(trading_days / 21)
    return f"{m} month{'s' if m != 1 else ''}"


# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2, c3, c4 = st.columns([1, 1.3, 1.2, 1.2])
    with c1:
        ticker = st.text_input("Instrument", value="SPY").upper().strip()
    with c2:
        fit_start = st.date_input(
            "Estimation Window Start",
            value=date(today.year - 3, today.month, today.day),
            min_value=date(2000, 1, 1),
            max_value=today - timedelta(days=252),
            help="The model is fit on price history from this date through the latest session.",
        )
    with c3:
        horizon = st.select_slider(
            "Projection Horizon (sessions)",
            options=[21, 42, 63, 126, 189, 252], value=126,
            help="21 sessions is approximately one calendar month.",
        )
    with c4:
        model_type = st.selectbox(
            "Variance Model",
            options=["gjr", "garch"],
            format_func={"gjr": "GJR-GARCH (asymmetric)", "garch": "GARCH (symmetric)"}.get,
            help="GJR-GARCH adds a leverage term — volatility responds more to "
                 "negative returns — and is the standard specification for equities.",
        )

    c5, c6 = st.columns([1.2, 1.2])
    with c5:
        drift_pct = st.number_input(
            "Drift Assumption (% per annum)",
            min_value=-50.0, max_value=100.0, value=0.0, step=0.5, format="%.1f",
            help="Optional directional return assumption. At zero, the projection "
                 "is centered symmetrically on the current price.",
        )
    drift_annual = drift_pct / 100.0
    target_slot = c6.empty()

fit_end = today

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return dt.get_prices(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _run(prices: pd.Series, horizon: int, drift_annual: float,
         model_type: str, n_sim: int = 10_000) -> vf.VolForecast:
    """Fit + bootstrap simulation. Reference level NOT in key — computed live."""
    fit     = vf.fit_garch(prices, model_type=model_type)
    current = float(prices.iloc[-1])
    return vf.simulate_paths(fit, current, horizon, drift_annual, n_sim)


# ── Fetch and fit ─────────────────────────────────────────────────────────────

if not ticker:
    ui.banner("info", "Enter an instrument symbol to begin.")
    st.stop()

with st.spinner(f"Retrieving {ticker}..."):
    try:
        price_df = _fetch(ticker, fit_start, fit_end)
    except Exception as e:
        ui.banner("error", f"Price retrieval failed: {e}")
        st.stop()

if price_df.empty or "adj_close" not in price_df.columns:
    ui.banner("error", f"No price data for <b>{ticker}</b>. Verify the symbol.")
    st.stop()

prices        = price_df["adj_close"].dropna()
current_price = float(prices.iloc[-1])
last_date     = prices.index[-1]
n_fit_days    = len(prices)

target_price = float(
    target_slot.number_input(
        "Reference Level ($)",
        min_value=0.01,
        value=float(round(current_price, 2)),
        step=float(round(max(current_price * 0.01, 0.01), 2)),
        format="%.2f",
        help="The probability that the simulated terminal price exceeds this "
             "level is reported below the projection chart.",
    )
)

with st.spinner("Fitting variance model and simulating paths..."):
    try:
        forecast = _run(prices, horizon, drift_annual, model_type)
    except ValueError as e:
        ui.banner(
            "error",
            f"<b>Model estimation error:</b> {e}<br>"
            "Extending the estimation window may resolve this. Instruments with "
            "limited history may not yield a stationary fit.",
        )
        st.stop()
    except Exception as e:
        ui.banner("error", f"Unexpected error during model estimation: {e}")
        st.stop()

fit = forecast.fit

# ── Live probabilities ────────────────────────────────────────────────────────

p_target  = vf.p_above(forecast, target_price)
p_current = vf.p_above(forecast, current_price)

terminal_end_date = pd.bdate_range(start=last_date, periods=horizon + 1)[-1]
p25_term = float(forecast.p25[-1])
p75_term = float(forecast.p75[-1])

model_label = "GJR-GARCH(1,1,1)" if fit.model == "gjr" else "GARCH(1,1)"

drift_note = (
    "" if drift_annual == 0.0
    else f" (tilted by a {drift_pct:+.1f}% per-annum drift assumption)"
)
ui.banner(
    "info",
    f"Over the next <b>{_months(horizon)}</b>, the simulated distribution places "
    f"<b>{ticker}</b> between <span class='mono'>${p25_term:,.2f}</span> and "
    f"<span class='mono'>${p75_term:,.2f}</span> with 50% probability{drift_note}. "
    f"Probability of finishing above the reference level of "
    f"<span class='mono'>${target_price:,.2f}</span> by {terminal_end_date.date()}: "
    f"<b>{p_target:.0%}</b>. Probability of finishing above the current price: "
    f"<b>{p_current:.0%}</b>.",
)

# ── Projection chart ──────────────────────────────────────────────────────────

with ui.panel(f"Simulated Price Distribution — {model_label}, "
              f"{n_fit_days:,} sessions, horizon {horizon} sessions"):
    fwd_dates = pd.bdate_range(start=last_date, periods=horizon + 1)
    hist_tail = prices.iloc[-63:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        mode="lines", line=dict(color=TEXT, width=2),
        name="Price history", showlegend=True,
    ))

    bands = [
        (forecast.p2_5, forecast.p97_5, "95% interval", PRIMARY_10),
        (forecast.p10,  forecast.p90,   "80% interval", PRIMARY_18),
        (forecast.p25,  forecast.p75,   "50% interval", PRIMARY_28),
    ]
    for lo, hi, name, fill_color in bands:
        fig.add_trace(go.Scatter(x=fwd_dates, y=hi, mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fwd_dates, y=lo, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=fill_color, name=name, showlegend=True))

    fig.add_trace(go.Scatter(
        x=fwd_dates, y=forecast.p50, mode="lines",
        line=dict(color=PRIMARY_80, width=2, dash="dash"),
        name="Median", showlegend=True,
    ))

    fig.add_hline(
        y=current_price, line_dash="dot", line_color=NEUTRAL, line_width=1,
        annotation_text=f"Current ${current_price:,.2f}",
        annotation_position="top left", annotation_font_size=11,
    )
    if abs(target_price - current_price) / current_price > 0.001:
        fig.add_hline(
            y=target_price, line_dash="dot", line_color=BENCHMARK, line_width=1.5,
            annotation_text=f"Reference ${target_price:,.2f}",
            annotation_position="top right", annotation_font_size=11,
        )
    fig.add_vline(x=_ts_ms(last_date), line_dash="solid", line_color=REFLINE, line_width=1)

    all_y = np.concatenate([hist_tail.values, forecast.p2_5, forecast.p97_5])
    fig.update_layout(
        yaxis_title=f"{ticker} price ($)",
        yaxis=dict(range=[float(np.nanmin(all_y)) * 0.97, float(np.nanmax(all_y)) * 1.03]),
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Bands are simulation intervals driven by conditional volatility — not a "
        "price forecast. Paths bootstrap the model's empirical standardized "
        "residuals, preserving fat tails. "
        + ("Centered on zero drift (no directional view)." if drift_annual == 0.0
           else f"Tilted by a {drift_pct:+.1f}% per-annum drift assumption.")
    )

ui.kpi_row([
    {"label": f"P(price > ${target_price:,.2f})", "value": f"{p_target:.1%}"},
    {"label": "P(price > current)", "value": f"{p_current:.1%}"},
    {"label": "50% interval, lower", "value": f"${p25_term:,.2f}"},
    {"label": "50% interval, upper", "value": f"${p75_term:,.2f}"},
])

# ── Volatility context ────────────────────────────────────────────────────────

vol_pct_int = int(round(fit.vol_percentile * 100))
regime_kind = {"elevated": "neg", "normal": "pos", "compressed": "accent"}[fit.vol_regime]
regime_text = {"elevated": "Elevated", "normal": "Normal", "compressed": "Compressed"}[fit.vol_regime]

with ui.panel("Volatility Regime"):
    ui.kpi_row([
        {"label": "Conditional Vol (annualized)", "value": f"{fit.current_ann_vol:.1%}"},
        {"label": "Long-Run Vol", "value": f"{fit.longrun_ann_vol:.1%}"},
        {"label": "Persistence", "value": f"{fit.persistence:.3f}"},
        {"label": "Regime Percentile", "value": f"{vol_pct_int}th"},
    ])
    st.markdown(
        f"Current regime: {ui.tag(regime_text.upper(), regime_kind)}",
        unsafe_allow_html=True,
    )

    cv = fit.cond_vol_series.dropna()
    regime_fig = go.Figure()
    regime_fig.add_trace(go.Scatter(
        x=cv.index, y=cv.values * 100, mode="lines",
        line=dict(color=PRIMARY, width=1.3), name="Conditional volatility",
    ))
    regime_fig.add_hline(
        y=fit.longrun_ann_vol * 100, line_dash="dash", line_color=NEUTRAL, line_width=1.2,
        annotation_text=f"Long-run {fit.longrun_ann_vol:.1%}",
        annotation_position="top left", annotation_font_size=11,
    )
    regime_fig.add_trace(go.Scatter(
        x=[cv.index[-1]], y=[fit.current_ann_vol * 100],
        mode="markers", marker=dict(size=9, color=BENCHMARK, symbol="diamond"),
        name="Current",
    ))
    regime_fig.update_layout(
        yaxis_title="Annualized volatility (%)",
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, hovermode="x unified",
    )
    apply_chart_theme(regime_fig)
    st.plotly_chart(regime_fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "In-sample conditional volatility from the fitted model. The regime "
        "percentile locates the current level within this history."
    )
    if fit.vol_regime == "elevated":
        st.caption(
            "Elevated conditional volatility implies relatively expensive option "
            "premiums: implied volatility tends to track realized volatility, so "
            "option sellers collect more while hedging costs more."
        )
    elif fit.vol_regime == "compressed":
        st.caption(
            "Compressed conditional volatility implies relatively inexpensive "
            "option premiums; low-volatility regimes can reverse abruptly."
        )
    else:
        st.caption("Conditional volatility is within the normal historical range for this instrument.")

# ── Model detail ──────────────────────────────────────────────────────────────

with st.expander("Model Estimates and Variance Term Structure"):
    rows = [
        ("omega", f"{fit.omega:.6f}"),
        ("alpha", f"{fit.alpha:.4f}"),
        ("beta", f"{fit.beta:.4f}"),
    ]
    if fit.model == "gjr":
        rows.append(("gamma (leverage)", f"{fit.gamma:.4f}"))
    rows += [
        ("Persistence", f"{fit.persistence:.4f}"),
        ("Long-run vol (annualized)", f"{fit.longrun_ann_vol:.2%}"),
        ("Conditional vol (annualized)", f"{fit.current_ann_vol:.2%}"),
        ("Log-likelihood", f"{fit.loglikelihood:.1f}"),
        ("AIC", f"{fit.aic:.1f}"),
    ]
    param_df = pd.DataFrame(rows, columns=["Parameter", "Estimate"])
    st.dataframe(param_df, hide_index=True, width="content")

    vol_path     = vf.analytic_vol_path(fit, horizon)
    vol_path_pct = np.concatenate([[fit.current_ann_vol * 100], vol_path * 100])
    vol_days     = np.arange(horizon + 1)

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
        x=vol_days, y=vol_path_pct, mode="lines",
        line=dict(color=PRIMARY, width=2), name="Forecast volatility",
    ))
    vol_fig.add_hline(
        y=fit.longrun_ann_vol * 100, line_dash="dash", line_color=NEUTRAL, line_width=1.5,
        annotation_text=f"Long-run {fit.longrun_ann_vol:.1%}", annotation_position="top right",
    )
    vol_fig.update_layout(
        xaxis_title="Forward sessions", yaxis_title="Annualized volatility (%)",
        height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    apply_chart_theme(vol_fig)
    st.plotly_chart(vol_fig, width="stretch", config=CHART_CONFIG)

    st.caption(
        f"The analytic variance forecast mean-reverts toward the long-run level at a "
        f"rate governed by persistence = {fit.persistence:.3f}. At long horizons the "
        "projection converges to the unconditional volatility. The simulation "
        "resamples empirical standardized residuals rather than normal draws, so "
        "fat tails and asymmetry in the historical returns are preserved."
        + (" The GJR leverage term loads additional variance on negative shocks, "
           "producing asymmetric forward intervals." if fit.model == "gjr" else "")
    )

ui.footer_disclaimer()
