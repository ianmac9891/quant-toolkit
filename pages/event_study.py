"""Event Study — market-model abnormal returns with Brown-Warner, Patell, and BMP inference."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import event_study as es
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE,
    PRIMARY_18, CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Equity Research", "Event Study",
    "Abnormal returns around announcement dates under the market model "
    "(Brown-Warner OLS), with Patell standardized-residual and BMP "
    "variance-robust test statistics, and cross-sectional aggregation "
    "across multiple events.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        ticker = st.text_input("Instrument", value="AAPL").upper().strip()
    with c2:
        benchmark = st.text_input("Benchmark", value="SPY").upper().strip()
    with c3:
        event_date_single = st.date_input(
            "Event Date",
            value=today - timedelta(days=30),
            max_value=today - timedelta(days=1),
            help="If the date falls on a non-trading day, the next session is used.",
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        estimation_days = st.slider(
            "Estimation Window (sessions)", min_value=60, max_value=500, value=250, step=10,
            help="Sessions used to estimate the market-model parameters.",
        )
    with c5:
        buffer_days = st.slider(
            "Pre-Event Buffer (sessions)", min_value=5, max_value=60, value=30, step=5,
            help="Gap between the estimation window and the event, to avoid "
                 "contamination from pre-event drift.",
        )
    with c6:
        pre_event = st.slider("Event Window: Sessions Before", 1, 20, 5, step=1)
    with c7:
        post_event = st.slider("Event Window: Sessions After", 1, 20, 5, step=1)

    multi_mode = st.checkbox(
        "Multi-Event Aggregation",
        value=False,
        help="Provide multiple dates to aggregate abnormal returns cross-sectionally.",
    )
    multi_dates_raw = ""
    if multi_mode:
        multi_dates_raw = st.text_area(
            "Event Dates (one per line, YYYY-MM-DD)", height=120,
            help="Each date is processed with its own estimation window; results are aggregated.",
        )


def _parse_multi_dates(raw: str) -> list[date]:
    dates = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dates.append(date.fromisoformat(line))
        except ValueError:
            ui.banner("warn", f"Skipping unrecognised date: <span class='mono'>{line}</span>")
    return sorted(set(dates))


# ── Cached runners ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _run_single(ticker, event_date, benchmark, est_days, buf, pre, post):
    return es.run_single_event(ticker, event_date, benchmark, est_days, buf, pre, post)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_multi(ticker, event_dates_tuple, benchmark, est_days, buf, pre, post):
    return es.run_multi_event(ticker, list(event_dates_tuple), benchmark, est_days, buf, pre, post)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _ar_bar_chart(event_times: np.ndarray, ar: np.ndarray, title: str) -> go.Figure:
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in ar]
    fig = go.Figure(go.Bar(
        x=event_times, y=ar * 100, marker_color=colors,
        text=[f"{v*100:+.2f}%" for v in ar], textposition="outside",
        hovertemplate="Session %{x}<br>AR: %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title=title, xaxis_title="Event time (sessions)",
        yaxis_title="Abnormal return (%)",
        height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


def _car_line_chart(event_times: np.ndarray, car: np.ndarray, sigma_e: float, title: str) -> go.Figure:
    days_elapsed = np.arange(1, len(car) + 1)
    ci_upper = car * 100 + 1.96 * sigma_e * np.sqrt(days_elapsed) * 100
    ci_lower = car * 100 - 1.96 * sigma_e * np.sqrt(days_elapsed) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=event_times, y=ci_upper, mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=event_times, y=ci_lower, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor=PRIMARY_18, name="95% CI"))
    fig.add_trace(go.Scatter(x=event_times, y=car * 100, mode="lines+markers",
                             line=dict(color=PRIMARY, width=2), name="CAR"))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title=title, xaxis_title="Event time (sessions)",
        yaxis_title="Cumulative abnormal return (%)",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
    )
    apply_chart_theme(fig)
    return fig


def _actual_vs_predicted_chart(event_times, actual, predicted) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=event_times, y=actual * 100, mode="lines+markers",
                             line=dict(color=PRIMARY, width=2), name="Realized return"))
    fig.add_trace(go.Scatter(x=event_times, y=predicted * 100, mode="lines+markers",
                             line=dict(color=BENCHMARK, width=2, dash="dash"),
                             name="Market-model prediction"))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title="Realized vs market-model predicted return",
        xaxis_title="Event time (sessions)", yaxis_title="Return (%)",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
    )
    apply_chart_theme(fig)
    return fig


def _sig_tag(p: float) -> str:
    if not np.isfinite(p):
        return ui.tag("N/A", "neu")
    if p < 0.05:
        return ui.tag(f"SIGNIFICANT AT 5% (p = {p:.4f})", "pos")
    return ui.tag(f"NOT SIGNIFICANT (p = {p:.4f})", "neu")


def _fmt_stat(v: float, fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if np.isfinite(v) else "—"


# ── Run ───────────────────────────────────────────────────────────────────────

if not ticker or not benchmark:
    ui.banner("info", "Enter an instrument and a benchmark to begin.")
    st.stop()

if multi_mode:
    event_dates = _parse_multi_dates(multi_dates_raw)
    if not event_dates:
        ui.banner("info", "Enter one or more event dates above to run the study.")
        st.stop()
else:
    event_dates = [event_date_single]

st.caption(
    f"{ticker} vs {benchmark} · Estimation window {estimation_days} sessions · "
    f"Buffer {buffer_days} sessions · Event window [{-pre_event}, +{post_event}]"
)

if multi_mode and len(event_dates) > 1:
    with st.spinner("Running multi-event study..."):
        try:
            result = _run_multi(ticker, tuple(event_dates), benchmark,
                                estimation_days, buffer_days, pre_event, post_event)
        except ValueError as e:
            ui.banner("error", str(e))
            st.stop()
        except Exception as e:
            ui.banner("error", f"Unexpected error: {e}")
            st.stop()

    ui.data_asof_caption(result.data_through)
    n_events = len(result.per_event)
    ui.kpi_row([
        {"label": "Events Processed", "value": f"{n_events}"},
        {"label": "Mean CAR", "value": f"{result.mean_car[-1]*100:+.2f}%"},
        {"label": "Cross-Sectional t", "value": _fmt_stat(result.t_stat)},
        {"label": "Patell Z", "value": _fmt_stat(result.patell_z)},
        {"label": "BMP t", "value": _fmt_stat(result.bmp_t)},
    ])

    with ui.panel("Statistical Significance Assessment"):
        st.markdown(
            f"Cross-sectional t-test on raw CARs: {_sig_tag(result.p_value)}<br>"
            f"Patell standardized-residual Z: {_sig_tag(result.patell_p)}<br>"
            f"BMP variance-robust t: {_sig_tag(result.bmp_p)}",
            unsafe_allow_html=True,
        )
        st.caption(
            "The BMP statistic is robust to event-induced variance inflation and is "
            "the preferred test when events coincide with volatility spikes. "
            "Divergence between the naive and BMP results typically indicates "
            "exactly that condition."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            _ar_bar_chart(result.event_times, result.mean_ar, "Mean abnormal return by session"),
            width="stretch", config=CHART_CONFIG,
        )
    with col_b:
        avg_sigma = float(np.mean([e.fit.sigma_e for e in result.per_event]))
        st.plotly_chart(
            _car_line_chart(result.event_times, result.mean_car, avg_sigma,
                            "Mean cumulative abnormal return"),
            width="stretch", config=CHART_CONFIG,
        )

    with ui.panel("Per-Event Detail"):
        rows = []
        for e in result.per_event:
            rows.append({
                "Event Date":  str(e.event_date),
                "CAR":         f"{e.car_total*100:+.2f}%",
                "t":           _fmt_stat(e.t_stat),
                "Patell Z":    _fmt_stat(e.patell_z),
                "p (naive)":   _fmt_stat(e.p_value, "{:.4f}"),
                "Significant": "Yes" if e.significant else "No",
                "Beta":        f"{e.fit.beta:.3f}",
                "R²":          f"{e.fit.r_squared:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

else:
    ed = event_dates[0]
    with st.spinner(f"Running event study for {ticker} on {ed}..."):
        try:
            result = _run_single(ticker, ed, benchmark, estimation_days,
                                 buffer_days, pre_event, post_event)
        except ValueError as e:
            ui.banner("error", str(e))
            st.stop()
        except Exception as e:
            ui.banner("error", f"Unexpected error: {e}")
            st.stop()

    ui.data_asof_caption(result.data_through)
    fit = result.fit
    ui.kpi_row([
        {"label": "Alpha (daily)", "value": f"{fit.alpha*100:.4f}%"},
        {"label": "Beta", "value": f"{fit.beta:.4f}"},
        {"label": "Model R²", "value": f"{fit.r_squared:.4f}"},
        {"label": "CAR", "value": f"{result.car_total*100:+.2f}%",
         "delta_kind": "pos" if result.car_total >= 0 else "neg"},
        {"label": "SE(CAR)", "value": f"{result.se_car*100:.3f}%"},
    ])

    with ui.panel("Statistical Significance Assessment"):
        st.markdown(
            f"Brown-Warner t = <span class='mono'>{_fmt_stat(result.t_stat)}</span>: "
            f"{_sig_tag(result.p_value)}<br>"
            f"Patell Z = <span class='mono'>{_fmt_stat(result.patell_z)}</span>: "
            f"{_sig_tag(result.patell_p)}",
            unsafe_allow_html=True,
        )
        st.caption(
            "The Patell test scales each abnormal return by its prediction-error-"
            "corrected standard deviation; it remains sensitive to event-induced "
            "variance, which requires a cross-section of events (BMP test, "
            "available in multi-event mode) to address."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            _ar_bar_chart(result.event_times, result.ar, "Abnormal return by session"),
            width="stretch", config=CHART_CONFIG,
        )
    with col_b:
        st.plotly_chart(
            _car_line_chart(result.event_times, result.car, fit.sigma_e,
                            "Cumulative abnormal return with 95% CI"),
            width="stretch", config=CHART_CONFIG,
        )

    st.plotly_chart(
        _actual_vs_predicted_chart(result.event_times, result.actual_return, result.predicted_return),
        width="stretch", config=CHART_CONFIG,
    )

# ── Methodology ───────────────────────────────────────────────────────────────

with st.expander("Methodology and Limitations"):
    st.markdown("""
**Market model OLS.** The instrument's daily return is regressed on the benchmark
return over an estimation window ending at least *buffer* sessions before the
event. The residual standard error captures idiosyncratic daily volatility.

**Test statistics.**
- *Brown-Warner (1985)*: CAR / (sigma_e × sqrt(L)). Assumes constant residual
  variance through the event window.
- *Patell (1976)*: standardizes each abnormal return by its prediction-error-
  corrected standard deviation before aggregating. Corrects for estimation
  error but still assumes no event-induced variance.
- *BMP (1991)*, multi-event only: cross-sectional t-test on standardized CARs.
  Robust to event-induced variance inflation.

**Known limitations.**
- *Event clustering*: when multiple events share calendar time, cross-sectional
  returns are correlated and all three tests overstate significance.
- *Pre-event drift*: significant abnormal returns before the event suggest
  information leakage or anticipation rather than reaction.
- *Non-trading day snap*: events on weekends or holidays are mapped to the next
  session, which can dilute the measured day-zero reaction.
""")

ui.footer_disclaimer()
