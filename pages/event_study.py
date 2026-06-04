"""Event Study — measure abnormal returns around a user-specified event date."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import event_study as es
from src.theme import PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE, apply_chart_theme

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
benchmark = st.sidebar.text_input("Benchmark", value="SPY").upper().strip()

today = date.today()
event_date_single = st.sidebar.date_input(
    "Event date",
    value=today - timedelta(days=30),
    max_value=today - timedelta(days=1),
    help="The date of the event. If it falls on a weekend or holiday, the next trading day is used.",
)

multi_mode = st.sidebar.checkbox(
    "Multi-event mode",
    value=False,
    help="Paste multiple dates (one per line, YYYY-MM-DD) to aggregate abnormal returns across events.",
)

multi_dates_raw = ""
if multi_mode:
    multi_dates_raw = st.sidebar.text_area(
        "Event dates (one per line, YYYY-MM-DD)",
        height=120,
        help="Each date is processed with its own estimation window. Results are averaged.",
    )

estimation_days = st.sidebar.slider(
    "Estimation window (trading days)", min_value=60, max_value=500, value=250, step=10,
    help="Number of trading days used to estimate the market model parameters.",
)
buffer_days = st.sidebar.slider(
    "Pre-event buffer (trading days)", min_value=5, max_value=60, value=30, step=5,
    help="Gap between the end of the estimation window and the event date, to avoid contamination from pre-event drift.",
)
pre_event = st.sidebar.slider(
    "Event window: days before event", min_value=1, max_value=20, value=5, step=1,
)
post_event = st.sidebar.slider(
    "Event window: days after event", min_value=1, max_value=20, value=5, step=1,
)


# ── Parse dates ───────────────────────────────────────────────────────────────

def _parse_multi_dates(raw: str) -> list[date]:
    dates = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dates.append(date.fromisoformat(line))
        except ValueError:
            st.warning(f"Skipping unrecognised date: '{line}'")
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
        x=event_times, y=ar * 100,
        marker_color=colors,
        text=[f"{v*100:+.2f}%" for v in ar],
        textposition="outside",
        hovertemplate="Day %{x}<br>AR: %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Event time (trading days)",
        yaxis_title="Abnormal return (%)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


def _car_line_chart(
    event_times: np.ndarray, car: np.ndarray, sigma_e: float, title: str
) -> go.Figure:
    days_elapsed = np.arange(1, len(car) + 1)
    ci_upper = car * 100 + 1.96 * sigma_e * np.sqrt(days_elapsed) * 100
    ci_lower = car * 100 - 1.96 * sigma_e * np.sqrt(days_elapsed) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=event_times, y=ci_upper,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=event_times, y=ci_lower,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(79,142,247,0.15)",
        name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=event_times, y=car * 100,
        mode="lines+markers", line=dict(color=PRIMARY, width=2),
        name="CAR",
    ))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Event time (trading days)",
        yaxis_title="Cumulative abnormal return (%)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    apply_chart_theme(fig)
    return fig


def _actual_vs_predicted_chart(
    event_times: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=event_times, y=actual * 100,
        mode="lines+markers", line=dict(color=PRIMARY, width=2),
        name="Actual return",
    ))
    fig.add_trace(go.Scatter(
        x=event_times, y=predicted * 100,
        mode="lines+markers", line=dict(color=BENCHMARK, width=2, dash="dash"),
        name="Model predicted",
    ))
    fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig.update_layout(
        title="Actual vs. market-model predicted return",
        xaxis_title="Event time (trading days)",
        yaxis_title="Return (%)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    apply_chart_theme(fig)
    return fig


# ── Verdict formatting ────────────────────────────────────────────────────────

def _verdict_md(significant: bool, p_value: float) -> str:
    if significant:
        return f"<span style='color:{POSITIVE};font-weight:bold'>Significant at 5% (p = {p_value:.4f})</span>"
    return f"<span style='color:{NEUTRAL}'>Not significant (p = {p_value:.4f})</span>"


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Event Study")

# Collect and validate event dates
if multi_mode:
    event_dates = _parse_multi_dates(multi_dates_raw)
    if not event_dates:
        st.info("Enter one or more event dates in the sidebar to run the study.")
        st.stop()
else:
    event_dates = [event_date_single]

st.markdown(
    f"**{ticker}** vs **{benchmark}** · "
    f"Estimation window: {estimation_days} days · Buffer: {buffer_days} days · "
    f"Event window: [{-pre_event}, +{post_event}]"
)

# ── Run ───────────────────────────────────────────────────────────────────────

if multi_mode and len(event_dates) > 1:
    with st.spinner("Fetching data and running multi-event study…"):
        try:
            result = _run_multi(
                ticker, tuple(event_dates), benchmark,
                estimation_days, buffer_days, pre_event, post_event,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.subheader("Aggregate results")
    n_events = len(result.per_event)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events processed", n_events)
    c2.metric("Mean CAR", f"{result.mean_car[-1]*100:+.2f}%")
    c3.metric("t-statistic", f"{result.t_stat:.3f}" if np.isfinite(result.t_stat) else "—")
    c4.metric("p-value", f"{result.p_value:.4f}" if np.isfinite(result.p_value) else "—")
    st.markdown(
        _verdict_md(result.significant, result.p_value) +
        "  &nbsp;&nbsp;Cross-sectional t-test on the distribution of individual CARs.",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _ar_bar_chart(result.event_times, result.mean_ar, "Mean abnormal return by event day"),
        use_container_width=True, config={"responsive": True, "displayModeBar": False},
    )
    # Use average sigma_e for CI
    avg_sigma = float(np.mean([e.fit.sigma_e for e in result.per_event]))
    st.plotly_chart(
        _car_line_chart(result.event_times, result.mean_car, avg_sigma, "Mean cumulative abnormal return"),
        use_container_width=True, config={"responsive": True, "displayModeBar": False},
    )

    st.subheader("Per-event detail")
    rows = []
    for e in result.per_event:
        rows.append({
            "Event date": str(e.event_date),
            "CAR": f"{e.car_total*100:+.2f}%",
            "t-stat": f"{e.t_stat:.3f}" if np.isfinite(e.t_stat) else "—",
            "p-value": f"{e.p_value:.4f}" if np.isfinite(e.p_value) else "—",
            "Significant": "Yes" if e.significant else "No",
            "Beta": f"{e.fit.beta:.3f}",
            "R²": f"{e.fit.r_squared:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

else:
    # Single event
    ed = event_dates[0]
    with st.spinner(f"Fetching data and running event study for {ticker} on {ed}…"):
        try:
            result = _run_single(
                ticker, ed, benchmark, estimation_days, buffer_days, pre_event, post_event,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    fit = result.fit
    st.subheader("Market model & event-window summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("α (alpha)", f"{fit.alpha*100:.4f}%", help="Daily intercept from the estimation window OLS.")
    c2.metric("β (beta)",  f"{fit.beta:.4f}",        help="Market sensitivity from the estimation window OLS.")
    c3.metric("R²",        f"{fit.r_squared:.4f}",   help="Market model in-sample R-squared.")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("CAR", f"{result.car_total*100:+.2f}%",
              help=f"Cumulative abnormal return over [{-pre_event}, +{post_event}] event window.")
    c5.metric("SE(CAR)", f"{result.se_car*100:.4f}%",
              help="σ_e × √L where L = event window length.")
    c6.metric("t-statistic", f"{result.t_stat:.3f}" if np.isfinite(result.t_stat) else "—")
    c7.metric("p-value", f"{result.p_value:.4f}" if np.isfinite(result.p_value) else "—")

    st.markdown(_verdict_md(result.significant, result.p_value), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            _ar_bar_chart(result.event_times, result.ar, "Abnormal return by event day"),
            use_container_width=True, config={"responsive": True, "displayModeBar": False},
        )
    with col_b:
        st.plotly_chart(
            _car_line_chart(result.event_times, result.car, fit.sigma_e,
                            "Cumulative abnormal return with 95% CI"),
            use_container_width=True, config={"responsive": True, "displayModeBar": False},
        )

    st.plotly_chart(
        _actual_vs_predicted_chart(result.event_times, result.actual_return, result.predicted_return),
        use_container_width=True, config={"responsive": True, "displayModeBar": False},
    )

# ── Caveats ───────────────────────────────────────────────────────────────────

with st.expander("Methodology and caveats"):
    st.markdown("""
**Market model OLS**

The market model regresses the ticker's daily simple return on the benchmark return
over an estimation window ending at least *buffer* days before the event, to avoid
contamination from pre-event drift. The intercept (α) and slope (β) are estimated
by OLS; the residual standard error σ_e captures idiosyncratic daily volatility.

**SE(CAR) and the t-statistic**

Under the standard event-study framework (Brown & Warner 1985), the variance of the
CAR is σ_e² × L where L is the number of days in the event window. This assumes
constant residual variance through the event period.

**Known limitations**

- **Variance inflation at events.** The constant-variance assumption is violated
  when events trigger volatility spikes. The Patell (1976) standardised residual
  test and the Boehmer-Musumeci-Poulsen (1991) test correct for this.
- **Event clustering.** When multiple events occur at the same calendar time
  (e.g., earnings seasons), cross-sectional returns are correlated and the
  standard t-test is no longer valid.
- **Pre-event abnormal returns.** A statistically significant negative (or positive)
  cumulative AR in the days before the event may indicate information leakage or
  anticipation by the market rather than a reaction to the event itself.
- **Non-trading day snap.** If the event date falls on a weekend or holiday, the
  study uses the next available trading day, which may understate the day-0 reaction.
""")
