"""Derivatives Workbench — multi-leg option position modeling under Black-Scholes-Merton."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import options as op
from src import data as dt
from src.theme import PRIMARY, BENCHMARK, NEUTRAL, REFLINE, CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Equity Research", "Derivatives Workbench",
    "Construct and analyze multi-leg option positions under the "
    "Black-Scholes-Merton model: payoff profiles, position Greeks, probability "
    "of profit, and implied volatility. All pricing is theoretical; no live "
    "option-market data is used.",
)

# ── Market parameters ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _latest_close(ticker: str) -> float:
    try:
        today = date.today()
        df = dt.get_prices(ticker, today - timedelta(days=10), today)
        if not df.empty and "adj_close" in df.columns:
            return float(df["adj_close"].dropna().iloc[-1])
    except Exception:
        pass
    return 100.0


with ui.panel("Market Parameters"):
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        spot_ticker = st.text_input(
            "Prefill Underlying From Symbol (optional)", value="",
            help="Enter a symbol to load the latest closing price as the underlying level.",
        ).upper().strip()
    default_spot = _latest_close(spot_ticker) if spot_ticker else 100.0
    with c2:
        S0 = st.number_input("Underlying Price ($)", min_value=0.01,
                             value=round(default_spot, 2), step=1.0, format="%.2f")
    with c3:
        r_pct = st.number_input(
            "Risk-Free Rate (% per annum)", min_value=0.0, max_value=20.0,
            value=4.5, step=0.25,
            help="Continuously compounded annualized rate.",
        )
    with c4:
        q_pct = st.number_input(
            "Dividend Yield (% per annum)", min_value=0.0, max_value=20.0,
            value=0.0, step=0.25,
            help="Continuous annualized dividend yield.",
        )
r = r_pct / 100.0
q = q_pct / 100.0

# ── Position builder ──────────────────────────────────────────────────────────

TEMPLATE_NAMES = [
    "Custom",
    "Long Call", "Long Put", "Covered Call",
    "Bull Call Spread", "Bear Put Spread",
    "Long Butterfly (Calls)", "Long Straddle", "Long Strangle",
    "Calendar Spread",
]

_DEFAULT_LEG_ROW = dict(option_type="call", direction="long",
                        strike=round(S0), dte=30, iv=0.30, quantity=1)


def _legs_df_from_dicts(dicts: list[dict]) -> pd.DataFrame:
    if not dicts:
        dicts = [_DEFAULT_LEG_ROW]
    df = pd.DataFrame(dicts)
    df["strike"]   = df["strike"].astype(float)
    df["dte"]      = df["dte"].astype(int)
    df["iv"]       = df["iv"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    return df


if "legs_df" not in st.session_state:
    st.session_state["legs_df"] = _legs_df_from_dicts([_DEFAULT_LEG_ROW])

with ui.panel("Position Structure"):
    tcol1, tcol2 = st.columns([2, 1])
    with tcol1:
        template_name = st.selectbox("Strategy Template", TEMPLATE_NAMES, index=0)
    with tcol2:
        st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
        if st.button("Load Template", disabled=(template_name == "Custom")):
            raw = op.template_legs(template_name, S0)
            if raw:
                st.session_state["legs_df"] = _legs_df_from_dicts(raw)
                st.rerun()

    st.caption(
        "Instrument: call, put, or stock (100 shares; models covered positions). "
        "Side: long or short. Implied volatility is annualized in decimal form "
        "(0.30 = 30%). Stock legs ignore strike, tenor, and volatility."
    )

    edited_df = st.data_editor(
        st.session_state["legs_df"],
        num_rows="dynamic",
        column_config={
            "option_type": st.column_config.SelectboxColumn(
                "Instrument", options=["call", "put", "stock"], required=True),
            "direction": st.column_config.SelectboxColumn(
                "Side", options=["long", "short"], required=True),
            "strike":   st.column_config.NumberColumn("Strike ($)", format="%.2f", min_value=0.01),
            "dte":      st.column_config.NumberColumn("Tenor (calendar days)", format="%d", min_value=0),
            "iv":       st.column_config.NumberColumn("Implied Vol (decimal)", format="%.3f", min_value=0.0),
            "quantity": st.column_config.NumberColumn("Contracts", format="%d", min_value=1),
        },
        width="stretch",
        key="legs_editor",
    )
    st.session_state["legs_df"] = edited_df


def _df_to_legs(df: pd.DataFrame) -> list[op.Leg]:
    legs = []
    for _, row in df.iterrows():
        try:
            legs.append(op.Leg(
                option_type=str(row["option_type"]).lower(),
                direction=str(row["direction"]).lower(),
                strike=float(row["strike"]),
                dte=int(row["dte"]),
                iv=float(row["iv"]),
                quantity=int(row["quantity"]),
            ))
        except (ValueError, KeyError):
            pass
    return legs


legs = _df_to_legs(edited_df)

if not legs:
    ui.banner("info", "Add at least one leg to the position structure to view analytics.")
    st.stop()

option_legs = [l for l in legs if l.option_type != "stock"]
front_dte = op.eval_horizon_dte(legs)
back_dte  = max((l.dte for l in option_legs), default=front_dte)
mixed_expiry = bool(option_legs) and front_dte != back_dte

# ── Position economics summary ────────────────────────────────────────────────

with ui.panel("Leg Valuation and Greeks"):
    summary_rows = []
    for i, leg in enumerate(legs):
        res = op.price_leg(leg, S0, S0, r, q, days_elapsed=0)
        direction_sign = 1 if leg.direction == "long" else -1
        debit_credit = (
            f"{'Pay' if direction_sign > 0 else 'Receive'} "
            f"${abs(res.entry_price) * leg.quantity * op.CONTRACT_MULTIPLIER:,.2f}"
        )
        summary_rows.append({
            "Leg":        f"{i+1}. {leg.direction} {leg.option_type}" +
                          (f" K={leg.strike:g}" if leg.option_type != "stock" else " (100 sh)"),
            "Price/Share": f"${res.entry_price:.4f}" if leg.option_type != "stock" else f"${S0:.2f}",
            "Premium":    debit_credit,
            "Delta":      f"{res.pos_delta:+.2f}",
            "Gamma":      f"{res.pos_gamma:+.4f}",
            "Theta/day":  f"{res.pos_theta:+.4f}",
            "Vega":       f"{res.pos_vega:+.4f}",
            "Rho":        f"{res.pos_rho:+.4f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

leg_results = [op.price_leg(l, S0, S0, r, q) for l in legs]
ui.kpi_row([
    {"label": "Net Delta", "value": f"{sum(lr.pos_delta for lr in leg_results):+.2f}"},
    {"label": "Net Gamma", "value": f"{sum(lr.pos_gamma for lr in leg_results):+.4f}"},
    {"label": "Net Theta / Day", "value": f"{sum(lr.pos_theta for lr in leg_results):+.4f}"},
    {"label": "Net Vega", "value": f"{sum(lr.pos_vega for lr in leg_results):+.4f}"},
    {"label": "Net Rho", "value": f"{sum(lr.pos_rho for lr in leg_results):+.4f}"},
])

# ── Payoff profile ────────────────────────────────────────────────────────────

with ui.panel("Payoff Profile"):
    if mixed_expiry:
        ui.banner(
            "warn",
            f"Mixed expirations detected (front {front_dte}d, back {back_dte}d). "
            "Terminal metrics are evaluated at the <b>front expiration</b> — later-"
            "dated legs are marked at their remaining Black-Scholes value. Valuing "
            "the position beyond the front expiry would require path-dependent "
            "settlement assumptions that a terminal-price model cannot represent.",
        )

    days_eval = st.slider(
        "Valuation Date (calendar days forward)",
        min_value=0, max_value=front_dte, value=front_dte, step=1,
        help="0 = trade date (mid-curve Black-Scholes value); maximum = front expiration.",
    )

    S_lo = max(S0 * 0.40, 0.01)
    S_hi = S0 * 1.80
    S_range = np.linspace(S_lo, S_hi, 600)

    pnl_now = op.position_pnl(legs, S_range, S0, r, q, days_elapsed=days_eval)
    pnl_exp = op.position_pnl(legs, S_range, S0, r, q, days_elapsed=front_dte)

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=S_range, y=pnl_exp, mode="lines",
        line=dict(color=NEUTRAL, width=1.5, dash="dash"),
        name=f"At front expiration (day {front_dte})",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=S_range, y=pnl_now, mode="lines",
        line=dict(color=PRIMARY, width=2.5),
        name=f"Day {days_eval} valuation",
    ))
    fig_pnl.add_hline(y=0, line_color=REFLINE, line_width=1)
    fig_pnl.add_vline(
        x=S0, line_dash="dot", line_color=BENCHMARK, line_width=1.5,
        annotation_text=f"Underlying ${S0:,.2f}", annotation_position="top right",
        annotation_font_size=11,
    )
    fig_pnl.update_layout(
        xaxis_title="Underlying price ($)", yaxis_title="Profit and loss ($)",
        height=420, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
    )
    apply_chart_theme(fig_pnl)
    st.plotly_chart(fig_pnl, width="stretch", config=CHART_CONFIG)

# ── Position economics ────────────────────────────────────────────────────────

default_sim_vol = op.atm_leg_iv(legs, S0)
with ui.panel("Position Economics (at front expiration)"):
    vcol1, vcol2 = st.columns([1.2, 2.8])
    with vcol1:
        sim_vol = st.number_input(
            "Underlying Volatility for Simulation (decimal)",
            min_value=0.01, max_value=5.0, value=float(round(default_sim_vol, 3)),
            step=0.01, format="%.3f",
            help="The underlying has a single diffusion regardless of leg count. "
                 "Defaults to the implied volatility of the leg nearest the money.",
        )

    bounds = op.payoff_bounds(legs, S0, r, q)
    bes = op.breakevens(legs, S0, r, q)
    pop = op.prob_of_profit(legs, S0, r, q, underlying_vol=sim_vol)

    max_profit_str = "Unlimited" if bounds.profit_unbounded else f"${bounds.max_profit:,.2f}"
    max_loss_str   = "Unlimited" if bounds.loss_unbounded else f"${abs(bounds.max_loss):,.2f}"

    ui.kpi_row([
        {"label": "Maximum Profit", "value": max_profit_str},
        {"label": "Maximum Loss", "value": max_loss_str},
        {"label": "Breakeven(s)",
         "value": ", ".join(f"${b:,.2f}" for b in bes) if bes else "None in range"},
        {"label": "Probability of Profit (risk-neutral)", "value": f"{pop:.1%}"},
    ])
    st.caption(
        "Unbounded outcomes are determined analytically from the position's net "
        "share exposure as the underlying rises without limit; the downside is "
        "always bounded by the underlying's zero floor. The probability of profit "
        "is a risk-neutral Monte Carlo estimate (20,000 paths) at the front "
        "expiration; under a positive real-world expected return it understates "
        "the probability for long-delta positions."
    )

# ── Implied volatility solver ─────────────────────────────────────────────────

with ui.panel("Implied Volatility Solver"):
    st.caption("Back-solve the Black-Scholes implied volatility from an observed option price.")
    iv_col1, iv_col2, iv_col3, iv_col4 = st.columns(4)
    iv_opt    = iv_col1.selectbox("Instrument", ["call", "put"], key="iv_opt")
    iv_mkt    = iv_col1.number_input("Observed Price ($)", min_value=0.01, value=5.0,
                                     step=0.1, format="%.2f", key="iv_mkt")
    iv_strike = iv_col2.number_input("Strike ($)", min_value=0.01, value=float(round(S0)),
                                     step=1.0, format="%.2f", key="iv_strike")
    iv_dte    = iv_col2.number_input("Tenor (calendar days)", min_value=1, value=30,
                                     step=1, key="iv_dte")
    iv_spot   = iv_col3.number_input("Underlying ($)", min_value=0.01, value=S0,
                                     step=1.0, format="%.2f", key="iv_spot")
    iv_r_pct  = iv_col3.number_input("Risk-Free Rate (%)", min_value=0.0, value=r_pct,
                                     step=0.25, format="%.2f", key="iv_r")
    iv_q_pct  = iv_col4.number_input("Dividend Yield (%)", min_value=0.0, value=q_pct,
                                     step=0.25, format="%.2f", key="iv_q")

    if st.button("Solve"):
        T_iv = iv_dte / 365.0
        iv_result = op.implied_vol(iv_opt, iv_mkt, iv_spot, iv_strike, T_iv,
                                   iv_r_pct / 100.0, iv_q_pct / 100.0)
        if iv_result is not None:
            ui.banner("success",
                      f"Implied volatility: <b><span class='mono'>{iv_result:.4f}</span></b> "
                      f"({iv_result*100:.2f}% annualized)")
        else:
            ui.banner("error",
                      "No solution. The observed price is below the option's "
                      "arbitrage-free lower bound for any volatility.")

# ── Model assumptions ─────────────────────────────────────────────────────────

with st.expander("Model Assumptions and Limitations"):
    st.markdown("""
**Black-Scholes-Merton.** European exercise, lognormal returns, constant
volatility per leg, continuous trading without transaction costs, constant
rates, continuous dividend yield.

**Deviations from listed options.**
- *American exercise.* Listed equity options permit early exercise; early
  exercise of calls is generally suboptimal except near ex-dividend dates, but
  American puts can carry meaningful early-exercise premium over these values.
- *Volatility surface.* Implied volatility varies by strike and tenor. Each leg
  here carries one constant volatility; far out-of-the-money premiums are
  typically understated.
- *Discrete dividends.* The continuous-yield approximation misprices options
  around large discrete distributions.

**Position modeling.** The stock instrument represents 100 shares (delta one,
all other Greeks zero) and exists to model covered structures. Terminal
metrics are evaluated at the front expiration; positions with mixed
expirations mark later-dated legs at their remaining model value.
""")

ui.footer_disclaimer()
