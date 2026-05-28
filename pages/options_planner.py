"""Options Trade Planner — Black-Scholes multi-leg position builder and analyser."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import options as op
from src import data as dt
from src.theme import PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE, apply_chart_theme

# ── Sidebar: market parameters ────────────────────────────────────────────────

st.sidebar.header("Market parameters")

spot_ticker = st.sidebar.text_input(
    "Prefill spot from ticker (optional)",
    value="",
    help="Enter a ticker to load the latest closing price as the spot.",
).upper().strip()


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


default_spot = _latest_close(spot_ticker) if spot_ticker else 100.0

S0 = st.sidebar.number_input(
    "Spot price ($)", min_value=0.01, value=round(default_spot, 2), step=1.0, format="%.2f",
)
r = st.sidebar.number_input(
    "Risk-free rate", min_value=0.0, max_value=0.5, value=0.045, step=0.005, format="%.3f",
    help="Continuously compounded annual rate, e.g. 0.045 for 4.5%.",
)
q = st.sidebar.number_input(
    "Dividend yield", min_value=0.0, max_value=0.5, value=0.0, step=0.005, format="%.3f",
    help="Continuous annual dividend yield, e.g. 0.015 for 1.5%.",
)

# ── Template loader ───────────────────────────────────────────────────────────

TEMPLATE_NAMES = [
    "Custom",
    "Long Call", "Long Put", "Covered Call",
    "Bull Call Spread", "Bear Put Spread",
    "Long Butterfly (Calls)", "Long Straddle", "Long Strangle",
    "Calendar Spread",
]

_LEGS_COLUMNS = {
    "option_type": "Type",
    "direction":   "Direction",
    "strike":      "Strike ($)",
    "dte":         "DTE (days)",
    "iv":          "IV (e.g. 0.30)",
    "quantity":    "Qty (contracts)",
}

_DEFAULT_LEG_ROW = dict(option_type="call", direction="long", strike=round(S0), dte=30, iv=0.30, quantity=1)

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

st.sidebar.markdown("---")
st.sidebar.markdown("**Load template**")
template_name = st.sidebar.selectbox("Template", TEMPLATE_NAMES, index=0)
if st.sidebar.button("Load", disabled=(template_name == "Custom")):
    raw = op.template_legs(template_name, S0)
    if raw:
        st.session_state["legs_df"] = _legs_df_from_dicts(raw)
        st.rerun()

# ── Main: position builder ────────────────────────────────────────────────────

st.title("Options Trade Planner")
st.caption(
    "Build a multi-leg options position using Black-Scholes. "
    "All pricing is theoretical — no live market data."
)

st.subheader("Position legs")
st.markdown(
    "Edit the table below. **Type**: call · put · stock. "
    "**Direction**: long · short. "
    "**IV**: annualized, e.g. 0.30 for 30%. "
    "Stock legs (to model covered calls) ignore Strike, DTE, and IV."
)

edited_df = st.data_editor(
    st.session_state["legs_df"],
    num_rows="dynamic",
    column_config={
        "option_type": st.column_config.SelectboxColumn(
            "Type", options=["call", "put", "stock"], required=True,
        ),
        "direction": st.column_config.SelectboxColumn(
            "Direction", options=["long", "short"], required=True,
        ),
        "strike":   st.column_config.NumberColumn("Strike ($)",      format="%.2f", min_value=0.01),
        "dte":      st.column_config.NumberColumn("DTE (days)",      format="%d",   min_value=0),
        "iv":       st.column_config.NumberColumn("IV (annualised)", format="%.3f", min_value=0.0),
        "quantity": st.column_config.NumberColumn("Qty (contracts)", format="%d",   min_value=1),
    },
    use_container_width=True,
    key="legs_editor",
)
st.session_state["legs_df"] = edited_df

# Convert DataFrame rows to Leg objects
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
    st.info("Add at least one leg to the position table to see analytics.")
    st.stop()

# ── Position summary table ────────────────────────────────────────────────────

st.subheader("Position summary")

summary_rows = []
for i, leg in enumerate(legs):
    res = op.price_leg(leg, S0, S0, r, q, days_elapsed=0)
    direction_sign = 1 if leg.direction == "long" else -1
    debit_credit = (
        f"{'Pay' if direction_sign > 0 else 'Receive'} "
        f"${abs(res.entry_price) * leg.quantity * op.CONTRACT_MULTIPLIER:,.2f}"
    )
    summary_rows.append({
        "Leg":       f"#{i+1} {leg.direction} {leg.option_type}" +
                     (f" K={leg.strike}" if leg.option_type != "stock" else " (100 shares)"),
        "Price/sh":  f"${res.entry_price:.4f}" if leg.option_type != "stock" else f"${S0:.2f}",
        "Premium":   debit_credit,
        "Δ (pos)":   f"{res.pos_delta:+.2f}",
        "Γ (pos)":   f"{res.pos_gamma:+.4f}",
        "Θ/day (pos)": f"{res.pos_theta:+.4f}",
        "Vega (pos)": f"{res.pos_vega:+.4f}",
        "ρ (pos)":   f"{res.pos_rho:+.4f}",
    })

st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# Aggregate Greeks
agg_delta = sum(op.price_leg(l, S0, S0, r, q).pos_delta for l in legs)
agg_gamma = sum(op.price_leg(l, S0, S0, r, q).pos_gamma for l in legs)
agg_theta = sum(op.price_leg(l, S0, S0, r, q).pos_theta for l in legs)
agg_vega  = sum(op.price_leg(l, S0, S0, r, q).pos_vega  for l in legs)
agg_rho   = sum(op.price_leg(l, S0, S0, r, q).pos_rho   for l in legs)

st.markdown("**Aggregate position Greeks**")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Net Δ",       f"{agg_delta:+.3f}")
g2.metric("Net Γ",       f"{agg_gamma:+.4f}")
g3.metric("Net Θ/day",   f"{agg_theta:+.4f}")
g4.metric("Net Vega",    f"{agg_vega:+.4f}")
g5.metric("Net ρ",       f"{agg_rho:+.4f}")

# ── P&L diagram ───────────────────────────────────────────────────────────────

st.subheader("P&L diagram")

max_dte = max((l.dte for l in legs if l.option_type != "stock"), default=365)
days_eval = st.slider(
    "Evaluation date (calendar days from today)",
    min_value=0, max_value=max_dte, value=max_dte, step=1,
    help="0 = today (using BS mid-curve value); max = at expiration of the longest leg.",
)

S_lo = max(S0 * 0.40, 0.01)
S_hi = S0 * 1.80
S_range = np.linspace(S_lo, S_hi, 600)

pnl_now = op.position_pnl(legs, S_range, S0, r, q, days_elapsed=days_eval)
pnl_exp = op.position_pnl(legs, S_range, S0, r, q, days_elapsed=max_dte)

fig_pnl = go.Figure()
fig_pnl.add_trace(go.Scatter(
    x=S_range, y=pnl_exp,
    mode="lines", line=dict(color=NEUTRAL, width=1.5, dash="dash"),
    name=f"At expiry (day {max_dte})",
))
fig_pnl.add_trace(go.Scatter(
    x=S_range, y=pnl_now,
    mode="lines", line=dict(color=PRIMARY, width=2.5),
    name=f"Day {days_eval} P&L",
))
fig_pnl.add_hline(y=0, line_color=REFLINE, line_width=1)
fig_pnl.add_vline(
    x=S0, line_dash="dot", line_color=BENCHMARK, line_width=1.5,
    annotation_text=f"Spot ${S0:,.2f}", annotation_position="top right",
    annotation_font_size=11,
)
fig_pnl.update_layout(
    xaxis_title="Spot price ($)",
    yaxis_title="P&L ($)",
    height=420,
    margin=dict(l=10, r=10, t=20, b=10),
    hovermode="x unified",
)
apply_chart_theme(fig_pnl)
st.plotly_chart(fig_pnl, use_container_width=True)

# ── Key metrics ───────────────────────────────────────────────────────────────

st.subheader("Key metrics")

bes = op.breakevens(legs, S0, r, q)
pop = op.prob_of_profit(legs, S0, r, q)

# Max profit and loss over the S_range at expiry
pnl_exp_arr = op.position_pnl(legs, S_range, S0, r, q, days_elapsed=max_dte)
max_profit_val = float(np.nanmax(pnl_exp_arr))
min_pnl_val    = float(np.nanmin(pnl_exp_arr))

# Check if profit/loss is unbounded at the edges
edge_pnl = [pnl_exp_arr[0], pnl_exp_arr[-1]]
profit_unlimited = max_profit_val >= pnl_exp_arr[-10:].mean() * 0.95 and max_profit_val > 0 and (
    pnl_exp_arr[-1] > pnl_exp_arr[-50] > 0
)
loss_unlimited = min_pnl_val <= pnl_exp_arr[:10].mean() * 0.95 and min_pnl_val < 0 and (
    pnl_exp_arr[0] < pnl_exp_arr[50] < 0
)

max_profit_str = f"${max_profit_val:,.2f}" if not profit_unlimited else "Unlimited"
max_loss_str   = f"${abs(min_pnl_val):,.2f}" if not loss_unlimited else "Unlimited"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Max profit (at expiry)",  max_profit_str)
m2.metric("Max loss (at expiry)",    max_loss_str)
m3.metric("Breakeven(s)",
          ", ".join(f"${b:,.2f}" for b in bes) if bes else "None in range")
m4.metric("Prob. of profit (RN)", f"{pop:.1%}",
          help="Risk-neutral Monte Carlo estimate at the longest expiry. "
               "Understates real-world POP under positive expected returns.")

# ── Implied volatility solver ─────────────────────────────────────────────────

st.subheader("Implied volatility calculator")
st.caption("Back-solve for implied volatility given an observed market price.")

iv_col1, iv_col2, iv_col3, iv_col4 = st.columns(4)
iv_opt   = iv_col1.selectbox("Type", ["call", "put"], key="iv_opt")
iv_mkt   = iv_col1.number_input("Market price ($)", min_value=0.01, value=5.0, step=0.1, format="%.2f", key="iv_mkt")
iv_strike = iv_col2.number_input("Strike ($)", min_value=0.01, value=float(round(S0)), step=1.0, format="%.2f", key="iv_strike")
iv_dte    = iv_col2.number_input("DTE (days)", min_value=1, value=30, step=1, key="iv_dte")
iv_spot   = iv_col3.number_input("Spot ($)", min_value=0.01, value=S0, step=1.0, format="%.2f", key="iv_spot")
iv_r      = iv_col3.number_input("Risk-free rate", min_value=0.0, value=r, step=0.005, format="%.3f", key="iv_r")
iv_q      = iv_col4.number_input("Dividend yield", min_value=0.0, value=q, step=0.005, format="%.3f", key="iv_q")

if st.button("Solve for IV"):
    T_iv = iv_dte / 365.0
    iv_result = op.implied_vol(iv_opt, iv_mkt, iv_spot, iv_strike, T_iv, iv_r, iv_q)
    if iv_result is not None:
        st.success(f"Implied volatility: **{iv_result:.4f}** ({iv_result*100:.2f}%)")
    else:
        st.error("No solution found. Check that the market price is above intrinsic value.")

# ── Caveats ───────────────────────────────────────────────────────────────────

with st.expander("Model assumptions and limitations"):
    st.markdown("""
**Black-Scholes assumptions**

All pricing uses the Black-Scholes-Merton model for European options on a
continuous-dividend-paying underlying. The model assumes lognormally distributed
returns, constant volatility, continuous trading with no transaction costs, and
a constant risk-free rate.

**Deviations from real-world options**

- **American options.** Real equity options are American-style and can be exercised
  early. Early exercise of American calls is generally suboptimal except near
  ex-dividend dates, but American puts should be compared against their Black-Scholes
  prices with caution.
- **Volatility smile.** Actual implied volatility varies by strike and expiry
  (skew and term structure). This tool uses a single constant IV per leg; the
  true premium for far-OTM options is typically higher than BS predicts.
- **Discrete dividends.** The dividend yield input models dividends as a continuous
  yield. Discrete large dividend payments require a different adjustment.
- **Probability of profit.** The POP estimate uses the risk-neutral (Q-measure)
  distribution, which assumes zero drift beyond the risk-free rate. Under a
  positive real-world expected return, POP for long positions is higher than shown.

**Position modelling**

The "stock" leg type represents 100 shares of underlying (delta = 1, all other
Greeks = 0). This is used to model covered call and protective-put payoffs without
further option-pricing approximation.
""")
