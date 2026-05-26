"""
Portfolio risk model: VaR/CVaR, Monte Carlo simulation, factor exposure, stress tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import analysis

import io
import zipfile

import requests

TRADING_DAYS = 252

# Full calendar year captures the entire rate-hike cycle: the initial Fed hike in
# March 2022 through the final hike of the cycle in December 2022, including the
# associated bond market drawdown and equity repricing.
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008 GFC":         ("2008-09-01", "2009-03-31"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate hikes":  ("2022-01-03", "2022-12-30"),
}


# ── VaR / CVaR ───────────────────────────────────────────────────────────────

@dataclass
class VaRResult:
    hist_var_95:  float
    hist_cvar_95: float
    hist_var_99:  float
    hist_cvar_99: float
    param_var_95: float
    param_var_99: float


def portfolio_var(weights: pd.Series, returns_df: pd.DataFrame) -> VaRResult:
    """
    Historical and parametric VaR/CVaR for a weight vector.
    All values are negative numbers (losses). Reuses analysis.py primitives.
    """
    port_rets = returns_df.reindex(columns=weights.index).fillna(0.0) @ weights.values
    return VaRResult(
        hist_var_95=analysis.historical_var(port_rets, 0.05),
        hist_cvar_95=analysis.historical_cvar(port_rets, 0.05),
        hist_var_99=analysis.historical_var(port_rets, 0.01),
        hist_cvar_99=analysis.historical_cvar(port_rets, 0.01),
        param_var_95=analysis.parametric_var(port_rets, 0.05),
        param_var_99=analysis.parametric_var(port_rets, 0.01),
    )


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo_paths(
    weights: pd.Series,
    returns_df: pd.DataFrame,
    n_paths: int = 5000,
    horizon_days: int = 252,
    random_state: int = 42,
) -> np.ndarray:
    """
    iid bootstrap Monte Carlo: sample daily portfolio returns with replacement.
    Returns array of shape (n_paths, horizon_days + 1); column 0 is always 1.0.
    """
    port_rets = (returns_df.reindex(columns=weights.index).fillna(0.0) @ weights.values)
    arr = port_rets.values

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(arr), size=(n_paths, horizon_days))
    sampled = arr[idx]

    wealth = np.ones((n_paths, horizon_days + 1))
    wealth[:, 1:] = np.cumprod(1.0 + sampled, axis=1)
    return wealth


# ── Fama-French factor regression ─────────────────────────────────────────────

@dataclass
class FactorResult:
    alpha_annual:        float
    alpha_tstat:         float
    betas:               pd.Series   # index: ["Mkt-RF", "SMB", "HML"]
    tstats:              pd.Series
    r_squared:           float
    residual_vol_annual: float
    regression_start:    date
    regression_end:      date
    n_obs:               int


_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def load_ff3_factors(start: date, end: date) -> pd.DataFrame:
    """
    Download Fama-French 3-factor daily data directly from Ken French's library.
    Returns DataFrame[Mkt-RF, SMB, HML, RF] in decimal units, sliced to [start, end].
    Caller is responsible for caching (use @st.cache_data(ttl=86400) in the page).

    Uses direct HTTP download + zipfile parse — avoids pandas_datareader compatibility
    issues on Python 3.12.
    """
    resp = requests.get(_FF3_URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
        raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

    # The CSV has a variable-length header describing the data; data rows start
    # with a date in YYYYMMDD format.  Find the first such line.
    lines = raw_text.splitlines()
    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(",")[0]) == 8
    )
    data_end = next(
        (i for i in range(data_start, len(lines))
         if lines[i].strip() and not lines[i].strip()[0].isdigit()),
        len(lines),
    )

    df = pd.read_csv(
        io.StringIO("\n".join(lines[data_start:data_end])),
        header=None,
        names=["date", "Mkt-RF", "SMB", "HML", "RF"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df / 100.0  # percent → decimal

    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    return df.loc[start_ts:end_ts].copy()


def factor_exposure(
    weights: pd.Series,
    prices_df: pd.DataFrame,
    ff3: pd.DataFrame,
) -> FactorResult:
    """
    OLS regression of portfolio excess returns against FF3 factors.
    ff3 must have columns [Mkt-RF, SMB, HML, RF] in decimal units
    (i.e., output of load_ff3_factors).
    """
    asset_rets = prices_df.reindex(columns=weights.index).pct_change().dropna()
    port_rets = pd.Series(
        (asset_rets @ weights.reindex(asset_rets.columns).fillna(0).values),
        index=asset_rets.index,
    )

    aligned = pd.concat(
        [port_rets.rename("port"), ff3[["Mkt-RF", "SMB", "HML", "RF"]]],
        axis=1, join="inner",
    ).dropna()

    excess = aligned["port"] - aligned["RF"]
    X = sm.add_constant(aligned[["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(excess, X).fit()

    factors = ["Mkt-RF", "SMB", "HML"]
    resid_vol = float(
        model.resid.std(ddof=int(model.df_model) + 1) * np.sqrt(TRADING_DAYS)
    )

    return FactorResult(
        alpha_annual=float(model.params["const"] * TRADING_DAYS),
        alpha_tstat=float(model.tvalues["const"]),
        betas=pd.Series(model.params[factors].values, index=factors),
        tstats=pd.Series(model.tvalues[factors].values, index=factors),
        r_squared=float(model.rsquared),
        residual_vol_annual=resid_vol,
        regression_start=aligned.index[0].date(),
        regression_end=aligned.index[-1].date(),
        n_obs=len(aligned),
    )


# ── Stress tests ──────────────────────────────────────────────────────────────

@dataclass
class StressResult:
    window:       str
    port_return:  float
    equal_return: float
    port_max_dd:  float
    equal_max_dd: float
    covered:      bool   # False if window is outside the prices_df date range


def stress_test(
    weights: pd.Series,
    prices_df: pd.DataFrame,
) -> list[StressResult]:
    """
    Replay portfolio performance across predefined stress windows.
    Uses only tickers present in both weights and prices_df.
    Returns StressResult with covered=False for windows outside the data range.
    """
    common = [t for t in weights.index if t in prices_df.columns]
    if not common:
        return []

    prices = prices_df[common]
    w = weights.reindex(common).fillna(0.0)
    w /= w.sum()
    equal_w = pd.Series(1.0 / len(common), index=common)

    results = []
    for name, (start_str, end_str) in STRESS_WINDOWS.items():
        window = prices.loc[start_str:end_str]
        if len(window) < 5:
            results.append(StressResult(
                window=name, port_return=float("nan"), equal_return=float("nan"),
                port_max_dd=float("nan"), equal_max_dd=float("nan"), covered=False,
            ))
            continue

        daily = window.pct_change().dropna()
        port_rets = pd.Series(daily @ w.values, index=daily.index)
        eq_rets   = pd.Series(daily @ equal_w.values, index=daily.index)

        results.append(StressResult(
            window=name,
            port_return=float((1.0 + port_rets).prod() - 1.0),
            equal_return=float((1.0 + eq_rets).prod() - 1.0),
            port_max_dd=analysis.drawdown(port_rets).max_drawdown,
            equal_max_dd=analysis.drawdown(eq_rets).max_drawdown,
            covered=True,
        ))

    return results
