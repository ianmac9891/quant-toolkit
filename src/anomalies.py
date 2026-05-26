"""
Calendar anomaly lab — tests whether calendar patterns predict green (positive-return) days.

Primary metric  : green-day rate (proportion of trading days with return > 0)
Primary test    : two-proportion z-test, BH-FDR corrected across all 20 hypotheses
Secondary metric: mean return (bps) and post-cost mean return — display only, never gates verdict

Anomaly categories (20 hypotheses total)
-----------------------------------------
Day of week       5  (Mon–Fri each vs the other four)
Month of year    12  (Jan–Dec each vs the other eleven)
Turn of month     1  (last trading day + first 3 of next month vs rest)
New moon          1  (±3-day window vs all other days)
Pre-holiday       1  (trading day before NYSE holiday vs all other days)

Verdict logic
-------------
"Real pattern"  iff  p_fdr < 0.05
                AND  sign(oos_gap) == sign(is_gap)
                AND  |oos_gap| >= 0.5 × |is_gap|   ← OOS must retain half the IS magnitude

Tradability flag (secondary, separate from verdict)
----------------------------------------------------
tradable = True  iff  post_cost_mean_return_bps > 0
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

_LUNAR_EPOCH = date(2000, 1, 6)   # known new moon (verified)
_SYNODIC     = 29.530589           # mean synodic month (days)
_MOON_WINDOW = 3.0                 # ±days around new / full moon for bucketing

_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_MIN_BUCKET_OBS = 10   # minimum observations in a half-sample for IS/OOS estimate


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BucketStats:
    label: str
    n: int
    green_rate: float   # proportion of days with return > 0
    mean_bps: float     # mean daily return in basis points (secondary)
    se_green: float     # sqrt(p*(1-p)/n) — standard error of green_rate


@dataclass
class HypothesisResult:
    category: str
    signal_label: str
    n_signal: int
    n_other: int
    # primary metric
    green_rate_signal: float
    green_rate_other: float
    green_rate_gap: float             # signal - other, in proportion (e.g. 0.03 = 3 pp)
    z_stat: float
    p_raw: float
    p_fdr: float                      = float("nan")   # set after BH pass
    is_green_rate_gap: float          = float("nan")   # gap in first half of sample
    oos_green_rate_gap: float         = float("nan")   # gap in second half
    # secondary metric (display only)
    mean_return_bps: float            = float("nan")
    other_mean_return_bps: float      = float("nan")
    post_cost_mean_return_bps: float  = float("nan")
    # verdict fields
    verdict: str                      = "Noise"
    tradable: bool                    = False


@dataclass
class AnomalyCategory:
    name: str
    description: str
    bucket_stats: list[BucketStats]
    hypotheses: list[HypothesisResult]


# ── Labelers ──────────────────────────────────────────────────────────────────

def label_day_of_week(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([_DOW_NAMES[d.weekday()] for d in dates], index=dates)


def label_month_of_year(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([_MONTH_NAMES[d.month - 1] for d in dates], index=dates)


def label_turn_of_month(dates: pd.DatetimeIndex) -> pd.Series:
    """Last trading day of each month + first 3 trading days of the next."""
    df = pd.DataFrame({"date": dates}, index=dates)
    df["ym"] = df["date"].dt.to_period("M")
    eom  = set(df.groupby("ym")["date"].last())
    som3 = set(df.groupby("ym")["date"].nth([0, 1, 2]))
    tom  = eom | som3
    return pd.Series(
        ["Turn of month" if d in tom else "Rest of month" for d in dates],
        index=dates,
    )


def _lunar_phase_days(d: date) -> float:
    return (d - _LUNAR_EPOCH).days % _SYNODIC


def label_moon_phase(dates: pd.DatetimeIndex) -> pd.Series:
    """New moon ±3d | Full moon ±3d | Other."""
    half = _SYNODIC / 2
    labels = []
    for d in dates.date:
        phase = _lunar_phase_days(d)
        if phase < _MOON_WINDOW or phase > _SYNODIC - _MOON_WINDOW:
            labels.append("New moon ±3d")
        elif abs(phase - half) < _MOON_WINDOW:
            labels.append("Full moon ±3d")
        else:
            labels.append("Other")
    return pd.Series(labels, index=dates)


def label_pre_holiday(dates: pd.DatetimeIndex) -> pd.Series:
    """Trading day immediately before a NYSE holiday."""
    holidays = _get_nyse_holidays(dates.min().year, dates.max().year)
    dates_set = {d.date() for d in dates}

    pre_holiday: set[date] = set()
    for h in holidays:
        d = h - timedelta(days=1)
        for _ in range(7):
            if d in dates_set:
                pre_holiday.add(d)
                break
            d -= timedelta(days=1)

    return pd.Series(
        ["Pre-holiday" if d.date() in pre_holiday else "Other" for d in dates],
        index=dates,
    )


# ── Holiday helpers ───────────────────────────────────────────────────────────

def _get_nyse_holidays(start_year: int, end_year: int) -> set[date]:
    try:
        import pandas_market_calendars as mcal   # type: ignore
        nyse = mcal.get_calendar("NYSE")
        return {pd.Timestamp(h).date() for h in nyse.holidays().holidays}
    except ImportError:
        return _approx_nyse_holidays(start_year, end_year)


def _approx_nyse_holidays(start_year: int, end_year: int) -> set[date]:
    """Rule-based NYSE holidays when pandas_market_calendars is unavailable."""
    result: set[date] = set()
    for y in range(start_year, end_year + 1):
        for m, d in [(1, 1), (7, 4), (12, 25)]:
            result.add(_obs(date(y, m, d)))
        result.add(_nth_weekday(y, 1, 0, 3))   # MLK: 3rd Monday Jan
        result.add(_nth_weekday(y, 2, 0, 3))   # Presidents': 3rd Monday Feb
        result.add(_easter(y) - timedelta(days=2))  # Good Friday
        result.add(_last_weekday(y, 5, 0))     # Memorial Day: last Monday May
        if y >= 2022:
            result.add(_obs(date(y, 6, 19)))   # Juneteenth
        result.add(_nth_weekday(y, 9, 0, 1))   # Labor Day: 1st Monday Sep
        result.add(_nth_weekday(y, 11, 3, 4))  # Thanksgiving: 4th Thursday Nov
    return result


def _obs(d: date) -> date:
    if d.weekday() == 5: return d - timedelta(days=1)
    if d.weekday() == 6: return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = calendar.monthrange(year, month)[1]
    d = date(year, month, last)
    d -= timedelta(days=(d.weekday() - weekday) % 7)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


# ── Core computation ──────────────────────────────────────────────────────────

def _count_runs(mask: pd.Series) -> int:
    """Number of contiguous True-blocks in a boolean Series."""
    if mask.empty or not mask.any():
        return 0
    starts = int((mask.astype(int).diff() == 1).sum())
    return starts + (1 if mask.iloc[0] else 0)


def _gap(a: pd.Series, b: pd.Series) -> float:
    """Green-rate gap (a - b). NaN if either side is too small."""
    if len(a) < _MIN_BUCKET_OBS or len(b) < _MIN_BUCKET_OBS:
        return float("nan")
    return float((a > 0).mean() - (b > 0).mean())


def _bucket_stats(returns: pd.Series, labels: pd.Series) -> list[BucketStats]:
    result = []
    for lbl in labels.unique():
        r = returns[labels == lbl]
        n = len(r)
        if n == 0:
            continue
        gr = float((r > 0).mean())
        result.append(BucketStats(
            label=lbl,
            n=n,
            green_rate=gr,
            mean_bps=float(r.mean() * 10_000),
            se_green=float(np.sqrt(gr * (1.0 - gr) / n)),
        ))
    return result


def _test_signal(
    returns: pd.Series,
    signal_mask: pd.Series,
    cost_bps: float,
) -> dict:
    sig = returns[signal_mask]
    oth = returns[~signal_mask]
    n_s, n_o = len(sig), len(oth)

    n_green_s = int((sig > 0).sum())
    n_green_o = int((oth > 0).sum())
    gr_s = n_green_s / n_s if n_s else float("nan")
    gr_o = n_green_o / n_o if n_o else float("nan")
    gap  = (gr_s - gr_o) if not (np.isnan(gr_s) or np.isnan(gr_o)) else float("nan")

    if n_s >= 5 and n_o >= 5:
        z_stat, p_raw = proportions_ztest(
            [n_green_s, n_green_o], [n_s, n_o], alternative="two-sided"
        )
    else:
        z_stat, p_raw = float("nan"), float("nan")

    mid = len(returns) // 2
    is_gap  = _gap(returns.iloc[:mid][signal_mask.iloc[:mid]],
                   returns.iloc[:mid][~signal_mask.iloc[:mid]])
    oos_gap = _gap(returns.iloc[mid:][signal_mask.iloc[mid:]],
                   returns.iloc[mid:][~signal_mask.iloc[mid:]])

    mean_s = float(sig.mean() * 10_000) if n_s > 0 else float("nan")
    mean_o = float(oth.mean() * 10_000) if n_o > 0 else float("nan")
    n_runs = _count_runs(signal_mask)
    # Round-trip cost per signal day: 1 round-trip per contiguous block
    cost_per_day = (n_runs * 2.0 * cost_bps / n_s) if n_s > 0 else 0.0
    post_cost = (mean_s - cost_per_day) if not np.isnan(mean_s) else float("nan")

    return dict(
        n_signal=n_s, n_other=n_o,
        green_rate_signal=gr_s, green_rate_other=gr_o, green_rate_gap=gap,
        z_stat=float(z_stat), p_raw=float(p_raw),
        is_green_rate_gap=is_gap, oos_green_rate_gap=oos_gap,
        mean_return_bps=mean_s, other_mean_return_bps=mean_o,
        post_cost_mean_return_bps=post_cost,
    )


def _analyze_category(
    returns: pd.Series,
    labels: pd.Series,
    signal_values: list[str],
    cost_bps: float,
    name: str,
    description: str,
    bucket_order: Optional[list[str]] = None,
) -> AnomalyCategory:
    bstats = _bucket_stats(returns, labels)
    if bucket_order:
        order = {v: i for i, v in enumerate(bucket_order)}
        bstats.sort(key=lambda b: order.get(b.label, 999))

    hypotheses: list[HypothesisResult] = []
    for sv in signal_values:
        mask = labels == sv
        if mask.sum() < 5:
            continue
        res = _test_signal(returns, mask, cost_bps)
        hypotheses.append(HypothesisResult(category=name, signal_label=sv, **res))

    return AnomalyCategory(name=name, description=description,
                           bucket_stats=bstats, hypotheses=hypotheses)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_anomaly_lab(
    prices: pd.Series,
    cost_bps: float = 5.0,
) -> list[AnomalyCategory]:
    """
    Run all 5 anomaly categories, apply BH-FDR correction across all ~20
    hypotheses simultaneously, set verdicts, and return annotated categories.
    """
    rets  = prices.pct_change().dropna()
    dates = rets.index

    categories: list[AnomalyCategory] = [
        _analyze_category(
            rets, label_day_of_week(dates), _DOW_NAMES, cost_bps,
            "Day of week",
            "Each day of the week tested against the other four combined.",
            bucket_order=_DOW_NAMES,
        ),
        _analyze_category(
            rets, label_month_of_year(dates), _MONTH_NAMES, cost_bps,
            "Month of year",
            "Each calendar month tested against the other eleven combined.",
            bucket_order=_MONTH_NAMES,
        ),
        _analyze_category(
            rets, label_turn_of_month(dates), ["Turn of month"], cost_bps,
            "Turn of month",
            "Last trading day of each month plus the first three of the next, "
            "tested against the remaining days.",
        ),
        _analyze_category(
            rets, label_moon_phase(dates), ["New moon ±3d"], cost_bps,
            "New moon",
            "Trading days within ±3 days of the new moon, tested against all "
            "other days. Bar chart shows New / Full / Other for context.",
            bucket_order=["New moon ±3d", "Full moon ±3d", "Other"],
        ),
        _analyze_category(
            rets, label_pre_holiday(dates), ["Pre-holiday"], cost_bps,
            "Pre-holiday",
            "The trading day immediately before a NYSE holiday, tested against "
            "all other days.",
        ),
    ]

    # BH FDR correction across all hypotheses in one pass
    all_hyps = [h for cat in categories for h in cat.hypotheses]
    p_vals   = np.array([h.p_raw for h in all_hyps], dtype=float)
    valid    = ~np.isnan(p_vals)

    if valid.any():
        corrected = np.ones(len(p_vals))
        _, corrected[valid], _, _ = multipletests(p_vals[valid], method="fdr_bh")
        for h, p_adj in zip(all_hyps, corrected):
            h.p_fdr = float(p_adj)

    # Verdict logic
    for h in all_hyps:
        is_nan   = np.isnan(h.p_fdr) or np.isnan(h.is_green_rate_gap) or np.isnan(h.oos_green_rate_gap)
        is_real  = (
            not is_nan
            and h.p_fdr < 0.05
            and h.is_green_rate_gap != 0.0
            and h.oos_green_rate_gap * h.is_green_rate_gap > 0       # same sign
            and abs(h.oos_green_rate_gap) >= 0.5 * abs(h.is_green_rate_gap)
        )
        h.verdict  = "Real pattern" if is_real else "Noise"
        h.tradable = (
            not np.isnan(h.post_cost_mean_return_bps)
            and h.post_cost_mean_return_bps > 0
        )

    return categories
