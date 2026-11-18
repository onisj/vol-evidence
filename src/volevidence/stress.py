"""
Stress gauge: four market inputs ranked against their own past year.

Design and weights are fixed in research/PREREGISTRATION.md. Each input becomes
a 0-100 "calm" score from its percentile rank within the trailing window; the
gauge is their equal-weighted mean. Pure functions over price series, so the
live reading and the historical backtest share one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WINDOW = 252
RV_DAYS = 21
WEIGHTS = {"vix_level": 0.25, "vix_term": 0.25, "credit": 0.25, "realized_vol": 0.25}
CALM_AT, STRESSED_BELOW = 60.0, 40.0
TICKERS = ["^VIX", "^VIX3M", "HYG", "IEF", "SPY"]


def percentile_rank(history: list[float], value: float) -> float:
    """Share of `history` strictly below `value`, on a 0-100 scale (50 if no history)."""
    if not history:
        return 50.0
    return 100.0 * sum(h < value for h in history) / len(history)


def realized_vol(closes: list[float], days: int = RV_DAYS) -> list[float]:
    """Rolling annualised realised volatility of log returns; one value per day from index `days`."""
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    out = []
    for i in range(days, len(rets) + 1):
        window = rets[i - days:i]
        mean = sum(window) / days
        var = sum((x - mean) ** 2 for x in window) / (days - 1)
        out.append(math.sqrt(var * 252))
    return out


@dataclass(frozen=True)
class Reading:
    score: float
    label: str
    components: dict[str, float]


def label(score: float) -> str:
    if score >= CALM_AT:
        return "calm"
    if score >= STRESSED_BELOW:
        return "normal"
    return "stressed"


def _calm(series: list[float], higher_is_calm: bool) -> float:
    pct = percentile_rank(series[:-1], series[-1])
    return pct if higher_is_calm else 100.0 - pct


def read(vix: list[float], vix3m: list[float], hyg: list[float], ief: list[float], spy: list[float]) -> Reading:
    """
    Gauge reading for the last day of the aligned series (oldest first).

    Inputs need at least WINDOW + RV_DAYS + 1 observations so every input
    has a full window; shorter inputs raise ValueError.
    """
    n = min(map(len, (vix, vix3m, hyg, ief, spy)))
    if n < WINDOW + RV_DAYS + 1:
        raise ValueError(f"need {WINDOW + RV_DAYS + 1} aligned observations, got {n}")
    vix, vix3m, hyg, ief, spy = (s[-n:] for s in (vix, vix3m, hyg, ief, spy))

    term = [a / b for a, b in zip(vix, vix3m)]
    credit = [a / b for a, b in zip(hyg, ief)]
    rv = realized_vol(spy)
    comps = {
        "vix_level": _calm(vix[-WINDOW:], higher_is_calm=False),
        "vix_term": _calm(term[-WINDOW:], higher_is_calm=False),
        "credit": _calm(credit[-WINDOW:], higher_is_calm=True),
        "realized_vol": _calm(rv[-WINDOW:], higher_is_calm=False),
    }
    score = sum(WEIGHTS[k] * v for k, v in comps.items())
    return Reading(round(score, 2), label(score), {k: round(v, 2) for k, v in comps.items()})


def vix_baseline(vix: list[float]) -> float:
    """The VIX-percentile input on its own, for comparison."""
    return _calm(vix[-WINDOW:], higher_is_calm=False)
