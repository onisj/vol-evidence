"""
Study v2: calibrated next-month risk forecasts, walk-forward from 2000.
Implements research/PREREGISTRATION_V2.md exactly.

    python research/stress_v2.py

Writes results/v2_summary.json and charts; raw series stay local (not committed).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from volevidence import riskforecast as rf

OUT = Path(__file__).parent / "results"
FIRST_TEST_YEAR = 2000
UNSEEN = ("2000-01-01", "2007-12-31")


def yf_close(ticker: str, start: str) -> pd.Series:
    import yfinance as yf

    s = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s.dropna()


def fred(series_id: str) -> pd.Series:
    import requests

    resp = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    df.columns = ["date", "value"]
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.to_datetime(df["date"])).dropna()


def run_market(close: pd.Series, vix: pd.Series, credit: pd.Series | None, models: dict[str, list[str]],
               first_test_year: int) -> dict[str, pd.DataFrame]:
    idx = close.index.intersection(vix.index)
    data = rf.features(close.loc[idx], vix.loc[idx], credit).join(rf.targets(close.loc[idx]))
    return {name: rf.walk_forward(data, inputs, first_test_year) for name, inputs in models.items()}


def align(forecasts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common = None
    for f in forecasts.values():
        common = f.index if common is None else common.intersection(f.index)
    return {k: v.loc[common] for k, v in forecasts.items()}


def scores(f: pd.DataFrame) -> dict:
    p, y = f["prob"].to_numpy(), f["dd_event"].to_numpy()
    return {
        "days": int(len(f)),
        "event_rate_pct": round(100 * float(y.mean()), 2),
        "brier": round(float(rf.brier(p, y).mean()), 5),
        "mean_pinball": round(float(rf.mean_pinball(f).mean()), 5),
        "calibration_in_the_large_pp": round(100 * float(p.mean() - y.mean()), 2),
        "interval_10_90_coverage_pct": round(100 * rf.coverage(f), 1),
    }


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Relative improvement of b over a, with block-bootstrap intervals on the loss difference."""
    out = {}
    for name, loss in (("brier", lambda f: rf.brier(f["prob"].to_numpy(), f["dd_event"].to_numpy())),
                       ("pinball", rf.mean_pinball)):
        la, lb = loss(a), loss(b)
        diff = la - lb                                   # positive = b better
        lo, hi = rf.block_bootstrap_mean(diff)
        out[name] = {"improvement_pct": round(100 * float(diff.mean() / la.mean()), 2),
                     "diff_ci95": [round(lo, 6), round(hi, 6)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
                     "better": bool(diff.mean() > 0)}
    return out


def window_report(f: dict[str, pd.DataFrame], start: str | None = None, end: str | None = None) -> dict:
    sub = {k: v.loc[start:end] for k, v in f.items()}
    m0, m1, m3 = sub["M0 climatology"], sub["M1 VIX"], sub["M3 VIX + VRP"]
    m4 = sub.get("M4 composite")
    rep: dict = {"models": {k: scores(v) for k, v in sub.items()}}
    c10 = compare(m0, m1)
    rep["M1_vs_M0"] = c10
    rep["M3_vs_M1"] = compare(m1, m3)
    bss = 1 - rep["models"]["M1 VIX"]["brier"] / rep["models"]["M0 climatology"]["brier"]
    verdicts = {
        "H1_vix_has_skill": bool(bss > 0 and c10["brier"]["better"] and c10["brier"]["ci_excludes_zero"]
                                 and c10["pinball"]["better"] and c10["pinball"]["ci_excludes_zero"]),
        "H3_vrp_adds_to_vix": bool(rep["M3_vs_M1"]["brier"]["better"]
                                   and rep["M3_vs_M1"]["brier"]["improvement_pct"] >= 1
                                   and rep["M3_vs_M1"]["brier"]["ci_excludes_zero"]),
    }
    if m4 is not None:
        c41 = compare(m1, m4)
        rep["M4_vs_M1"] = c41
        rep["M2_vs_M1"] = compare(m1, sub["M2 HAR"])
        verdicts["H2_composite_beats_vix"] = bool(all(
            c41[k]["better"] and c41[k]["improvement_pct"] >= 2 and c41[k]["ci_excludes_zero"]
            for k in ("brier", "pinball")))
        cal = rep["models"]["M1 VIX"]
        verdicts["H4_vix_calibrated"] = bool(abs(cal["calibration_in_the_large_pp"]) <= 2
                                             and 75 <= cal["interval_10_90_coverage_pct"] <= 85)
    rep["brier_skill_vs_climatology"] = {k: round(1 - v["brier"] / rep["models"]["M0 climatology"]["brier"], 4)
                                         for k, v in rep["models"].items()}
    rep["verdicts"] = verdicts
    return rep


def charts(f: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
    for name, color in (("M1 VIX", "#3b6ea5"), ("M4 composite", "#b5523b")):
        rel = rf.reliability(f[name]["prob"].to_numpy(), f[name]["dd_event"].to_numpy())
        x = [r["mean_forecast"] for r in rel if r["days"] >= 50]
        y = [r["observed"] for r in rel if r["days"] >= 50]
        ax.plot(x, y, marker="o", color=color, label=name)
    ax.set_xlabel("Forecast probability of a 5%+ drawdown next month")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability, out of sample 2000 onward\n(bins with at least 50 days)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "v2_reliability.png", dpi=160)

    fig, ax = plt.subplots(figsize=(10, 4))
    years = sorted({d.year for d in f["M0 climatology"].index})
    base = f["M0 climatology"]
    for name, color in (("M1 VIX", "#3b6ea5"), ("M2 HAR", "#6a9a4b"), ("M4 composite", "#b5523b")):
        skill = []
        for yr in years:
            b0 = rf.brier(base.loc[str(yr), "prob"].to_numpy(), base.loc[str(yr), "dd_event"].to_numpy()).mean()
            bm = rf.brier(f[name].loc[str(yr), "prob"].to_numpy(), f[name].loc[str(yr), "dd_event"].to_numpy()).mean()
            skill.append(1 - bm / b0 if b0 > 0 else np.nan)
        ax.plot(years, skill, marker="o", ms=3, color=color, label=name)
    ax.axhline(0, color="#999", lw=1)
    ax.set_ylabel("Brier skill vs climatology")
    ax.set_title("Out-of-sample skill by year (above 0 beats the base rate)")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "v2_skill_by_year.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spx, vix = yf_close("^GSPC", "1990-01-01"), yf_close("^VIX", "1990-01-01")
    credit = fred("BAA10Y")
    us = align(run_market(spx, vix, credit, rf.MODELS, FIRST_TEST_YEAR))
    summary: dict = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "research/PREREGISTRATION_V2.md",
        "us": {"data_start": str(spx.index[0].date()), "oos_start": str(us["M1 VIX"].index[0].date()),
               "oos_end": str(us["M1 VIX"].index[-1].date()),
               "full": window_report(us), "unseen_2000_2007": window_report(us, *UNSEEN),
               "reliability_M1": rf.reliability(us["M1 VIX"]["prob"].to_numpy(), us["M1 VIX"]["dd_event"].to_numpy()),
               "reliability_M4": rf.reliability(us["M4 composite"]["prob"].to_numpy(),
                                                us["M4 composite"]["dd_event"].to_numpy())},
    }
    charts(us)

    try:
        sx5e, v2tx = yf_close("^STOXX50E", "1990-01-01"), yf_close("^V2TX", "1990-01-01")
        common = sx5e.index.intersection(v2tx.index)
        years = (common.max() - common.min()).days / 365.25 if len(common) else 0
        if years >= 10:
            first = common.min().year + 10
            eu_models = {k: v for k, v in rf.MODELS.items() if k != "M4 composite"}
            eu = align(run_market(sx5e, v2tx, None, eu_models, first))
            summary["europe"] = {"data_start": str(common.min().date()), "years_of_data": round(years, 1),
                                 "oos_start": str(eu["M1 VIX"].index[0].date()), **window_report(eu)}
        else:
            summary["europe"] = {"status": "not run", "reason": f"only {years:.1f} years of overlapping data"}
    except Exception as exc:
        summary["europe"] = {"status": "not run", "reason": f"data unavailable: {exc}"}

    (OUT / "v2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"us_full": summary["us"]["full"]["verdicts"],
                      "us_unseen": summary["us"]["unseen_2000_2007"]["verdicts"],
                      "europe": summary["europe"].get("verdicts", summary["europe"])}, indent=2))


if __name__ == "__main__":
    main()
