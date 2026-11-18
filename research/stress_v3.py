"""
Study v3, Part A: replicate v2 in four markets. Implements PREREGISTRATION_V3.md.

    python research/stress_v3.py

Writes results/v3_summary.json and a chart; raw series stay local.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stress_v2 import align, compare, run_market, scores, yf_close  # noqa: E402

from volevidence import riskforecast as rf  # noqa: E402

OUT = Path(__file__).parent / "results"
MARKETS = {
    "Nasdaq-100": ("^NDX", "^VXN"),
    "Russell 2000": ("^RUT", "^RVX"),
    "Emerging markets": ("EEM", "^VXEEM"),
    "Gold": ("GLD", "^GVZ"),
}
MODELS = {
    "M0 climatology": [],
    "M1 implied vol": ["log_vix"],         # the market's own implied-vol index, in the log_vix column
    "M2 HAR": ["log_rv_d", "log_rv_w", "log_rv_m"],
    "M3 implied vol + VRP": ["log_vix", "vrp"],
}
MIN_YEARS = 15


def brier_skill(model: pd.DataFrame, base: pd.DataFrame) -> float:
    b0 = rf.brier(base["prob"].to_numpy(), base["dd_event"].to_numpy()).mean()
    bm = rf.brier(model["prob"].to_numpy(), model["dd_event"].to_numpy()).mean()
    return round(float(1 - bm / b0), 4) if b0 > 0 else float("nan")


def market_report(f: dict[str, pd.DataFrame]) -> dict:
    m0, m1, m2, m3 = (f[k] for k in MODELS)
    c10, c31, c21 = compare(m0, m1), compare(m1, m3), compare(m2, m1)
    yearly = m0.groupby(m0.index.year)["dd_event"].mean()
    worst = int(yearly.idxmax())
    w = str(worst)
    return {
        "oos_start": str(m1.index[0].date()), "oos_end": str(m1.index[-1].date()),
        "models": {k: scores(v) for k, v in f.items()},
        "brier_skill_vs_climatology": {k: brier_skill(v, m0) for k, v in f.items()},
        "M1_vs_M0": c10, "M3_vs_M1": c31, "M1_vs_M2": c21,
        "verdicts": {
            "R1_implied_vol_has_skill": bool(all(c10[k]["better"] and c10[k]["ci_excludes_zero"]
                                                 for k in ("brier", "pinball"))),
            "R2_vrp_adds_nothing": not bool(c31["brier"]["better"] and c31["brier"]["improvement_pct"] >= 1
                                            and c31["brier"]["ci_excludes_zero"]),
            "R3_implied_beats_realised_for_vol_range": bool(c21["pinball"]["better"]
                                                            and c21["pinball"]["ci_excludes_zero"]),
        },
        "R4_worst_year": {"year": worst, "event_rate_pct": round(100 * float(yearly.max()), 1),
                          "brier_skill": {k: brier_skill(v.loc[w], m0.loc[w]) for k, v in f.items() if k != "M0 climatology"}},
    }


def chart(results: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ran = {k: v for k, v in results.items() if "models" in v}
    names = list(ran)
    models = ["M1 implied vol", "M2 HAR", "M3 implied vol + VRP"]
    colors = ["#3b6ea5", "#6a9a4b", "#b5523b"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    for j, (m, c) in enumerate(zip(models, colors)):
        vals = [100 * ran[n]["brier_skill_vs_climatology"][m] for n in names]
        ax.bar(np.arange(len(names)) + (j - 1) * width, vals, width, label=m, color=c)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Brier skill vs base rate (%)")
    ax.set_title("Out-of-sample skill for next-month 5%+ drawdowns, by market")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "v3_skill_by_market.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name, (price_t, vol_t) in MARKETS.items():
        try:
            price, vol = yf_close(price_t, "1990-01-01"), yf_close(vol_t, "1990-01-01")
        except Exception as exc:
            results[name] = {"status": "not run", "reason": f"data unavailable: {exc}"}
            continue
        common = price.index.intersection(vol.index)
        years = (common.max() - common.min()).days / 365.25 if len(common) else 0.0
        if years < MIN_YEARS:
            results[name] = {"status": "not run", "reason": f"only {years:.1f} years of overlapping data"}
            continue
        first = common.min().year + 10
        f = align(run_market(price.loc[common], vol.loc[common], None, MODELS, first))
        results[name] = {"tickers": [price_t, vol_t], "data_start": str(common.min().date()),
                         "years_of_data": round(years, 1), **market_report(f)}

    ran = [r for r in results.values() if "verdicts" in r]
    replicated = sum(r["verdicts"]["R1_implied_vol_has_skill"] and r["verdicts"]["R2_vrp_adds_nothing"] for r in ran)
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "research/PREREGISTRATION_V3.md",
        "markets": results,
        "replication": {"markets_run": len(ran), "markets_where_R1_and_R2_hold": replicated,
                        "replicated": bool(replicated >= 3)},
    }
    (OUT / "v3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    chart(results)
    print(json.dumps({k: (v.get("verdicts") or v) for k, v in results.items()}, indent=2))
    print(json.dumps(summary["replication"]))


if __name__ == "__main__":
    main()
