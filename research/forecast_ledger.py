"""
Study v3, Part B: a public forward ledger of S&P 500 next-month risk forecasts.

    python research/forecast_ledger.py           append today's forecasts (once per trading day)
    python research/forecast_ledger.py --score   score every forecast whose 21-day window has closed

Rules (PREREGISTRATION_V3.md): forecasts are appended for the latest close only,
never backfilled, and committed on the day they are made. The ledger holds dates,
model names and forecast numbers; no index levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stress_v2 import yf_close  # noqa: E402

from volevidence import riskforecast as rf  # noqa: E402

OUT = Path(__file__).parent / "results"
LEDGER = OUT / "forecast_ledger.csv"
MODELS = {k: v for k, v in rf.MODELS.items() if k != "M4 composite"}
MIN_RESOLVED = 120


def load_data() -> pd.DataFrame:
    spx, vix = yf_close("^GSPC", "1990-01-01"), yf_close("^VIX", "1990-01-01")
    idx = spx.index.intersection(vix.index)
    return rf.features(spx.loc[idx], vix.loc[idx]).join(rf.targets(spx.loc[idx]))


def append_today() -> None:
    data = load_data()
    today = data.index[-1]
    ledger = pd.read_csv(LEDGER, parse_dates=["date"]) if LEDGER.exists() else pd.DataFrame()
    if not ledger.empty and ledger["date"].max() >= today:
        print(f"ledger already has forecasts for {ledger['date'].max().date()}; nothing to add for {today.date()}")
        return
    train = data.dropna(subset=["dd_event", "log_fwd_rv"])          # only rows whose window has closed
    test = data.loc[[today]]
    rows = []
    for name, inputs in MODELS.items():
        fit_rows = train.dropna(subset=inputs) if inputs else train
        fc = rf.fit_predict(fit_rows, test, inputs)
        q = np.exp(fc.quantiles[0]) * 100                            # annualised volatility, percent
        rows.append({"date": today.date(), "model": name, "prob_dd5": round(float(fc.prob[0]), 4),
                     "vol_q10_pct": round(float(q[0]), 2), "vol_q50_pct": round(float(q[1]), 2),
                     "vol_q90_pct": round(float(q[2]), 2), "train_rows": len(fit_rows)})
    new = pd.DataFrame(rows)
    out = pd.concat([ledger, new], ignore_index=True) if not ledger.empty else new
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEDGER, index=False)
    print(new.to_string(index=False))


def score() -> None:
    ledger = pd.read_csv(LEDGER, parse_dates=["date"])
    data = load_data()
    merged = ledger.merge(data[["dd_event", "log_fwd_rv"]], left_on="date", right_index=True, how="left")
    resolved = merged.dropna(subset=["dd_event", "log_fwd_rv"])
    days = resolved["date"].nunique()
    report: dict = {"forecast_days": int(ledger["date"].nunique()), "resolved_days": int(days),
                    "min_resolved_for_verdict": MIN_RESOLVED}
    if days < MIN_RESOLVED:
        report["status"] = "too few resolved forecasts for a verdict"
        print(json.dumps(report, indent=2))
        return
    for name, g in resolved.groupby("model"):
        frame = pd.DataFrame({"prob": g["prob_dd5"].to_numpy(), "dd_event": g["dd_event"].to_numpy(),
                              "log_fwd_rv": g["log_fwd_rv"].to_numpy(),
                              "q10": np.log(g["vol_q10_pct"].to_numpy() / 100),
                              "q50": np.log(g["vol_q50_pct"].to_numpy() / 100),
                              "q90": np.log(g["vol_q90_pct"].to_numpy() / 100)})
        report[name] = {"brier": round(float(rf.brier(frame["prob"].to_numpy(), frame["dd_event"].to_numpy()).mean()), 5),
                        "mean_pinball": round(float(rf.mean_pinball(frame).mean()), 5),
                        "coverage_10_90_pct": round(100 * rf.coverage(frame), 1)}
    (OUT / "forecast_ledger_score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    score() if args.score else append_today()


if __name__ == "__main__":
    main()
