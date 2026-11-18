"""
Follow-up to iv_audit.py: how much of the in-the-money gap is American exercise?

US equity and ETF options (including SPY, QQQ, IWM) are American-style. The main
audit uses a European model throughout. This script re-solves IV on a binomial
tree with early exercise, for:
  * every quote the European model could not explain (priced below its floor)
  * a fixed random sample of in-the-money contracts with usable quotes

It reads the local results/iv_audit_contracts.csv written by iv_audit.py and
needs no network access. Exploratory: not part of a pre-registered test.

    python research/iv_american.py --sample 600
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from volevidence import pricing

OUT = Path(__file__).parent / "results"
MAX_SPREAD, MIN_DTE, SEED, STEPS = 0.10, 7, 7, 150


def usable(df: pd.DataFrame) -> pd.Series:
    return (df["spread_pct"] <= MAX_SPREAD) & (df["dte"] >= MIN_DTE)


def in_the_money(df: pd.DataFrame) -> pd.Series:
    return ((df["kind"] == "call") & (df["moneyness"] < 0.97)) | ((df["kind"] == "put") & (df["moneyness"] > 1.03))


def solve_american(row: pd.Series) -> float | None:
    return pricing.american_implied_vol(row["mid"], row["spot"], row["strike"], row["dte"] / 365.0,
                                        row["rate"], row["kind"], row["div_yield"], steps=STEPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()
    df = pd.read_csv(OUT / "iv_audit_contracts.csv")
    ok = usable(df)

    below = df[df["status"] == "below_floor"].copy()
    below["american_iv"] = below.apply(solve_american, axis=1)

    itm = df[(df["status"] == "solved") & in_the_money(df) & ok]
    sample = itm.sample(n=min(args.sample, len(itm)), random_state=SEED).copy()
    sample["american_iv"] = sample.apply(solve_american, axis=1)
    sample = sample[sample["american_iv"].notna()]
    eu_gap = (sample["vendor_iv"] - sample["solved_iv"]).abs() * 100
    am_gap = (sample["vendor_iv"] - sample["american_iv"]).abs() * 100
    premium = (sample["american_iv"] - sample["solved_iv"]) * 100

    usable_solved = df[(df["status"] == "solved") & ok]
    gap_usable = usable_solved["gap_vol_pts"].abs()
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filters": {"max_spread_pct_of_mid": MAX_SPREAD * 100, "min_dte": MIN_DTE, "binomial_steps": STEPS},
        "usable_quotes": {
            "contracts": int(ok.sum()), "share_of_two_sided_pct": round(100 * float(ok.mean()), 1),
            "median_abs_gap_vol_pts": round(float(gap_usable.median()), 3),
            "share_gap_over_5_pts_pct": round(100 * float((gap_usable > 5).mean()), 2),
            "itm_median_abs_gap": round(float(usable_solved[in_the_money(usable_solved)]["gap_vol_pts"]
                                              .abs().median()), 3),
            "otm_atm_median_abs_gap": round(float(usable_solved[~in_the_money(usable_solved)]["gap_vol_pts"]
                                                  .abs().median()), 3),
        },
        "below_european_floor": {
            "contracts": int(len(below)),
            "usable_quotes": int(usable(below).sum()),
            "explained_by_american_exercise_pct": round(100 * float(below["american_iv"].notna().mean()), 1),
            "explained_among_usable_quotes_pct": round(
                100 * float(below[usable(below)]["american_iv"].notna().mean()), 1) if usable(below).any() else None,
        },
        "itm_sample": {
            "contracts": int(len(sample)),
            "median_abs_gap_european_vol_pts": round(float(eu_gap.median()), 3),
            "median_abs_gap_american_vol_pts": round(float(am_gap.median()), 3),
            "median_american_minus_european_vol_pts": round(float(premium.median()), 3),
            "by_kind_median_premium_vol_pts": {k: round(float(premium[sample["kind"] == k].median()), 3)
                                               for k in ("call", "put")},
        },
    }
    (OUT / "iv_american_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
