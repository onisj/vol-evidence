"""
Vendor IV vs quote-implied IV across whole option chains.

    python research/iv_audit.py --tickers SPY QQQ IWM AAPL MSFT NVDA KO --max-dte 120

Writes results/iv_audit_contracts.csv, results/iv_audit_summary.json and a chart.
Run during US market hours: outside them many quotes are one-sided or stale.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from volevidence.iv_audit import audit_ticker

OUT = Path(__file__).parent / "results"


def summarize(df: pd.DataFrame) -> dict:
    solved = df[df["status"] == "solved"]
    gap = solved["gap_vol_pts"].abs()
    unsolved = df[df["status"] != "solved"]
    below = unsolved[unsolved["status"] == "below_floor"]
    return {
        "two_sided_contracts": int(len(df)),
        "solved": int(len(solved)),
        "unsolvable_pct": round(100 * len(unsolved) / max(len(df), 1), 2),
        "unsolvable_breakdown": unsolved["status"].value_counts().to_dict(),
        "below_floor_put_share_pct": round(100 * float((below["kind"] == "put").mean()), 1) if len(below) else None,
        "median_abs_gap_vol_pts": round(float(gap.median()), 3),
        "p90_abs_gap_vol_pts": round(float(gap.quantile(0.9)), 3),
        "share_gap_over_2_pts_pct": round(100 * float((gap > 2).mean()), 2),
        "share_gap_over_5_pts_pct": round(100 * float((gap > 5).mean()), 2),
        "median_abs_vendor_price_err_pct": round(float(solved["vendor_price_err_pct"].abs().median()), 3),
        "tight_spread_median_abs_gap": round(float(solved[solved["spread_pct"] < 0.05]["gap_vol_pts"].abs().median()), 3),
        "staleness": staleness(solved),
        "by_ticker_median_abs_gap": {t: round(float(g["gap_vol_pts"].abs().median()), 3)
                                     for t, g in solved.groupby("ticker")},
    }


def staleness(solved: pd.DataFrame) -> dict:
    """Does the vendor IV follow the last trade rather than the current quote?"""
    both = solved[solved["last_price_iv"].notna()]
    vs_quote = (both["vendor_iv"] - both["solved_iv"]).abs() * 100
    vs_last = (both["vendor_iv"] - both["last_price_iv"]).abs() * 100
    age = both["last_trade_age_days"]
    buckets = pd.cut(age, [-1, 1, 5, 10_000], labels=["traded within 1 day", "1-5 days", "over 5 days"])
    by_age = {
        str(b): {"contracts": int(len(g)),
                 "median_abs_gap_vs_quote_iv": round(float(((g["vendor_iv"] - g["solved_iv"]).abs() * 100).median()), 3),
                 "median_abs_gap_vs_last_trade_iv": round(float(((g["vendor_iv"] - g["last_price_iv"]).abs() * 100)
                                                                .median()), 3)}
        for b, g in both.groupby(buckets, observed=True)
    }
    return {
        "contracts_with_last_trade_iv": int(len(both)),
        "median_abs_vendor_gap_vs_quote_iv": round(float(vs_quote.median()), 3),
        "median_abs_vendor_gap_vs_last_trade_iv": round(float(vs_last.median()), 3),
        "vendor_closer_to_last_trade_pct": round(100 * float((vs_last < vs_quote).mean()), 1),
        "by_last_trade_age": by_age,
    }


def chart(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = df[df["status"] == "solved"].copy()
    edges = [0, 0.8, 0.9, 0.97, 1.03, 1.1, 1.2, 10]
    names = ["<0.8", "0.8-0.9", "0.9-0.97", "0.97-1.03", "1.03-1.1", "1.1-1.2", ">1.2"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.4
    for j, (kind, color) in enumerate((("call", "#3b6ea5"), ("put", "#b5523b"))):
        part = solved[solved["kind"] == kind]
        med = part.groupby(pd.cut(part["moneyness"], edges, labels=names), observed=False)["gap_vol_pts"] \
            .apply(lambda s: s.abs().median())
        ax.bar([i + (j - 0.5) * width for i in range(len(names))], med.values, width, label=f"{kind}s", color=color)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlabel("Strike / spot")
    ax.set_ylabel("Median |vendor IV − quote IV| (vol points)")
    ax.set_title("How far vendor IV sits from the IV implied by the quote")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "KO"])
    ap.add_argument("--max-dte", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row for t in args.tickers for row in audit_ticker(t, args.max_dte)])
    df.to_csv(OUT / "iv_audit_contracts.csv", index=False)
    summary = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "tickers": args.tickers, "max_dte": args.max_dte,
               "rate": float(df["rate"].iloc[0]), **summarize(df)}
    (OUT / "iv_audit_summary.json").write_text(json.dumps(summary, indent=2))
    chart(df, OUT / "iv_audit_gap_by_moneyness.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
