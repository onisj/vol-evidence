"""
Command line.

    vol-evidence iv SPY --max-dte 60       vendor IV vs quote-implied IV for one ticker
    vol-evidence stress                    today's stress gauge reading
    vol-evidence news NVDA                 cited headline digest (needs an LLM API key)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from volevidence import __version__


def cmd_iv(args: argparse.Namespace) -> int:
    from volevidence.iv_audit import audit_ticker

    rows = audit_ticker(args.ticker, args.max_dte)
    solved = [r for r in rows if r["status"] == "solved"]
    if not rows:
        print(f"{args.ticker}: no two-sided quotes (market closed or no listed options)")
        return 1
    gaps = [abs(r["gap_vol_pts"]) for r in solved]
    print(f"{args.ticker}: {len(rows)} two-sided contracts, {len(solved)} solvable")
    if gaps:
        print(f"  median |vendor IV - quote IV|: {statistics.median(gaps):.2f} vol points")
        print(f"  contracts off by more than 5 points: {100 * sum(g > 5 for g in gaps) / len(gaps):.1f}%")
    worst = sorted(solved, key=lambda r: -abs(r["gap_vol_pts"]))[: args.top]
    for r in worst:
        print(f"  {r['kind']:4s} {r['strike']:>8.2f} {r['expiry']}  vendor {100 * r['vendor_iv']:6.2f}%"
              f"  quote {100 * r['solved_iv']:6.2f}%  gap {r['gap_vol_pts']:+6.2f}")
    return 0


def cmd_stress(_: argparse.Namespace) -> int:
    from volevidence import market_data, stress

    px = market_data.closes(stress.TICKERS, start="2024-01-01")
    need = stress.WINDOW + stress.RV_DAYS + 1
    cols = [list(px[t].to_numpy(dtype=float)[-need:]) for t in ["^VIX", "^VIX3M", "HYG", "IEF", "SPY"]]
    r = stress.read(*cols)
    print(f"Stress gauge {r.score:.1f}/100 ({r.label}) as of {px.index[-1].date()}")
    for name, value in r.components.items():
        print(f"  {name:13s} {value:6.1f}")
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    from volevidence import news

    heads = news.fetch(args.ticker, args.days)
    try:
        result, usage = news.digest(args.ticker, heads, args.model)
    except Exception as exc:
        print(f"digest failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    for i, h in enumerate(heads, start=1):
        print(f"[{i}] {h.source}: {h.title}")
    print(f"checks: {news.citation_check(result, len(heads))}  tokens: {usage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vol-evidence", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("iv", help="vendor IV vs quote-implied IV")
    p.add_argument("ticker")
    p.add_argument("--max-dte", type=int, default=60)
    p.add_argument("--top", type=int, default=5)
    sub.add_parser("stress", help="today's stress gauge")
    n = sub.add_parser("news", help="cited headline digest")
    n.add_argument("ticker")
    n.add_argument("--days", type=int, default=3)
    n.add_argument("--model", default="claude-haiku-5-5")
    args = ap.parse_args(argv)
    return {"iv": cmd_iv, "stress": cmd_stress, "news": cmd_news}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
