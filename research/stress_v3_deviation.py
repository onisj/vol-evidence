"""
Deviation from PREREGISTRATION_V3.md, reported separately from the primary result.

Two registered implied-volatility series were unavailable from the registered
source (yfinance `^RVX` and `^VXEEM`), so those markets did not run and the
pre-registered replication rule could not be met. The same Cboe indexes are
published by FRED (`RVXCLS`, `VXEEMCLS`). This script runs those two markets with
FRED volatility data and otherwise identical models, protocol and scoring.

It was written after the primary v3 results were seen. Its results do not change
the pre-registered verdict.

    python research/stress_v3_deviation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stress_v2 import align, fred, run_market, yf_close  # noqa: E402
from stress_v3 import MIN_YEARS, MODELS, market_report  # noqa: E402

OUT = Path(__file__).parent / "results"
MARKETS = {"Russell 2000": ("^RUT", "RVXCLS"), "Emerging markets": ("EEM", "VXEEMCLS")}


def main() -> None:
    results: dict[str, dict] = {}
    for name, (price_t, fred_id) in MARKETS.items():
        price, vol = yf_close(price_t, "1990-01-01"), fred(fred_id)
        common = price.index.intersection(vol.index)
        years = (common.max() - common.min()).days / 365.25 if len(common) else 0.0
        if years < MIN_YEARS:
            results[name] = {"status": "not run", "reason": f"only {years:.1f} years of overlapping data"}
            continue
        f = align(run_market(price.loc[common], vol.loc[common], None, MODELS, common.min().year + 10))
        results[name] = {"sources": [f"yfinance {price_t}", f"FRED {fred_id}"],
                         "data_start": str(common.min().date()), "years_of_data": round(years, 1),
                         **market_report(f)}
    summary = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "status": "deviation from pre-registration; does not change the v3 verdict",
               "markets": results}
    (OUT / "v3_deviation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: (v.get("verdicts") or v) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
