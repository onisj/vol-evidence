"""
Compare a vendor's implied volatility with the IV implied by each quote.

For every contract with a two-sided market, solve Black-Scholes-Merton IV from
the bid/ask mid and compare it with the vendor's `impliedVolatility`. Contracts
whose mid lies outside the no-arbitrage bounds are classified, not solved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

from volevidence import market_data, pricing


@dataclass(frozen=True)
class Row:
    ticker: str
    expiry: str
    dte: int
    kind: str
    strike: float
    spot: float
    moneyness: float
    mid: float
    spread_pct: float
    vendor_iv: float
    solved_iv: float | None
    gap_vol_pts: float | None
    vendor_price_err_pct: float | None
    status: str          # solved | below_floor | above_cap | no_solution
    last_trade_age_days: float | None = None   # since the contract last traded, at audit time
    last_price_iv: float | None = None         # IV implied by the last trade price instead of the quote


def audit_quotes(ticker: str, spot: float, quotes: list[market_data.Quote], r: float, q: float,
                 now: datetime | None = None) -> list[Row]:
    now = now or datetime.now(timezone.utc)
    rows = []
    for c in quotes:
        mid = c.mid
        if mid is None or c.vendor_iv is None or c.vendor_iv <= 0 or c.bid is None or c.ask is None:
            continue
        T = c.dte / 365.0
        solved = pricing.implied_vol(mid, spot, c.strike, T, r, c.kind, q)
        if solved is not None:
            status = "solved"
        else:
            lo, hi = pricing.bounds(spot, c.strike, T, r, c.kind, q)
            status = "below_floor" if mid <= lo else "above_cap" if mid >= hi else "no_solution"
        vendor_px = pricing.price(spot, c.strike, T, r, c.vendor_iv, c.kind, q)
        rows.append(Row(
            ticker=ticker, expiry=c.expiry, dte=c.dte, kind=c.kind, strike=c.strike, spot=spot,
            moneyness=c.strike / spot, mid=mid, spread_pct=(c.ask - c.bid) / mid, vendor_iv=c.vendor_iv,
            solved_iv=solved, gap_vol_pts=(c.vendor_iv - solved) * 100 if solved else None,
            vendor_price_err_pct=(vendor_px - mid) / mid * 100, status=status,
            last_trade_age_days=(now - c.last_trade).total_seconds() / 86400 if c.last_trade else None,
            last_price_iv=pricing.implied_vol(c.last_price, spot, c.strike, T, r, c.kind, q)
            if c.last_price else None,
        ))
    return rows


def audit_ticker(ticker: str, max_dte: int = 120, today: date | None = None) -> list[dict]:
    r, q = market_data.risk_free_rate(), market_data.dividend_yield(ticker)
    s = market_data.spot(ticker)
    return [asdict(row) | {"rate": r, "div_yield": q}
            for row in audit_quotes(ticker, s, market_data.chain(ticker, max_dte, today), r, q)]
