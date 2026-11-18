"""
Market data from yfinance, with the unit and format traps handled.

yfinance is unofficial and its fields change. Two traps this module guards:

* Dividend yield. `dividendYield` is quoted in percent (2.15 means 2.15%),
  while `trailingAnnualDividendYield` and `yield` are fractions. Values above
  MAX_DIVIDEND_YIELD are treated as a units error and rejected.
* Risk-free rate. ^IRX (13-week T-bill) is quoted in percent; it is converted
  to a continuously compounded rate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

FALLBACK_RATE = 0.04
MAX_DIVIDEND_YIELD = 0.20


@dataclass(frozen=True)
class Quote:
    expiry: str
    dte: int
    kind: str
    strike: float
    bid: float | None
    ask: float | None
    vendor_iv: float | None
    volume: float | None
    open_interest: float | None

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return None


def _num(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def dividend_yield_from_info(info: dict) -> float:
    """Dividend yield as a fraction from a yfinance `info` dict (0.0 if unknown)."""
    for field, divisor in (("trailingAnnualDividendYield", 1.0), ("yield", 1.0), ("dividendYield", 100.0)):
        raw = _num(info.get(field))
        if raw is None:
            continue
        value = raw / divisor
        if 0.0 <= value <= MAX_DIVIDEND_YIELD:
            return value
        log.warning("rejected dividend yield %s=%r as a probable units error", field, raw)
    return 0.0


def continuous_rate(percent: float) -> float:
    """Annual simple rate quoted in percent (4.15) -> continuously compounded (0.0406)."""
    return math.log1p(percent / 100.0)


@lru_cache(maxsize=1)
def risk_free_rate() -> float:
    try:
        import yfinance as yf

        closes = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        if not closes.empty and 0.0 <= float(closes.iloc[-1]) < 25.0:
            return continuous_rate(float(closes.iloc[-1]))
    except Exception as exc:  # network or schema failure
        log.warning("risk-free rate unavailable (%s); using %.3f", exc, FALLBACK_RATE)
    return FALLBACK_RATE


@lru_cache(maxsize=256)
def dividend_yield(ticker: str) -> float:
    try:
        import yfinance as yf

        return dividend_yield_from_info(yf.Ticker(ticker).info or {})
    except Exception as exc:
        log.warning("%s: dividend yield unavailable (%s); using 0", ticker, exc)
        return 0.0


def spot(ticker: str) -> float:
    import yfinance as yf

    return float(yf.Ticker(ticker).history(period="5d")["Close"].dropna().iloc[-1])


def chain(ticker: str, max_dte: int, today: date | None = None) -> list[Quote]:
    """Every listed contract out to `max_dte` days, calls and puts."""
    import yfinance as yf

    today = today or date.today()
    tk = yf.Ticker(ticker)
    quotes: list[Quote] = []
    for expiry in tk.options or ():
        dte = (date.fromisoformat(expiry) - today).days
        if not 0 < dte <= max_dte:
            continue
        frames = tk.option_chain(expiry)
        for kind, frame in (("call", frames.calls), ("put", frames.puts)):
            for row in frame.to_dict("records"):
                quotes.append(Quote(
                    expiry=expiry, dte=dte, kind=kind, strike=float(row["strike"]),
                    bid=_num(row.get("bid")), ask=_num(row.get("ask")),
                    vendor_iv=_num(row.get("impliedVolatility")),
                    volume=_num(row.get("volume")), open_interest=_num(row.get("openInterest")),
                ))
    return quotes


def closes(tickers: list[str], start: str) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, only days where all traded."""
    import yfinance as yf

    frame = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return frame.dropna(how="any")
