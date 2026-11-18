import pytest

from volevidence import pricing
from volevidence.iv_audit import audit_quotes
from volevidence.market_data import Quote

S, R, Q = 100.0, 0.04, 0.01


def quote(kind, K, true_iv, vendor_iv, dte=60, half_spread=0.01):
    mid = pricing.price(S, K, dte / 365, R, true_iv, kind, Q)
    return Quote("2027-01-15", dte, kind, K, mid - half_spread, mid + half_spread, vendor_iv, 10, 100)


def test_gap_equals_vendor_minus_quote_iv():
    rows = audit_quotes("X", S, [quote("call", 105, 0.25, 0.28), quote("put", 95, 0.30, 0.30)], R, Q)
    assert [r.status for r in rows] == ["solved", "solved"]
    assert rows[0].gap_vol_pts == pytest.approx(3.0, abs=1e-4)
    assert rows[1].gap_vol_pts == pytest.approx(0.0, abs=1e-4)
    assert rows[0].vendor_price_err_pct > 0


def test_quotes_below_floor_are_classified_not_solved():
    floor, _ = pricing.bounds(S, 130, 30 / 365, R, "put", Q)
    stale = Quote("2027-01-15", 30, "put", 130.0, floor - 0.6, floor - 0.2, 0.3, 1, 1)
    row = audit_quotes("X", S, [stale], R, Q)[0]
    assert row.status == "below_floor" and row.solved_iv is None and row.gap_vol_pts is None


def test_one_sided_quotes_are_skipped():
    one_sided = Quote("2027-01-15", 30, "call", 100.0, 0.0, 1.0, 0.3, 1, 1)
    assert audit_quotes("X", S, [one_sided], R, Q) == []


def test_last_trade_age_is_recorded():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 11, 18, 21, tzinfo=timezone.utc)
    stale = Quote("2027-01-15", 60, "call", 100.0, 5.0, 5.2, 0.3, 1, 1, last_trade=now - timedelta(days=10))
    assert audit_quotes("X", S, [stale], R, Q, now=now)[0].last_trade_age_days == pytest.approx(10.0)
