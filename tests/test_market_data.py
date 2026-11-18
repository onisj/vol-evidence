import math

import pytest

from volevidence import market_data as md


def test_prefers_fraction_field_over_percent_field():
    # yfinance quotes `dividendYield` in percent: 2.15 means 2.15%
    assert md.dividend_yield_from_info({"dividendYield": 2.15, "trailingAnnualDividendYield": 0.0213}) == 0.0213


def test_percent_field_divided_by_100_when_alone():
    assert md.dividend_yield_from_info({"dividendYield": 2.15}) == pytest.approx(0.0215)


def test_units_errors_rejected():
    assert md.dividend_yield_from_info({"trailingAnnualDividendYield": 2.15}) == 0.0
    assert md.dividend_yield_from_info({"yield": float("nan")}) == 0.0
    assert md.dividend_yield_from_info({}) == 0.0


def test_continuous_rate():
    assert md.continuous_rate(4.15) == pytest.approx(math.log(1.0415))


def test_quote_mid_needs_two_sided_market():
    q = md.Quote("2027-01-15", 30, "call", 100.0, 1.0, 1.2, 0.3, None, None)
    assert q.mid == pytest.approx(1.1)
    assert md.Quote("2027-01-15", 30, "call", 100.0, 0.0, 1.2, 0.3, None, None).mid is None
    assert md.Quote("2027-01-15", 30, "call", 100.0, None, 1.2, 0.3, None, None).mid is None


def test_timestamp_parsing():
    assert md._timestamp("2026-11-17 15:07:46+00:00").day == 17
    assert md._timestamp(None) is None and md._timestamp(float("nan")) is None
