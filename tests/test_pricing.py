import math

import pytest

from volevidence import pricing as p


def test_textbook_atm_call():
    # S = K = 100, T = 1, r = 5%, sigma = 20%: d1 = 0.35, call = 10.4506
    assert p.price(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(10.450584, abs=1e-6)
    assert p.greeks(100, 100, 1, 0.05, 0.2, "call").delta == pytest.approx(0.636831, abs=1e-6)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    S, K, T, r, s = 105, 100, 0.75, 0.04, 0.25
    lhs = p.price(S, K, T, r, s, "call", q) - p.price(S, K, T, r, s, "put", q)
    assert lhs == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, s, q, h = 100, 105, 0.8, 0.04, 0.25, 0.02, 0.01
    g = p.greeks(S, K, T, r, s, kind, q)
    f = lambda **kw: p.price(**{"S": S, "K": K, "T": T, "r": r, "sigma": s, "kind": kind, "q": q, **kw})  # noqa: E731
    assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-3)
    assert g.vega == pytest.approx((f(sigma=s + 1e-4) - f(sigma=s - 1e-4)) / 2e-4 / 100, rel=1e-6)
    assert g.rho == pytest.approx((f(r=r + 1e-5) - f(r=r - 1e-5)) / 2e-5 / 100, rel=1e-5)
    assert g.theta == pytest.approx(-(f(T=T + 1e-5) - f(T=T - 1e-5)) / 2e-5 / 365, rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("K", [70, 100, 140])
def test_implied_vol_round_trip(kind, sigma, K):
    target = p.price(100, K, 0.5, 0.04, sigma, kind, 0.01)
    floor, _ = p.bounds(100, K, 0.5, 0.04, kind, 0.01)
    if target - floor < 1e-6:
        pytest.skip("no time value left, so volatility is not identifiable")
    assert p.implied_vol(target, 100, K, 0.5, 0.04, kind, 0.01) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_refuses_prices_outside_bounds():
    lo, hi = p.bounds(100, 90, 0.5, 0.04, "call")
    assert p.implied_vol(lo * 0.99, 100, 90, 0.5, 0.04, "call") is None
    assert p.implied_vol(hi * 1.01, 100, 90, 0.5, 0.04, "call") is None
    assert p.implied_vol(0.0, 100, 90, 0.5, 0.04, "call") is None


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        p.price(100, 100, 1, 0.05, 0.2, "straddle")
    with pytest.raises(ValueError):
        p.price(100, 100, 0, 0.05, 0.2, "call")


def test_american_call_without_dividends_equals_european():
    eu = p.price(100, 95, 0.5, 0.04, 0.3, "call")
    assert p.american_price(100, 95, 0.5, 0.04, 0.3, "call", steps=800) == pytest.approx(eu, abs=0.01)


def test_american_put_is_worth_at_least_european_and_intrinsic():
    for K in (90, 100, 130):
        am = p.american_price(100, K, 0.5, 0.05, 0.25, "put", 0.0)
        assert am >= p.price(100, K, 0.5, 0.05, 0.25, "put") - 1e-9
        assert am >= max(K - 100, 0)
    # deep in the money with a 5% rate: early exercise premium is material
    assert p.american_price(100, 130, 0.5, 0.05, 0.25, "put") > p.price(100, 130, 0.5, 0.05, 0.25, "put") + 0.3


def test_binomial_converges():
    a, b = (p.american_price(100, 105, 1, 0.04, 0.3, "put", 0.01, steps=s) for s in (400, 800))
    assert a == pytest.approx(b, abs=0.01)


@pytest.mark.parametrize("K,kind", [(80, "call"), (100, "put"), (130, "put")])
def test_american_implied_vol_round_trip(K, kind):
    target = p.american_price(100, K, 0.4, 0.04, 0.35, kind, 0.01, steps=200)
    assert p.american_implied_vol(target, 100, K, 0.4, 0.04, kind, 0.01) == pytest.approx(0.35, abs=1e-4)


def test_american_implied_vol_refuses_prices_at_intrinsic():
    assert p.american_implied_vol(30.0, 100, 130, 0.4, 0.04, "put") is None
