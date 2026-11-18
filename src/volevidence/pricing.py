"""
Black-Scholes-Merton pricing for European options with a continuous dividend yield.

Price, Greeks, no-arbitrage bounds and an implied-volatility solver.
Pure Python: only the math module.

Conventions:
    T      time to expiry in years
    r, q   continuously compounded rate and dividend yield
    theta  per calendar day
    vega   per 1 point of volatility (25% -> 26%)
    rho    per 1 point of rate
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SIGMA_MIN, SIGMA_MAX = 1e-4, 5.0   # solver search range: 0.01% to 500% volatility


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _check(kind: str) -> str:
    k = kind.lower()
    if k not in ("call", "put"):
        raise ValueError(f"option kind must be 'call' or 'put', got {kind!r}")
    return k


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(f"invalid inputs: S={S}, K={K}, T={T}, sigma={sigma}")
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def price(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> float:
    """European option price under Black-Scholes-Merton."""
    k = _check(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    fwd_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if k == "call":
        return fwd_s * norm_cdf(d1) - pv_k * norm_cdf(d2)
    return pv_k * norm_cdf(-d2) - fwd_s * norm_cdf(-d1)


def bounds(S: float, K: float, T: float, r: float, kind: str, q: float = 0.0) -> tuple[float, float]:
    """Model-free (lower, upper) price bounds for a European option."""
    k = _check(kind)
    fwd_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if k == "call":
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def implied_vol(
    target: float, S: float, K: float, T: float, r: float, kind: str,
    q: float = 0.0, tol: float = 1e-8, max_iter: int = 100,
) -> float | None:
    """
    Volatility that reproduces `target` under Black-Scholes-Merton.

    Newton steps on vega inside a shrinking bisection bracket, so it converges
    whenever a solution exists. Returns None when the price lies on or outside
    the no-arbitrage bounds, where no volatility can produce it.
    """
    if target is None or target <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    lower, upper = bounds(S, K, T, r, kind, q)
    if not lower < target < upper:
        return None
    lo, hi = SIGMA_MIN, SIGMA_MAX
    if price(S, K, T, r, lo, kind, q) > target or price(S, K, T, r, hi, kind, q) < target:
        return None

    sigma = 0.3
    for _ in range(max_iter):
        err = price(S, K, T, r, sigma, kind, q) - target
        if abs(err) < tol:
            return sigma
        if err > 0:
            hi = sigma
        else:
            lo = sigma
        d1, _ = d1_d2(S, K, T, r, sigma, q)
        vega = S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)
        newton = sigma - err / vega if vega > 1e-12 else None
        sigma = newton if newton is not None and lo < newton < hi else 0.5 * (lo + hi)
        if hi - lo < tol:
            return sigma
    return sigma


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> Greeks:
    k = _check(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r, root_t = math.exp(-q * T), math.exp(-r * T), math.sqrt(T)
    pdf = norm_pdf(d1)
    decay = -S * disc_q * pdf * sigma / (2.0 * root_t)
    if k == "call":
        delta = disc_q * norm_cdf(d1)
        theta = decay - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = decay + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2)
    return Greeks(
        delta=delta,
        gamma=disc_q * pdf / (S * sigma * root_t),
        theta=theta / 365.0,
        vega=S * disc_q * root_t * pdf / 100.0,
        rho=rho / 100.0,
    )


# ── American exercise ────────────────────────────────────────────────────────

def american_price(S: float, K: float, T: float, r: float, sigma: float, kind: str,
                   q: float = 0.0, steps: int = 300) -> float:
    """
    American option price on a Cox-Ross-Rubinstein binomial tree.

    Uses a continuous dividend yield, an approximation for stocks that pay
    discrete dividends. Converges to the European price when early exercise
    is never optimal (a call with q = 0).
    """
    k = _check(kind)
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(f"invalid inputs: S={S}, K={K}, T={T}, sigma={sigma}")
    dt = T / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((r - q) * dt)
    p_up = (growth - down) / (up - down)
    if not 0.0 < p_up < 1.0:
        raise ValueError("binomial probability out of range; increase steps")
    disc = math.exp(-r * dt)
    sign = 1.0 if k == "call" else -1.0
    values = [max(sign * (S * up ** j * down ** (steps - j) - K), 0.0) for j in range(steps + 1)]
    for n in range(steps - 1, -1, -1):
        for j in range(n + 1):
            cont = disc * (p_up * values[j + 1] + (1.0 - p_up) * values[j])
            exercise = sign * (S * up ** j * down ** (n - j) - K)
            values[j] = max(cont, exercise)
    return values[0]


def american_implied_vol(
    target: float, S: float, K: float, T: float, r: float, kind: str,
    q: float = 0.0, steps: int = 200, tol: float = 1e-6, max_iter: int = 60,
) -> float | None:
    """
    Volatility that reproduces `target` under American exercise, by bisection.

    Returns None when the price is at or below immediate-exercise value, where
    no volatility is identifiable, or above the maximum the tree can produce.
    """
    k = _check(kind)
    if target is None or target <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max((S - K) if k == "call" else (K - S), 0.0)
    if target <= intrinsic + 1e-9:
        return None
    # The tree needs per-step volatility above per-step drift, or the up-probability leaves (0, 1).
    lo = max(SIGMA_MIN, 2.0 * abs(r - q) * math.sqrt(T / steps) + 1e-4)
    hi = SIGMA_MAX
    if american_price(S, K, T, r, hi, kind, q, steps) < target:
        return None
    if american_price(S, K, T, r, lo, kind, q, steps) > target:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if american_price(S, K, T, r, mid, kind, q, steps) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
