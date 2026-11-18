"""
Next-month risk forecasts as calibrated probabilities and volatility quantiles.

Implements the models, walk-forward protocol and scoring rules fixed in
research/PREREGISTRATION_V2.md. Features use only data known at each day's
close; targets look 21 trading days ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 21
DD_THRESHOLD = -0.05
QUANTILES = (0.1, 0.5, 0.9)

MODELS: dict[str, list[str]] = {
    "M0 climatology": [],
    "M1 VIX": ["log_vix"],
    "M2 HAR": ["log_rv_d", "log_rv_w", "log_rv_m"],
    "M3 VIX + VRP": ["log_vix", "vrp"],
    "M4 composite": ["log_vix", "log_rv_w", "log_rv_m", "vrp", "credit", "credit_chg"],
}


def features(close: pd.Series, vix: pd.Series, credit: pd.Series | None = None) -> pd.DataFrame:
    """Inputs known at each close. `vix` in index points (e.g. 18.5); `credit` in percent."""
    r = np.log(close).diff()
    rv_m_var = r.rolling(22).var() * 252
    out = pd.DataFrame(index=close.index)
    out["log_vix"] = np.log(vix)
    out["log_rv_d"] = np.log(r.abs() * np.sqrt(252) + 1e-4)
    out["log_rv_w"] = np.log(r.rolling(5).std() * np.sqrt(252) + 1e-4)
    out["log_rv_m"] = np.log(np.sqrt(rv_m_var) + 1e-4)
    out["vrp"] = (vix / 100.0) ** 2 - rv_m_var
    if credit is not None:
        c = credit.reindex(close.index).ffill()
        out["credit"] = c
        out["credit_chg"] = c - c.shift(22)
    return out


def targets(close: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    """Drawdown event and log realised volatility over the next `horizon` days (NaN at the end)."""
    r = np.log(close).diff()
    ahead_min = pd.concat([close.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1)
    fwd_rv = pd.concat([r.shift(-k) for k in range(1, horizon + 1)], axis=1).std(axis=1) * np.sqrt(252)
    out = pd.DataFrame(index=close.index)
    out["dd_event"] = (ahead_min / close - 1 <= DD_THRESHOLD).astype(float)
    out["log_fwd_rv"] = np.log(fwd_rv)
    out.iloc[-horizon:] = np.nan
    return out


@dataclass
class Forecast:
    prob: np.ndarray            # P(drawdown event)
    quantiles: np.ndarray       # shape (n, len(QUANTILES)), log volatility


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, inputs: list[str]) -> Forecast:
    """Fit one model on `train` and forecast `test`. Both frames hold inputs and targets."""
    import statsmodels.api as sm

    n = len(test)
    if not inputs:
        prob = np.full(n, train["dd_event"].mean())
        qs = np.tile(np.quantile(train["log_fwd_rv"], QUANTILES), (n, 1))
        return Forecast(prob, qs)

    X_tr = sm.add_constant(train[inputs].to_numpy(), has_constant="add")
    X_te = sm.add_constant(test[inputs].to_numpy(), has_constant="add")
    logit = sm.Logit(train["dd_event"].to_numpy(), X_tr).fit(disp=0, maxiter=200)
    prob = np.clip(logit.predict(X_te), 1e-4, 1 - 1e-4)
    qs = np.column_stack([
        sm.QuantReg(train["log_fwd_rv"].to_numpy(), X_tr).fit(q=q, max_iter=5000).predict(X_te)
        for q in QUANTILES
    ])
    qs.sort(axis=1)    # guard against crossing quantiles
    return Forecast(prob, qs)


def walk_forward(data: pd.DataFrame, inputs: list[str], first_test_year: int, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Expanding-window refit each calendar year. A training row is used only if its
    forward window ended before the test year began (no overlap with the test period).
    """
    data = data.dropna(subset=[*inputs, "dd_event", "log_fwd_rv"])
    rows = []
    for year in range(first_test_year, data.index.max().year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        test = data[(data.index >= start) & (data.index < pd.Timestamp(year=year + 1, month=1, day=1))]
        before = data[data.index < start]
        train = before.iloc[: max(len(before) - horizon, 0)]   # forward window must end before `start`
        if test.empty or len(train) < 500:
            continue
        fc = fit_predict(train, test, inputs)
        frame = pd.DataFrame({"prob": fc.prob, "dd_event": test["dd_event"].to_numpy(),
                              "log_fwd_rv": test["log_fwd_rv"].to_numpy()}, index=test.index)
        for j, q in enumerate(QUANTILES):
            frame[f"q{int(q * 100)}"] = fc.quantiles[:, j]
        rows.append(frame)
    return pd.concat(rows) if rows else pd.DataFrame()


# ── Scoring ──────────────────────────────────────────────────────────────────

def brier(prob: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return (prob - outcome) ** 2


def pinball(pred: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    diff = y - pred
    return np.maximum(q * diff, (q - 1) * diff)


def mean_pinball(frame: pd.DataFrame) -> np.ndarray:
    """Per-day pinball loss averaged over the registered quantiles."""
    y = frame["log_fwd_rv"].to_numpy()
    return np.mean([pinball(frame[f"q{int(q * 100)}"].to_numpy(), y, q) for q in QUANTILES], axis=0)


def coverage(frame: pd.DataFrame) -> float:
    y = frame["log_fwd_rv"]
    return float(((y >= frame["q10"]) & (y <= frame["q90"])).mean())


def reliability(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    table = []
    for b in range(bins):
        m = idx == b
        if m.any():
            table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "days": int(m.sum()),
                          "mean_forecast": round(float(prob[m].mean()), 4),
                          "observed": round(float(outcome[m].mean()), 4)})
    return table


def block_bootstrap_mean(diff: np.ndarray, block: int = 63, resamples: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% interval for the mean of an autocorrelated series (moving-block bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    blocks = int(np.ceil(n / block))
    means = np.empty(resamples)
    for i in range(resamples):
        starts = rng.integers(0, n - block + 1, size=blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        means[i] = diff[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
