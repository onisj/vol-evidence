"""
Backtest of the stress gauge against the hypotheses in PREREGISTRATION.md.

    python research/stress_backtest.py

Writes results/stress_daily.csv, results/stress_summary.json and two charts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from volevidence import market_data, stress

OUT = Path(__file__).parent / "results"
HORIZON = 21
DD_THRESHOLD = -0.05
BLOCK, RESAMPLES, SEED = 6, 2000, 7
NEED = stress.WINDOW + stress.RV_DAYS + 1


def score_history(px: pd.DataFrame) -> pd.DataFrame:
    cols = {t: px[t].to_numpy(dtype=float) for t in stress.TICKERS}
    rows = []
    for i in range(NEED - 1, len(px)):
        sl = slice(i - NEED + 1, i + 1)
        r = stress.read(*(list(cols[t][sl]) for t in ["^VIX", "^VIX3M", "HYG", "IEF", "SPY"]))
        rows.append({"date": px.index[i], "gauge": r.score, "label": r.label, "vix": cols["^VIX"][i],
                     "vix_pct_input": stress.vix_baseline(list(cols["^VIX"][sl])),
                     **{f"c_{k}": v for k, v in r.components.items()}})
    return pd.DataFrame(rows).set_index("date")


def forward_outcomes(spy: pd.Series) -> pd.DataFrame:
    rets = np.log(spy).diff()
    fwd = pd.DataFrame(index=spy.index)
    ahead = pd.concat([spy.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    fwd["fwd_ret"] = spy.shift(-HORIZON) / spy - 1
    fwd["fwd_maxdd"] = (ahead.min(axis=1) / spy - 1).clip(upper=0)
    fwd["fwd_vol"] = pd.concat([rets.shift(-k) for k in range(1, HORIZON + 1)], axis=1).std(axis=1) * np.sqrt(252)
    fwd["fwd_dd5"] = (fwd["fwd_maxdd"] <= DD_THRESHOLD).astype(float)
    fwd.iloc[-HORIZON:] = np.nan
    return fwd


def bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    stats = []
    for _ in range(RESAMPLES):
        starts = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        stats.append(spearmanr(x[idx], y[idx]).statistic)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def rank_ols(sample: pd.DataFrame, target: str) -> dict:
    ranks = sample[["gauge", "neg_vix", target]].rank()
    X = np.column_stack([np.ones(len(ranks)), ranks["gauge"], ranks["neg_vix"]])
    y = ranks[target].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(y) - 3) * np.linalg.inv(X.T @ X)))
    return {"t_gauge": round(float(beta[1] / se[1]), 2), "t_raw_vix": round(float(beta[2] / se[2]), 2)}


def analyze(df: pd.DataFrame) -> dict:
    df = df.dropna().assign(neg_vix=lambda d: -d["vix"])
    sample = df.iloc[::HORIZON]
    signals = {"gauge": "gauge", "vix_percentile_input": "vix_pct_input", "raw_vix": "neg_vix"}
    corr = {}
    for name, col in signals.items():
        corr[name] = {}
        for target in ["fwd_maxdd", "fwd_vol", "fwd_dd5", "fwd_ret"]:
            rho = float(spearmanr(sample[col], sample[target]).statistic)
            lo, hi = bootstrap_ci(sample[col].to_numpy(), sample[target].to_numpy())
            corr[name][target] = {"spearman": round(rho, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    q = df.assign(quintile=pd.qcut(df["gauge"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]))
    by_q = q.groupby("quintile", observed=True).agg(
        days=("gauge", "size"), gauge_min=("gauge", "min"), gauge_max=("gauge", "max"),
        p_dd5_pct=("fwd_dd5", lambda s: 100 * s.mean()),
        mean_fwd_vol_pct=("fwd_vol", lambda s: 100 * s.mean()),
        mean_fwd_maxdd_pct=("fwd_maxdd", lambda s: 100 * s.mean()),
        mean_fwd_ret_pct=("fwd_ret", lambda s: 100 * s.mean()),
    ).round(2)
    by_label = df.groupby("label").agg(
        days=("gauge", "size"), p_dd5_pct=("fwd_dd5", lambda s: 100 * s.mean()),
        mean_fwd_vol_pct=("fwd_vol", lambda s: 100 * s.mean()),
        mean_fwd_ret_pct=("fwd_ret", lambda s: 100 * s.mean()),
    ).round(2)

    dd_sorted = list(by_q["p_dd5_pct"])
    vol_sorted = list(by_q["mean_fwd_vol_pct"])
    reg = rank_ols(sample, "fwd_maxdd")
    verdicts = {
        "H1_sorting": all(a > b for a, b in zip(dd_sorted, dd_sorted[1:]))
        and all(a > b for a, b in zip(vol_sorted, vol_sorted[1:])),
        "H2_beyond_raw_vix_on_drawdown": abs(reg["t_gauge"]) > 2,
        "H3_volatility_vs_raw_vix": corr["gauge"]["fwd_vol"]["spearman"] < corr["raw_vix"]["fwd_vol"]["spearman"],
    }
    return {
        "first_day": str(df.index[0].date()), "last_day": str(df.index[-1].date()),
        "days": int(len(df)), "non_overlapping_samples": int(len(sample)),
        "base_rate_dd5_pct": round(100 * float(df["fwd_dd5"].mean()), 2),
        "verdicts": verdicts, "rank_regression_fwd_maxdd": reg,
        "rank_regression_fwd_vol": rank_ols(sample, "fwd_vol"),
        "spearman_next_21d": corr,
        "by_quintile": by_q.reset_index().to_dict(orient="records"),
        "by_label": by_label.reset_index().to_dict(orient="records"),
    }


# ── Exploratory analyses (not pre-registered) ───────────────────────────────

def har_forecast(spy: pd.Series, refit_every: int = HORIZON) -> pd.Series:
    """
    HAR-RV forecast of next-21-day realised volatility, using only past data.

    Regresses next-month variance on daily, weekly and monthly trailing variance,
    refitting every `refit_every` days on observations whose outcome was
    already known (target window ended before the forecast date).
    """
    r2 = np.log(spy).diff() ** 2
    feats = pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()})
    target = pd.concat([r2.shift(-k) for k in range(1, HORIZON + 1)], axis=1).mean(axis=1)
    out = pd.Series(np.nan, index=spy.index)
    beta = None
    valid = feats.notna().all(axis=1)
    for i in range(300, len(spy)):
        if beta is None or i % refit_every == 0:
            known = valid & target.notna()
            known.iloc[i - HORIZON:] = False
            X = np.column_stack([np.ones(int(known.sum())), feats[known].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, target[known].to_numpy(), rcond=None)
        if valid.iloc[i]:
            var = float(beta[0] + feats.iloc[i].to_numpy() @ beta[1:])
            out.iloc[i] = np.sqrt(max(var, 0.0) * 252)
    return out


def exploratory(df: pd.DataFrame, spy: pd.Series) -> dict:
    d = df.copy()
    d["trailing_rv"] = np.log(spy).diff().rolling(HORIZON).std().reindex(d.index) * np.sqrt(252)
    d["har"] = har_forecast(spy).reindex(d.index)
    d = d.dropna()
    sample = d.iloc[::HORIZON]
    vol_corr = {
        "gauge (sign-flipped)": round(float(spearmanr(-sample["gauge"], sample["fwd_vol"]).statistic), 3),
        "raw_vix": round(float(spearmanr(sample["vix"], sample["fwd_vol"]).statistic), 3),
        "trailing_21d_realised_vol": round(float(spearmanr(sample["trailing_rv"], sample["fwd_vol"]).statistic), 3),
        "har_rv_forecast": round(float(spearmanr(sample["har"], sample["fwd_vol"]).statistic), 3),
    }
    calibration = d.groupby("label").agg(
        days=("gauge", "size"),
        median_fwd_maxdd_pct=("fwd_maxdd", lambda x: 100 * x.median()),
        worst5pct_fwd_maxdd_pct=("fwd_maxdd", lambda x: 100 * x.quantile(0.05)),
        median_fwd_vol_pct=("fwd_vol", lambda x: 100 * x.median()),
        p95_fwd_vol_pct=("fwd_vol", lambda x: 100 * x.quantile(0.95)),
    ).round(2)
    windows = {"ex_2008_09": ("2008-09-01", "2009-06-30"), "ex_2020_covid": ("2020-02-15", "2020-06-30")}
    robustness = {}
    for name, (a, b) in windows.items():
        sub = df.dropna()
        sub = sub[(sub.index < a) | (sub.index > b)].assign(neg_vix=lambda x: -x["vix"])
        samp = sub.iloc[::HORIZON]
        q = sub.groupby(pd.qcut(sub["gauge"], 5, labels=False))["fwd_dd5"].mean() * 100
        robustness[name] = {
            "days": int(len(sub)),
            "p_dd5_lowest_vs_highest_quintile_pct": [round(float(q.iloc[0]), 2), round(float(q.iloc[-1]), 2)],
            "rank_regression_fwd_maxdd": rank_ols(samp, "fwd_maxdd"),
            "spearman_fwd_vol_gauge_vs_raw_vix": [
                round(float(spearmanr(samp["gauge"], samp["fwd_vol"]).statistic), 3),
                round(float(spearmanr(samp["neg_vix"], samp["fwd_vol"]).statistic), 3)],
        }
    return {"note": "Exploratory, not pre-registered. Correlations use non-overlapping 21-day samples.",
            "next_month_vol_rank_correlation": vol_corr,
            "calibration_by_label": calibration.reset_index().to_dict(orient="records"),
            "robustness_excluding_crises": robustness}


def charts(df: pd.DataFrame, spy: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna()
    grp = d.groupby(pd.qcut(d["gauge"], 5, labels=["Q1\nmost stressed", "Q2", "Q3", "Q4", "Q5\ncalmest"]),
                    observed=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(grp.size().index.astype(str), grp["fwd_dd5"].mean() * 100, color="#b5523b")
    axes[0].axhline(d["fwd_dd5"].mean() * 100, color="#555", lw=1, ls="--", label="all days")
    axes[0].set_title("Chance of a 5%+ SPY drawdown\nin the next 21 trading days")
    axes[0].set_ylabel("% of days")
    axes[0].legend(frameon=False)
    axes[1].bar(grp.size().index.astype(str), grp["fwd_vol"].mean() * 100, color="#3b6ea5")
    axes[1].set_title("SPY realised volatility\nover the next 21 trading days")
    axes[1].set_ylabel("annualised %")
    for ax in axes:
        ax.set_xlabel("Stress gauge quintile")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "stress_quintiles.png", dpi=160)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    dd = spy / spy.cummax() - 1
    ax1.fill_between(dd.index, dd * 100, 0, color="#b5523b", alpha=0.35, label="SPY drawdown from peak")
    ax1.set_ylabel("SPY drawdown %")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["gauge"].rolling(5).mean(), color="#3b6ea5", lw=0.8, label="Stress gauge (5-day mean)")
    ax2.set_ylabel("Gauge (100 = calmest)")
    ax2.set_ylim(0, 100)
    ax1.set_title("Stress gauge vs SPY drawdowns", pad=28)
    ax1.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.savefig(OUT / "stress_timeline.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = market_data.closes(stress.TICKERS, start="2006-01-01")
    df = score_history(px).join(forward_outcomes(px["SPY"]))
    df.to_csv(OUT / "stress_daily.csv")
    summary = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "weights": stress.WEIGHTS, "window": stress.WINDOW, "horizon": HORIZON, **analyze(df),
               "exploratory": exploratory(df, px["SPY"])}
    (OUT / "stress_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    charts(df, px["SPY"].loc[df.index[0]:])
    print(json.dumps({k: summary[k] for k in ["first_day", "last_day", "days", "verdicts",
                                              "rank_regression_fwd_maxdd", "exploratory"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
