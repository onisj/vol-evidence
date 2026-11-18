import numpy as np
import pandas as pd
import pytest

from volevidence import riskforecast as rf


def series(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1995-01-02", periods=n)
    vol = 0.15 + 0.1 * (np.sin(np.arange(n) / 150) > 0.6)          # calm and stormy regimes
    r = rng.normal(0, vol / np.sqrt(252))
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    vix = pd.Series(100 * vol * 1.1 + rng.normal(0, 0.5, n), index=idx)
    credit = pd.Series(2 + 10 * (vol - 0.15) + rng.normal(0, 0.05, n), index=idx)
    return close, vix, credit


def test_targets_look_only_forward_and_end_with_nan():
    close = pd.Series([100, 100, 100, 94, 100, 100], index=pd.bdate_range("2020-01-01", periods=6), dtype=float)
    t = rf.targets(close, horizon=2)
    assert t["dd_event"].iloc[1] == 1.0          # day 1 sees the 6% drop on day 3 within 2 days
    assert t["dd_event"].iloc[0] == 0.0          # day 0's window (days 1-2) has no drop
    assert t.iloc[-2:].isna().all().all()


def test_features_do_not_use_future_data():
    close, vix, credit = series()
    full = rf.features(close, vix, credit)
    cut = rf.features(close.iloc[:2000], vix.iloc[:2000], credit.iloc[:2000])
    pd.testing.assert_frame_equal(full.iloc[:2000], cut)


def test_vrp_definition():
    close, vix, _ = series()
    f = rf.features(close, vix)
    rv_var = np.log(close).diff().rolling(22).var() * 252
    assert f["vrp"].iloc[100] == pytest.approx((vix.iloc[100] / 100) ** 2 - rv_var.iloc[100])


def test_scoring_rules():
    assert rf.brier(np.array([0.2]), np.array([1.0]))[0] == pytest.approx(0.64)
    assert rf.pinball(np.array([1.0]), np.array([2.0]), 0.9)[0] == pytest.approx(0.9)
    assert rf.pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.1)
    rel = rf.reliability(np.array([0.05, 0.05, 0.95]), np.array([0.0, 1.0, 1.0]))
    assert rel[0] == {"bin": "0.0-0.1", "days": 2, "mean_forecast": 0.05, "observed": 0.5}


def test_block_bootstrap_brackets_the_mean():
    x = np.random.default_rng(3).normal(0.5, 1, 2000)
    lo, hi = rf.block_bootstrap_mean(x, block=20, resamples=500)
    assert lo < x.mean() < hi and lo > 0


def test_walk_forward_trains_only_on_completed_windows(monkeypatch):
    close, vix, credit = series()
    data = rf.features(close, vix, credit).join(rf.targets(close))
    seen = []
    real = rf.fit_predict

    def spy(train, test, inputs):
        seen.append((train.index.max(), test.index.min()))
        return real(train, test, inputs)

    monkeypatch.setattr(rf, "fit_predict", spy)
    out = rf.walk_forward(data, ["log_vix"], first_test_year=2000)
    assert not out.empty
    for train_end, test_start in seen:
        # the last training row's 21-day forward window ends before the test year starts
        assert data.index.get_loc(train_end) + rf.HORIZON < data.index.get_loc(test_start)


def test_informative_model_beats_climatology_on_synthetic_regimes():
    close, vix, credit = series(n=4000)
    data = rf.features(close, vix, credit).join(rf.targets(close))
    clim = rf.walk_forward(data, [], first_test_year=2003)
    vixm = rf.walk_forward(data, ["log_vix"], first_test_year=2003)
    assert rf.mean_pinball(vixm).mean() < rf.mean_pinball(clim).mean()
    assert 0.6 < rf.coverage(vixm) < 0.95
