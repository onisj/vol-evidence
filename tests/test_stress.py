import math

import pytest

from volevidence import stress

N = stress.WINDOW + stress.RV_DAYS + 1


def ramp(start, step, n=N):
    return [start + step * i for i in range(n)]


def wiggle(base, amp, n=N):
    return [base * (1 + amp * math.sin(i)) for i in range(n)]


def test_weights_sum_to_one():
    assert sum(stress.WEIGHTS.values()) == pytest.approx(1.0)


def test_percentile_rank():
    assert stress.percentile_rank([1, 2, 3, 4], 3) == 50.0
    assert stress.percentile_rank([], 3) == 50.0
    assert stress.percentile_rank([1, 2], 9) == 100.0


def test_realized_vol_of_constant_growth_is_zero():
    out = stress.realized_vol([100 * 1.001**i for i in range(40)])
    assert len(out) == 40 - 1 - stress.RV_DAYS + 1
    assert max(out) == pytest.approx(0.0, abs=1e-12)


def test_calm_market_reads_higher_than_stressed():
    calm = stress.read(ramp(30, -0.05), ramp(28, -0.02), ramp(70, 0.02), [100.0] * N, wiggle(400, 0.002))
    # a bounded wiggle keeps prices positive while realised volatility jumps at the end
    tail = [400 * (1 + (0.002 if i < N - 22 else 0.03) * math.sin(i)) for i in range(N)]
    stressed = stress.read(ramp(12, 0.05), ramp(15, 0.01), ramp(80, -0.02), [100.0] * N, tail)
    assert calm.score > 80 and calm.label == "calm"
    assert stressed.score < 20 and stressed.label == "stressed"


def test_read_is_deterministic_and_bounded():
    args = (wiggle(20, 0.2), wiggle(22, 0.1), wiggle(75, 0.05), wiggle(95, 0.02), wiggle(400, 0.01))
    a, b = stress.read(*args), stress.read(*args)
    assert a == b and 0 <= a.score <= 100 and set(a.components) == set(stress.WEIGHTS)


def test_short_history_is_rejected():
    with pytest.raises(ValueError):
        stress.read(*([1.0] * 10 for _ in range(5)))


@pytest.mark.parametrize("score,label", [(60, "calm"), (59.9, "normal"), (40, "normal"), (39.9, "stressed")])
def test_labels(score, label):
    assert stress.label(score) == label
