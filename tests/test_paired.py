"""The paired-comparison statistics behind the accuracy gate, checked against brute force and recorded counts.

Every interval quoted in the docs is computed here from counts, not measured."""

from __future__ import annotations

import math

import pytest

from rigtune.calibrate.paired import (
    certifies,
    items_needed,
    mcnemar_p,
    restricted_lost_rate,
    score_bounds,
    score_z,
)


def _log_likelihood(p: float, d: float, lost: int, gained: int, n: int) -> float:
    q, rest = p + d, 1.0 - 2.0 * p - d
    if p < 0 or q < 0 or rest < 0:
        return -math.inf
    total = 0.0
    for count, prob in ((lost, p), (gained, q), (n - lost - gained, rest)):
        if count:
            if prob <= 0:
                return -math.inf
            total += count * math.log(prob)
    return total


def _grid_argmax(d: float, lost: int, gained: int, n: int, steps: int = 400) -> float:
    """The lost-item probability that maximises the likelihood with the difference held at d, by brute force:
    a grid over the whole feasible range, then a grid as fine again around its best point (the log-likelihood
    is concave in p, so the maximum is inside that cell's neighbours)."""
    lo, hi = max(0.0, -d), (1.0 - d) / 2.0
    for _ in range(3):
        grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
        best = max(grid, key=lambda p: _log_likelihood(p, d, lost, gained, n))
        step = (hi - lo) / steps
        lo, hi = max(max(0.0, -d), best - step), min((1.0 - d) / 2.0, best + step)
    return best


@pytest.mark.parametrize("lost,gained,n", [(4, 0, 200), (0, 0, 300), (6, 2, 300), (65, 34, 1319), (0, 5, 50)])
@pytest.mark.parametrize("d", [-0.06, -0.02, -0.005, 0.01, 0.04])
def test_the_closed_form_restricted_estimate_is_the_likelihood_maximum(lost, gained, n, d):
    closed = restricted_lost_rate(d, lost, gained, n)
    assert closed == pytest.approx(_grid_argmax(d, lost, gained, n), abs=1e-6)


STEP = 0.13 / 1300  # the d grid: -8 to +5 points in steps of 0.01 points


def _brute_force_interval(lost: int, gained: int, n: int):
    """The set of differences the score test does not reject at 5%, found by grid search over d with the
    restricted estimate itself found by grid maximisation: no closed form or bisection anywhere."""
    accepted = []
    for i in range(1301):
        d = -0.08 + STEP * i
        p = _grid_argmax(d, lost, gained, n, steps=100)
        var = n * (2 * p + d * (1 - d))
        if var <= 0:
            continue
        if abs((gained - lost - n * d) / math.sqrt(var)) <= 1.959963984540054:
            accepted.append(d)
    return min(accepted), max(accepted)


@pytest.mark.parametrize("lost,gained,n", [(4, 0, 200), (0, 0, 300), (6, 2, 300)])
def test_the_score_interval_matches_a_brute_force_search(lost, gained, n):
    lo, hi = score_bounds(lost, gained, n)
    blo, bhi = _brute_force_interval(lost, gained, n)
    assert lo == pytest.approx(blo, abs=2 * STEP) and hi == pytest.approx(bhi, abs=2 * STEP)


def test_no_changed_answers_in_300_items_still_leaves_1_3_points_either_way():
    """Where the Wald interval says +/-0: with nothing discordant its standard error is zero at any size."""
    lo, hi = score_bounds(0, 0, 300)
    assert round(lo * 100, 1) == -1.3 and round(hi * 100, 1) == 1.3
    assert lo == pytest.approx(-3.8415 / (300 + 3.8415), rel=1e-3)  # the closed form when b = c = 0
    assert round(score_bounds(0, 0, 100)[0] * 100, 1) == -3.7
    assert round(score_bounds(0, 0, 200)[0] * 100, 1) == -1.9


def test_qwen_14b_gptq_against_fp8_cannot_be_certified_within_3_points():
    """Recorded counts (benchmarks/strategies/quality-14b-gsm8k.json): 200 problems, 4 lost, 0 gained."""
    lo, hi = score_bounds(4, 0, 200)
    assert round(lo * 100, 1) == -5.0 and round(hi * 100, 1) == -0.1
    assert mcnemar_p(4, 0) == 0.125  # not significant, which is why a significance test would admit it
    assert not certifies(4, 0, 200, 0.03) and not certifies(4, 0, 200, 0.05)
    assert certifies(4, 0, 200, 0.06)
    # The same rate over the whole test set (26 of 1319) would certify 3 points.
    assert round(score_bounds(26, 0, 1319)[0] * 100, 1) == -2.9 and certifies(26, 0, 1319, 0.03)


def test_the_score_statistic_falls_as_the_hypothesised_difference_rises():
    for lost, gained, n in ((4, 0, 200), (0, 0, 300), (10, 30, 100)):
        zs = [score_z(-0.3 + 0.6 * i / 200, lost, gained, n) for i in range(1, 200)]
        assert all(a >= b for a, b in zip(zs, zs[1:]))


def test_bounds_bracket_the_observed_difference_and_narrow_with_more_items():
    lo, hi = score_bounds(10, 4, 300)
    assert lo < (4 - 10) / 300 < hi
    lo2, hi2 = score_bounds(20, 8, 600)
    assert hi2 - lo2 < hi - lo
    assert score_bounds(0, 0, 0) == (-1.0, 1.0)
    assert score_bounds(5, 0, 5)[0] == -1.0
    with pytest.raises(ValueError):
        score_bounds(3, 3, 5)


def test_mcnemar_moved_without_changing():
    from rigtune.bench.ablation import mcnemar_p as old

    assert old is mcnemar_p
    assert abs(mcnemar_p(65, 34) - 0.0024) < 0.0002  # GSM8K 3B, fp8 against bf16
    assert mcnemar_p(0, 0) == 1.0


def test_items_needed_grows_as_the_tolerance_tightens_and_is_none_past_the_loss():
    """Identical answers need about 125 items to certify 3 points (the bound on 100 is -3.7); 4 lost of 200
    needs about 2045 for an 80% chance."""
    assert items_needed(0, 0, 100, 0.03) == 125
    assert items_needed(4, 0, 200, 0.03) == 2045
    tolerances = [0.025, 0.03, 0.04, 0.06, 0.1]
    needs = [items_needed(4, 0, 200, t) for t in tolerances]
    assert all(a > b for a, b in zip(needs, needs[1:]))
    assert items_needed(4, 0, 200, 0.02) is None  # an observed loss of 2 points never certifies 2
    assert items_needed(6, 0, 200, 0.03) is None
    assert items_needed(0, 0, 0, 0.03) is None


def test_items_needed_gives_the_promised_power():
    """Simulated: at the suggested size, with the observed rates as the truth, about 80% of samples certify."""
    import random

    rng = random.Random(0)
    n = items_needed(20, 0, 1000, 0.03)
    hits = 0
    for _ in range(300):
        draws = [rng.random() for _ in range(n)]
        hits += certifies(sum(u < 0.02 for u in draws), 0, n, 0.03)
    assert 0.72 <= hits / 300 <= 0.88
