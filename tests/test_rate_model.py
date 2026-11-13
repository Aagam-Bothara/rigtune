"""The rate model: joules per request at an arrival rate, composed from closed-loop levels."""

from __future__ import annotations

import math

import pytest

from rigtune.calibrate.rate import (compose, hold_factor, in_flight, levels, mean_power, poisson_quantile,
                                    serving_energy)
from rigtune.models import TrialMetrics


def level(c: int, rate_rps: float, power_w: float, duration_s: float = 20.0, tokens: int = 500,
          p95: float = 100.0) -> TrialMetrics:
    n = int(round(rate_rps * duration_s))
    return TrialMetrics(concurrency=c, requests=n, output_tokens=n * tokens, duration_s=duration_s, power_w=power_w,
                        joules_per_token=power_w * duration_s / (n * tokens), tok_s=n * tokens / duration_s,
                        ttft_ms=p95 / 2, ttft_p95_ms=p95)


def trial(*lv: TrialMetrics, idle_w=60.0, hold_j=30.0, hold_s=1.0) -> TrialMetrics:
    return TrialMetrics(by_concurrency={str(m.concurrency): m for m in lv}, requests=sum(m.requests for m in lv),
                        output_tokens=sum(m.output_tokens for m in lv), idle_power_w=idle_w, hold_joules=hold_j,
                        hold_s=hold_s)


# W = c / R: 2 s at c=1, 2.5 s at c=4, 3.33 s at c=8.
THREE = trial(level(1, 0.5, 200.0), level(4, 1.6, 300.0), level(8, 2.4, 350.0))


def test_at_a_low_rate_idle_dominates_and_the_parts_sum_to_the_total():
    c = compose(THREE, 0.01)
    assert c.total_j == pytest.approx(c.idle_j + c.active_j + c.hold_j)
    assert c.idle_j == pytest.approx(60.0 / 0.01)  # the baseline over a 100 s gap
    assert c.idle_j / c.total_j > 0.95 and c.feasible and not c.fallback
    # n = 0.02: busy 2% of the time, almost all of it with one request at 200 W, 140 W above idle.
    assert c.active_j == pytest.approx((1 - math.exp(-0.02)) * 140.0 / 0.01, rel=0.01)


def test_the_hold_is_paid_when_the_server_empties():
    assert hold_factor(1e-6, 1.0) == pytest.approx(1.0, abs=1e-5)
    assert hold_factor(1.0, 0.0) == 1.0
    assert hold_factor(1000.0, 1.0) == pytest.approx(0.001, rel=1e-3)
    quiet = compose(trial(level(1, 0.5, 200.0), level(8, 2.4, 350.0), hold_s=1e-9), 0.01)
    assert quiet.hold_j == pytest.approx(math.exp(-quiet.in_flight) * 30.0)  # n = 0.02: almost every arrival
    assert quiet.hold_j == pytest.approx(30.0, rel=0.03)
    busy = compose(trial(level(1, 0.5, 200.0), level(8, 2.4, 350.0), hold_s=1e-9), 1.9)
    assert busy.in_flight > 5 and busy.hold_j < 0.01 * 30.0  # the server almost never empties


def test_the_operating_point_lands_between_levels():
    pts = levels(THREE, 60.0)
    n = in_flight(pts, 1.0)
    # n = 1 x W(n) with W linear from 2 s at c=1 to 2.5 s at c=4: n = 2.2.
    assert n == pytest.approx(2.2)
    c = compose(THREE, 1.0)
    assert c.in_flight == pytest.approx(2.2)
    assert c.active_j == pytest.approx(mean_power(pts, 2.2, 60.0) - 60.0)  # the mean power above idle, over 1 req/s
    assert serving_energy(pts, 1.0, 60.0) == pytest.approx(pts[0].per_request_j)  # exact at a measured level
    assert serving_energy(pts, 4.0, 60.0) == pytest.approx(pts[1].per_request_j)


def test_latency_is_judged_on_every_level_up_to_what_95_percent_of_arrivals_are_served_with():
    assert [poisson_quantile(m, 0.95) for m in (0.0, 0.05, 0.57, 2.2)] == [0, 0, 2, 5]
    c = compose(THREE, 1.0)  # 2.2 in flight: an arrival meets 5 or fewer others 95% of the time, so up to 6 served
    assert c.latency_in_flight == 6 and [m.concurrency for m in c.latency_levels] == [4, 8]
    quiet = compose(THREE, 0.3)  # 0.6 in flight: meets 2 or more others about 1 time in 8, so up to 3 served
    assert quiet.latency_in_flight == 3 and [m.concurrency for m in quiet.latency_levels] == [1, 4]
    alone = compose(THREE, 0.01)  # 0.02 in flight: an arrival almost always has the server to itself
    assert alone.latency_in_flight == 1 and [m.concurrency for m in alone.latency_levels] == [1]


def test_a_rate_whose_latency_would_need_an_unmeasured_level_is_not_eligible():
    c = compose(THREE, 1.9)  # sustainable (1.9 < 0.8 x 2.4), but 5.2 in flight means arrivals meet up to 10 others
    assert not c.unsustainable and c.latency_beyond_top and not c.feasible
    assert "more than the busiest level measured" in c.note()


def test_a_rate_the_measured_levels_cannot_sustain_is_infeasible():
    c = compose(THREE, 2.0)  # 0.8 x 2.4 = 1.92 req/s is the most the model will vouch for
    assert not c.feasible and c.unsustainable and c.violation == pytest.approx(2.0 - 0.8 * 2.4)
    assert "not sustainable" in c.note()
    beyond = compose(THREE, 3.0)
    assert beyond.in_flight == 8 and beyond.latency_beyond_top


def test_throughput_that_falls_under_load_caps_the_rate():
    # 4 users serve 2.0 req/s and 8 users only 1.8: at n = 3, arrivals are served with up to 7, so the 8-user level
    # is the one covering that, and its 1.8 binds.
    sagging = trial(level(1, 0.5, 200.0), level(4, 2.0, 300.0), level(8, 1.8, 330.0))
    c = compose(sagging, 1.5)
    assert c.in_flight == pytest.approx(3.0) and c.utilisation == pytest.approx(1.5 / 1.8) and c.unsustainable
    low = compose(sagging, 0.3)  # n = 0.6: served with up to 3, covered by 4 users; 4 and 8 count, 1 does not
    assert low.utilisation == pytest.approx(0.3 / 1.8) and not low.unsustainable


def test_single_request_throughput_is_not_a_ceiling():
    """Below one request in flight on average, a rate near the 1-user level's own rate is fine: arrivals that
    overlap are batched. Counting that level as capacity made 0.8-1.0 of its rate unsustainable and 1.2 fine."""
    for lam in (0.42, 0.48, 0.6):  # 0.84x, 0.96x and 1.2x the 1-user level's 0.5 req/s
        c = compose(THREE, lam)
        assert not c.unsustainable, lam


def test_shorter_answers_do_not_make_cheaper_requests():
    """Natural-stop answers differ in length between configurations; counted in one request size, a
    configuration that answers in half the tokens at the same energy per token gains nothing."""
    long = trial(level(1, 0.5, 200.0, tokens=500), level(8, 2.4, 350.0, tokens=500))
    short = trial(level(1, 1.0, 200.0, tokens=250), level(8, 4.8, 350.0, tokens=250))  # same tokens/s and watts
    own = compose(short, 0.1).active_j, compose(long, 0.1).active_j
    assert own[0] < own[1] * 0.6  # each judged by its own answers, the short one looks far cheaper
    shared = compose(short, 0.1, tokens_per_request=500), compose(long, 0.1, tokens_per_request=500)
    assert shared[0].total_j == pytest.approx(shared[1].total_j)
    assert shared[0].in_flight == pytest.approx(shared[1].in_flight)
    assert "of 500 tokens" in shared[0].note()


def test_without_idle_power_it_falls_back_to_joules_per_token_and_says_so():
    c = compose(trial(level(1, 0.5, 200.0), level(8, 2.4, 350.0), idle_w=None), 0.1)
    assert c.fallback and c.idle_j == 0.0 and c.hold_j == 0.0
    assert c.total_j == pytest.approx(level(1, 0.5, 200.0).joules_per_token * 500)
    assert "no idle power" in c.note()


def test_a_rate_below_the_lowest_level_uses_that_level_and_notes_it():
    c = compose(trial(level(4, 1.6, 300.0), level(8, 2.4, 350.0)), 0.1)
    assert c.below_lowest and c.feasible and c.operating.concurrency == 4
    assert "below the lowest level" in c.note()
    assert not compose(THREE, 0.1).below_lowest  # c=1 is a lone request: nothing below it to miss


def test_a_trial_that_skipped_the_workloads_low_levels_cannot_be_judged_below_them():
    stopped = trial(level(8, 2.4, 350.0))  # measured busiest first and stopped there
    c = compose(stopped, 0.1, workload_lowest=1)
    assert c.missing_low_levels and not c.feasible and c.violation > 0
    assert "not eligible" in c.note()
    assert compose(stopped, 0.1).feasible  # without the workload's levels to compare against, it is only noted
    assert not compose(THREE, 0.1, workload_lowest=1).missing_low_levels


def test_the_counter_is_preferred_over_the_sampled_integral():
    lv = level(1, 0.5, 200.0)
    lv.counter_energy_j = 3000.0  # the counter read 3000 J where the samples integrate to 4000
    c = compose(trial(lv), 0.01)
    # 150 W while serving by the counter (3000 J over 20 s), not the samples' 200 W.
    assert c.active_j == pytest.approx((1 - math.exp(-0.02)) * (150.0 - 60.0) / 0.01, rel=1e-6)


def test_no_energy_at_all_gives_no_total():
    lv = TrialMetrics(concurrency=1, requests=10, output_tokens=5000, duration_s=20.0)
    assert compose(TrialMetrics(by_concurrency={"1": lv}), 0.1).total_j is None
    with pytest.raises(ValueError):
        compose(THREE, 0.0)


# --------------------------------------------------------------------------- on recorded calibrations

RECORDED = ("benchmarks/strategies/results-dolly/profiles/balanced-file-dolly-d03d5896.json",  # A40, Qwen2.5-3B
            "benchmarks/strategies/results-llama/profiles/4090/balanced-file-dolly-d03d5896-levels-all-ttft-p95.json")


@pytest.mark.parametrize("path", RECORDED)
def test_a_left_out_level_is_predicted_from_its_neighbours(path):
    """Leave the 4-user level out and predict its serving energy per request from the 1- and 8-user levels, with
    60 W assumed idle: interpolating power and time in the server lands within 10% at the median, where
    interpolating energy per request straight across overshot by 2.2-2.5x on the same calibrations."""
    import json
    import statistics
    from pathlib import Path

    from rigtune.models import TrialResult

    data = json.loads((Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8"))
    ratios, straight = [], []
    for row in data["calibration_table"]:
        t = TrialResult.model_validate(row)
        lv = t.metrics.by_concurrency
        if not t.ok or not all(k in lv and lv[k].ok for k in ("1", "4", "8")):
            continue
        pts = levels(t.metrics, 60.0, 500.0)
        actual = next(p for p in pts if p.concurrency == 4).per_request_j
        ends = [p for p in pts if p.concurrency != 4]
        ratios.append(serving_energy(ends, 4.0, 60.0) / actual)
        straight.append((ends[0].per_request_j + 3 / 7 * (ends[1].per_request_j - ends[0].per_request_j)) / actual)
    assert len(ratios) >= 10
    assert 0.9 <= statistics.median(ratios) <= 1.1
    assert statistics.median(straight) > 2.0  # the interpolation the model no longer uses


def test_mean_power_is_averaged_over_the_poisson_count_in_flight():
    """Worked by hand: every level draws 300 W and keeps a request 2 s (W flat), so 0.85 req/s puts n = 1.7 in
    flight. The server is empty a share e^-1.7 = 0.1827 of the time, at the 60 W idle, and busy at 300 W otherwise:
    P_mean = 0.1827 x 60 + 0.8173 x 300 = 256.16 W, so 70.59 J/request of idle and 230.78 of serving."""
    flat = trial(level(1, 0.5, 300.0), level(4, 2.0, 300.0), level(8, 4.0, 300.0), hold_j=0.0)
    c = compose(flat, 0.85)
    empty = math.exp(-1.7)
    assert empty == pytest.approx(0.1827, abs=1e-4)
    assert c.in_flight == pytest.approx(1.7)
    assert c.idle_j == pytest.approx(60.0 / 0.85)
    assert c.active_j == pytest.approx((empty * 60.0 + (1 - empty) * 300.0 - 60.0) / 0.85)
    assert c.active_j == pytest.approx(230.78, abs=0.01) and c.total_j == pytest.approx(301.36, abs=0.01)
    # Run as if always at 1.7 in flight, the same trial would never idle: 300 W / 0.85 = 352.9 J/request.
    assert 300.0 / 0.85 - c.total_j == pytest.approx(empty * 240.0 / 0.85)
