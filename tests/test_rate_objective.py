"""`efficiency` at --arrival-rate: joules per request at the stated rate, and nothing changes without one."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from rigtune.calibrate.objectives import Constraints, pick, rank
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.cli import _check_rate, _opts, _rate, app
from rigtune.models import Config, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, calibrate, idle_tail_for
from tests.test_objectives_and_search import RESULTS
from tests.test_rate_model import level


def _config(batch: int, levels, idle_w=60.0) -> TrialResult:
    by = {str(m.concurrency): m for m in levels}
    top = max(levels, key=lambda m: m.tok_s)
    m = TrialMetrics(tok_s=top.tok_s, ttft_ms=top.ttft_ms, ttft_p95_ms=top.ttft_p95_ms, concurrency=top.concurrency,
                     requests=sum(x.requests for x in levels), output_tokens=sum(x.output_tokens for x in levels),
                     idle_power_w=idle_w, hold_joules=20.0, hold_s=1.0, by_concurrency=by)
    return TrialResult(config=Config(backend="vllm", quant="bf16", batch=batch, gpu_memory_utilization=0.9),
                       stage="t", metrics=m)


# A spends less above idle on a lone request, B much less once several are in flight. Same capacity, same idle.
A = _config(16, [level(1, 0.5, 150.0), level(4, 1.6, 300.0), level(8, 2.4, 400.0)])
B = _config(64, [level(1, 0.5, 200.0), level(4, 1.6, 220.0), level(8, 2.4, 280.0)])


def test_the_pick_follows_the_rate():
    quiet, _ = pick([A, B], "efficiency", Constraints(arrival_rate=0.05))
    busy, notes = pick([A, B], "efficiency", Constraints(arrival_rate=1.8))
    assert quiet is A and busy is B
    note = next(n for n in notes if "J/request" in n)
    assert "at 1.8 req/s" in note and "idle" in note and "serving" in note and "wind-down" in note
    assert "across the 2 configurations measured" in note


def test_at_a_low_rate_the_note_says_idle_is_beyond_any_setting():
    _, notes = pick([A, B], "efficiency", Constraints(arrival_rate=0.05))
    assert any("no configuration changes" in n for n in notes)


def test_a_rate_past_every_trial_s_capacity_is_relaxed_and_flagged():
    winner, notes = pick([A, B], "efficiency", Constraints(arrival_rate=5.0))
    assert winner is not None and any("least-violating" in n for n in notes)
    assert all(not x.feasible for x in rank([A, B], "efficiency", Constraints(arrival_rate=5.0)))


def test_the_latency_ceiling_is_judged_at_the_operating_level():
    slow = _config(16, [level(1, 0.5, 150.0, p95=100.0), level(4, 1.6, 300.0, p95=100.0),
                        level(8, 2.4, 400.0, p95=900.0)])
    ranked = rank([slow], "efficiency", Constraints(arrival_rate=1.8, ttft_ceiling_ms=500))
    assert ranked[0].metrics.concurrency == 8 and not ranked[0].feasible  # ~4.8 in flight, judged at 8
    assert rank([slow], "efficiency", Constraints(arrival_rate=0.05, ttft_ceiling_ms=500))[0].feasible


def test_answering_shorter_does_not_win():
    # S spends the same energy per token as B at every level but stops its answers at half the length.
    s = _config(32, [level(1, 1.0, 200.0, tokens=250), level(4, 3.2, 220.0, tokens=250),
                     level(8, 4.8, 280.0, tokens=250)])
    ranked = rank([A, B, s], "efficiency", Constraints(arrival_rate=0.05))
    by = {x.result.config.batch: x for x in ranked}
    assert by[32].score == pytest.approx(by[64].score)  # in 500-token requests, S and B cost the same
    assert all(x.rate.tokens_per_request == 500 for x in ranked)  # the median of 500, 500, 250 at 1 user
    _, notes = pick([A, B, s], "efficiency", Constraints(arrival_rate=0.05))
    assert any("requests of 500 tokens, the median answer at 1 users across 3 trials" in n for n in notes)


def test_a_trial_without_idle_data_cannot_win_by_leaving_idle_out():
    """Replica trials record no idle tail; with `--layout auto` at a low rate their fallback J/request, which
    leaves out ~P_idle / λ, would otherwise beat the single engine that counts it."""
    replicas = _config(16, [level(1, 1.0, 300.0), level(4, 3.2, 400.0), level(8, 4.8, 500.0)], idle_w=None)
    replicas.replicas = 2
    winner, notes = pick([A, replicas], "efficiency", Constraints(arrival_rate=0.05))
    assert winner is A
    assert any("1 configuration was measured without an idle tail" in n for n in notes)
    ranked = rank([A, replicas], "efficiency", Constraints(arrival_rate=0.05))
    assert ranked[1].result is replicas and not ranked[1].feasible


def test_when_no_trial_has_idle_data_all_use_the_fallback_alike():
    a = _config(16, [level(1, 0.5, 150.0), level(4, 1.6, 300.0), level(8, 2.4, 400.0)], idle_w=None)
    b = _config(64, [level(1, 0.5, 200.0), level(4, 1.6, 220.0), level(8, 2.4, 280.0)], idle_w=None)
    ranked = rank([a, b], "efficiency", Constraints(arrival_rate=0.05))
    assert all(x.feasible and x.rate.fallback for x in ranked) and ranked[0].result is a


def test_latency_is_judged_where_95_percent_of_arrivals_find_the_server():
    # Fine alone, slow with four in flight: at 0.3 req/s it averages 0.6 in flight, but 2 or more 1 time in 8.
    tail = _config(16, [level(1, 0.5, 150.0, p95=200.0), level(4, 1.6, 300.0, p95=1371.0),
                        level(8, 2.4, 400.0, p95=2000.0)])
    at = rank([tail], "efficiency", Constraints(arrival_rate=0.3, ttft_ceiling_ms=1000))[0]
    assert at.rate.in_flight == pytest.approx(0.6) and at.metrics.concurrency == 4 and not at.feasible
    assert rank([tail], "efficiency", Constraints(arrival_rate=0.01, ttft_ceiling_ms=1000))[0].feasible


def test_without_a_rate_nothing_changes():
    for objective in ("throughput", "latency", "balanced", "efficiency"):
        plain = [(x.result.config.key(), x.score, x.feasible) for x in rank(RESULTS, objective, Constraints())]
        explicit = [(x.result.config.key(), x.score, x.feasible)
                    for x in rank(RESULTS, objective, Constraints(arrival_rate=None))]
        assert plain == explicit
    assert pick(RESULTS, "efficiency", Constraints(tok_s_floor_abs=1000))[0].config.batch == 256
    assert all(x.rate is None for x in rank(RESULTS, "efficiency"))
    # A rate changes nothing under the other objectives either: they never read it.
    for objective in ("throughput", "latency", "balanced"):
        assert ([x.score for x in rank(RESULTS, objective)]
                == [x.score for x in rank(RESULTS, objective, Constraints(arrival_rate=0.5))])


def test_the_cache_key_moves_only_with_a_rate():
    assert SearchOptions().key() == {} and SearchOptions().key("efficiency") == {}
    assert SearchOptions(arrival_rate=0.5).key("efficiency") == {"rate": "0.5rps"}
    assert SearchOptions(arrival_rate=0.2).key("efficiency") == {"rate": "0.2rps"}
    assert idle_tail_for("efficiency", SearchOptions(arrival_rate=0.5)) == 6.0


def test_rates_parse_in_the_units_people_use():
    assert _rate(None) is None
    assert _rate("0.5") == 0.5 and _rate("2/s") == 2.0
    assert _rate("12/min") == pytest.approx(0.2) and _rate("720/h") == pytest.approx(0.2)
    for bad in ("0", "-1", "fast", "3/day"):
        with pytest.raises(typer.BadParameter):
            _rate(bad)
    assert _opts("auto", "on", "on", "on", arrival_rate=0.5).arrival_rate == 0.5
    assert _opts("auto", "on", "on", "on").arrival_rate is None


def test_a_rate_is_refused_under_other_objectives():
    _check_rate("efficiency", 0.5)
    _check_rate("balanced", None)
    with pytest.raises(typer.BadParameter):
        _check_rate("balanced", 0.5)
    result = CliRunner().invoke(app, ["bench", "some/model", "--objective", "balanced", "--arrival-rate", "0.5"])
    assert result.exit_code == 2  # refused before anything is probed or launched
    with pytest.raises(RuntimeError, match="efficiency only"):
        calibrate(None, None, "latency", None, {}, options=SearchOptions(arrival_rate=0.5))


def test_every_calibrating_command_takes_the_flag():
    commands = typer.main.get_command(app).commands
    for command in ("bench", "recalibrate", "compare", "serve"):
        assert any("--arrival-rate" in p.opts for p in commands[command].params), command


def test_the_workload_records_the_rate_only_when_there_is_one():
    assert "arrival_rate_rps" not in get_workload("chat").spec()
    wl = Workload(name="chat", arrival_rate_rps=0.5)
    assert wl.spec()["arrival_rate_rps"] == 0.5 and "0.5 requests/s" in wl.describe()
    assert "requests/s" not in get_workload("chat").describe()


def test_idle_is_pooled_per_engine_and_a_config_differs_only_beyond_the_spread():
    from rigtune.calibrate.objectives import idle_pools, pooled_idle

    lv = [level(1, 0.5, 150.0), level(4, 1.6, 300.0), level(8, 2.4, 400.0)]
    trials = [_config(b, lv, idle_w=w) for b, w in ((16, 59.0), (32, 61.0), (64, 60.0), (128, 75.0))]
    pools = idle_pools(trials)
    pool = pools[("vllm", None, None)]
    assert (pool.median_w, pool.low_w, pool.high_w, pool.trials) == (60.5, 59.0, 75.0, 4)
    assert pooled_idle(trials[0], pools) == 60.5 and pooled_idle(trials[1], pools) == 60.5  # within the band
    assert pooled_idle(trials[3], pools) == 75.0  # 14.5 W off: its own idle, not noise
    _, notes = pick(trials, "efficiency", Constraints(arrival_rate=0.05))
    assert any(n.startswith("idle power pooled per engine: vllm 60.5 W, the median of 4 trials (59.0-75.0") for n in notes)


def test_with_nothing_feasible_sustainability_comes_before_latency():
    """Requests per second and milliseconds do not add: a trial far past its latency ceiling but sustainable must
    rank before one barely past sustainability, whatever the sum of the two would say."""
    slow = _config(16, [level(1, 0.5, 150.0, p95=5000.0), level(4, 1.6, 300.0, p95=5000.0),
                        level(8, 2.4, 400.0, p95=5000.0)])  # sustains 0.3 req/s; its latency is far over
    small = _config(64, [level(1, 0.3, 150.0), level(4, 0.35, 160.0), level(8, 0.36, 170.0)])  # cannot sustain it
    ranked = rank([small, slow], "efficiency", Constraints(arrival_rate=0.3, ttft_ceiling_ms=500))
    assert not any(x.feasible for x in ranked)
    assert ranked[0].result is slow and ranked[0].violation_key[1] == 0.0 and ranked[0].violation_key[2] > 4000
    assert ranked[1].violation_key[1] > 0.0


def test_the_note_gives_the_winners_own_length_and_flags_output_changing_configs():
    # C answers in 1000 tokens where the others answer in 500: a different output, flagged rather than hidden.
    c = _config(256, [level(1, 0.25, 160.0, tokens=1000), level(4, 0.8, 310.0, tokens=1000),
                      level(8, 1.2, 410.0, tokens=1000)])
    winner, notes = pick([A, B, c], "efficiency", Constraints(arrival_rate=0.05, noise_tolerance=0))  # no tie-break
    note = next(n for n in notes if "J/request" in n and "idle" in n)
    assert "at its own answers' length, 500 tokens, that is" in note
    flag = next(n for n in notes if n.startswith("output-changing"))
    assert "1 configuration answers more than 25% longer or shorter than the median 500 tokens" in flag
    assert winner is A and "/b256/" in flag and "(1000 tokens)" in flag and "The winner is one" not in flag
