"""The rate check: the pick measured once with Poisson arrivals at the rate it was scored for."""

from __future__ import annotations

import asyncio

import pytest
import typer

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.measure import arrival_schedule, run_trial
from rigtune.calibrate.objectives import Constraints, pick
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import Workload
from rigtune.cli import app
from rigtune.models import Config, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions
from tests.test_measure_and_proxy import upstream  # noqa: F401  (the fake engine, as a fixture)
from tests.test_objectives_and_search import FakeRunner


def test_the_schedule_is_seeded_and_its_window_is_exactly_n_over_the_rate():
    times, window = arrival_schedule(0.5, 24, seed=3)
    assert (times, window) == arrival_schedule(0.5, 24, seed=3)
    assert times != arrival_schedule(0.5, 24, seed=4)[0]
    assert len(times) == 24 and times[0] == 0.0 and times == sorted(times)
    assert window == pytest.approx(48.0) and times[-1] < window


def test_a_slow_rate_is_capped_at_five_minutes_and_a_fast_one_lasts_twenty_seconds():
    times, window = arrival_schedule(0.05, 24)
    assert len(times) == 15 and window == pytest.approx(300.0)  # 24 at 0.05/s would take 8 minutes
    times, window = arrival_schedule(1 / 300, 24)
    assert len(times) == 1 and window == pytest.approx(300.0)  # the lowest rate it can check: never past the cap
    times, window = arrival_schedule(5.0, 24)
    assert len(times) == 100 and window == pytest.approx(20.0)  # 24 at 5/s would be over in 4.8 s
    for too_slow in (0.0, 0.001, 1 / 301):
        with pytest.raises(ValueError, match="1/300"):
            arrival_schedule(too_slow, 4)


def test_open_loop_prompts_are_not_repeated_while_the_workload_has_more():
    wl = Workload(n_prompts=4, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4)).ensure_prompts()
    got = M._open_loop_prompts(wl, 10, TokenCounter())
    assert len(got) == 10 and len(set(got)) == 10 and got[:4] == wl.prompts


def test_open_loop_requests_against_a_fake_engine(upstream, monkeypatch):  # noqa: F811
    srv, app_ = upstream
    monkeypatch.setattr(M, "OPEN_LOOP_MIN_S", 0.0)  # six quick arrivals, not twenty seconds of them
    monkeypatch.setattr(M, "OPEN_LOOP_REQUESTS", 6)
    wl = Workload(n_prompts=6, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4))
    times, window = arrival_schedule(40.0, 6, seed=1, min_s=0.0)
    outcomes, wall, peak = asyncio.run(M._drive_open(srv.url, LlmtraceHooks(model_name="fake-model"), wl.ensure_prompts(),
                                                     times, window, 30.0))
    assert len(outcomes) == 6 and all(o.ok for o in outcomes) and wall >= window and peak >= 1

    m = run_trial(srv.url, LlmtraceHooks(model_name="fake-model", gpu_ids=[]), wl, warmup=False, arrival_rate=40.0)
    assert m.open_loop and m.arrival_rate_rps == 40.0 and m.requests == 6 and m.ok
    assert m.by_concurrency == {}  # one run, not levels


def test_joules_per_request_is_the_whole_window_over_its_requests(monkeypatch):
    class Tel:
        def __init__(self, hooks, pid=None, interval_ms=100):
            self.summary = M.TelemetrySummary(source="pynvml", power_w=90.0, energy_j=1100.0, counter_energy_j=1000.0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    async def drive(base_url, hooks, workload, times, window, timeout, counter=None):
        return [M.RequestOutcome(ok=True, ttft_s=0.05, duration_s=1.0, tokens=50, token_source="usage")] * len(times), \
            window, 1

    monkeypatch.setattr(M, "Telemetry", Tel)
    monkeypatch.setattr(M, "_drive_open", drive)
    monkeypatch.setattr(M, "OPEN_LOOP_REQUESTS", 10)
    wl = Workload(n_prompts=10, prefill_tokens=16, decode_tokens=4)
    m = run_trial("http://x", LlmtraceHooks(gpu_ids=[0]), wl, warmup=False, arrival_rate=0.5)
    assert m.joules_per_request == 100.0 and m.energy_source == "nvml_counter"  # 1000 J over 10 requests
    assert m.duration_s == pytest.approx(20.0)  # 10 arrivals at 0.5/s


def _open_loop_result(cfg: Config) -> TrialResult:
    return TrialResult(config=cfg, stage="rate-check", metrics=TrialMetrics(
        open_loop=True, arrival_rate_rps=1.0, tok_s=99999.0, requests=24, output_tokens=24 * 128, ttft_ms=50.0,
        ttft_p95_ms=120.0, joules_per_request=500.0, joules_per_token=500.0 / 128, duration_s=24.0, concurrency=2))


def test_the_check_is_never_part_of_the_choice():
    plain = TrialResult(config=Config(backend="vllm", quant="bf16", batch=16), stage="t",
                        metrics=TrialMetrics(tok_s=100.0, ttft_ms=50.0, requests=16, output_tokens=2048))
    check = _open_loop_result(Config(backend="vllm", quant="bf16", batch=64))
    assert pick([plain, check], "throughput")[0] is plain  # 99999 tok/s, and ignored


class RateRunner(FakeRunner):
    """FakeRunner trials with a duration and an idle tail, so the rate model has something to compose."""

    def __init__(self):
        super().__init__()
        self.open_loop = []

    def run(self, cfg, stage):
        res = super().run(cfg, stage)
        if res.ok:
            res.metrics = res.metrics.model_copy(update={"duration_s": 10.0, "concurrency": 8, "idle_power_w": 60.0,
                                                         "hold_joules": 20.0, "hold_s": 1.0})
        return res

    def run_open_loop(self, cfg, rate, stage="rate-check"):
        self.open_loop.append((cfg.key(), rate))
        return _open_loop_result(cfg)


def _search(runner, **kw):
    cfgs = [Config(backend="vllm", quant=q, ctx=4096, batch=b, gpu_memory_utilization=0.9)
            for q in ("bf16", "fp8") for b in (16, 64)]
    search = StagedSearch(objective="efficiency", runner=runner,
                          constraints=Constraints(arrival_rate=1.0, ttft_ceiling_ms=5000), **kw)
    return search, search.run(cfgs)


def test_the_pick_is_checked_once_at_its_rate():
    runner = RateRunner()
    search, (winner, notes) = _search(runner)
    assert runner.open_loop == [(winner.config.key(), 1.0)]
    rows = [r for r in search.results if r.stage == "rate-check"]
    assert len(rows) == 1 and rows[0].metrics.open_loop
    note = next(n for n in notes if n.startswith("rate check"))
    assert "at 1 req/s over 24 Poisson arrivals (24 s)" in note
    assert "measured 500.0 J/request at 128 tokens/request" in note
    # The prediction is recomputed for requests as long as the check's own (128 tokens), not scaled per token.
    assert "J/request at those 128 tokens/request, measured" in note
    assert "p95 TTFT 120 ms, from 24 requests, so close to their slowest" in note

    quiet = RateRunner()
    unchecked, (same, notes) = _search(quiet, rate_check=False)
    assert quiet.open_loop == [] and same.config.key() == winner.config.key()
    assert not any(r.stage == "rate-check" for r in unchecked.results)


def test_no_check_without_a_rate_or_under_other_objectives():
    runner = RateRunner()
    StagedSearch(objective="efficiency", runner=runner).run(
        [Config(backend="vllm", quant="bf16", batch=16, gpu_memory_utilization=0.9)])
    StagedSearch(objective="throughput", runner=runner, constraints=Constraints(arrival_rate=1.0)).run(
        [Config(backend="vllm", quant="bf16", batch=16, gpu_memory_utilization=0.9)])
    assert runner.open_loop == []


def test_turning_the_check_off_is_cached_apart_and_every_command_takes_it():
    assert SearchOptions(arrival_rate=0.5, rate_check=False).key("efficiency") == {"rate": "0.5rps",
                                                                                   "ratecheck": "off"}
    assert SearchOptions(rate_check=False).key("efficiency") == {}  # meaningless without a rate
    commands = typer.main.get_command(app).commands
    for command in ("bench", "recalibrate", "compare", "serve"):
        assert any("--rate-check" in p.opts for p in commands[command].params), command


def test_a_rate_too_slow_to_check_is_skipped_with_a_note():
    runner = RateRunner()
    search = StagedSearch(objective="efficiency", runner=runner,
                          constraints=Constraints(arrival_rate=0.001, ttft_ceiling_ms=5000))
    _, notes = search.run([Config(backend="vllm", quant="bf16", batch=16, gpu_memory_utilization=0.9)])
    assert runner.open_loop == [] and any("rate check skipped" in n and "300 s" in n for n in notes)


def test_the_reference_rows_of_later_stages_are_never_the_rate_check(hw_a100, spec, prepared_vllm):
    from rigtune.disagg import _unified_reference
    from rigtune.layout import _single_reference
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)
    trial = TrialResult(config=p.config, stage="confirm",
                        metrics=TrialMetrics(tok_s=100.0, ttft_ms=50.0, requests=16, output_tokens=2048))
    p.calibration_table = [trial, _open_loop_result(p.config)]  # same key, and 99999 tok/s
    assert _single_reference(p) is trial and _unified_reference(p) is trial


def test_the_prediction_is_recomputed_at_the_checks_own_answer_length():
    """P_idle / λ is paid per request whatever its length, so a prediction in another request size, scaled per
    token, would mis-state what the check's requests should cost."""
    from rigtune.calibrate.objectives import rank
    from rigtune.calibrate.rate import compose

    class ShortAnswers(RateRunner):  # the check's answers run to 64 tokens, half the calibration's 128
        def run_open_loop(self, cfg, rate, stage="rate-check"):
            res = super().run_open_loop(cfg, rate, stage)
            res.metrics = res.metrics.model_copy(update={"output_tokens": 24 * 64})
            return res

    search, (winner, notes) = _search(ShortAnswers())
    comp = next(x for x in rank(search.results, "efficiency", search.constraints) if x.result is winner).rate
    assert comp.tokens_per_request == 128
    want = compose(winner.metrics, 1.0, tokens_per_request=64, idle_w=comp.idle_w).total_j
    note = next(n for n in notes if n.startswith("rate check"))
    assert f"predicted {want:.1f} J/request at those 64 tokens/request" in note
    scaled = comp.total_j / comp.tokens_per_request * 64  # the per-token scaling this replaces
    assert scaled < want - 1.0  # it would halve the idle term too
