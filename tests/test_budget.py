"""--budget: calibration stops starting trials once the next would end past the budget."""

from __future__ import annotations

import pytest
import typer

from rigtune.calibrate.search import StagedSearch
from rigtune.cli import _duration
from rigtune.models import Config, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions

FEASIBLE = [Config(backend="llamacpp-cuda", quant=q, ctx=4096, batch=b, n_gpu_layers=37, n_batch=512)
            for q in ("Q4_K_M", "Q8_0") for b in (1, 4, 8)]


class ClockRunner:
    """Every trial takes 60 s on a fake clock; throughput grows with the batch."""

    def __init__(self):
        self.now, self.ran = 0.0, []

    def clock(self) -> float:
        return self.now

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.now += 60.0
        self.ran.append(cfg)
        m = TrialMetrics(tok_s=100.0 + cfg.batch, ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048,
                         concurrency=8)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


def _search(budget, **kw):
    runner = ClockRunner()
    return runner, StagedSearch(objective="throughput", runner=runner, budget_s=budget, clock=runner.clock, **kw)


def test_the_budget_holds_back_the_picks_re_measurement():
    # On an A100 the hour ran out in the tuning stages and the pick was never re-measured. Here 300 s fits five
    # 60 s trials; the search stops after four, so that the pick is re-measured and chosen on a fresh run.
    runner, search = _search(300, confirm_top=3)
    winner, _ = search.run(FEASIBLE)
    stages = [r.stage for r in search.results]
    assert stages.count("confirm") == 1 and len(stages) == 5
    assert winner is not None and winner.stage == "confirm"
    assert runner.now <= 300
    # The note counts the re-measurement that ran, not the shortlist (the Modal rerun said "the best 3").
    assert any(n.startswith("re-measured the best configuration of 3 shortlisted (the budget ran out) and chose")
               for n in search.notes)


def test_each_confirm_round_is_held_back():
    runner, search = _search(300, confirm_top=3, confirm_rounds=2)
    winner, _ = search.run(FEASIBLE)
    stages = [r.stage for r in search.results]
    assert stages.count("confirm") == 2 and len(stages) == 5  # two rounds' time, spent on the shortlist in order
    assert winner is not None and winner.stage == "confirm" and runner.now <= 300


def test_a_short_budget_still_reaches_the_speculation_trials():
    """The timings of a 20-minute calibration of Qwen3.5-9B on an A100: a cold first start of about 5 minutes, then
    about 2 minutes a trial. Holding back three re-measurements would skip the speculation trials; holding back the
    pick's one runs them, and MTP at 3 tokens wins."""
    base = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.95)
    speed = {None: 607.0, "ngram_gpu:4": 535.0, "mtp:1": 880.0, "mtp:3": 1030.0}

    class Runner(ClockRunner):
        def run(self, cfg, stage):
            self.now += 300.0 if not self.ran else 130.0
            self.ran.append(cfg)
            m = TrialMetrics(tok_s=speed[cfg.spec_decode] * (0.995 if cfg.kv_dtype != "auto" else 1.0),
                             ttft_ms=150, tpot_ms=7, requests=16, output_tokens=2048, concurrency=8)
            return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))

    def kv(c):
        return [] if c.kv_dtype != "auto" else [c.model_copy(update={"kv_dtype": "fp8_e5m2"})]

    def spec(c):
        return [] if c.spec_decode else [c.model_copy(update={"spec_decode": s}) for s in ("ngram_gpu:4", "mtp:1",
                                                                                         "mtp:3")]

    runner = Runner()
    search = StagedSearch(objective="throughput", runner=runner, budget_s=1200, clock=runner.clock, confirm_top=3,
                          variant_stages=[("kv", kv), ("spec", spec)])
    winner, _ = search.run([base])
    assert [c.spec_decode for c in runner.ran].count("mtp:3") >= 1
    assert winner is not None and winner.config.spec_decode == "mtp:3" and winner.stage == "confirm"
    assert runner.now <= 1200


def test_the_budget_stops_new_trials_and_the_best_so_far_wins():
    runner, search = _search(150)
    winner, notes = search.run(FEASIBLE)
    assert len(runner.ran) == 2  # 60 s each: a third would end at 180 s
    assert winner is not None and winner.config in runner.ran
    assert all(r.launched for r in search.results)  # skipped trials are not results
    assert any("budget of 150s reached after 2 trials" in n for n in notes)


def test_one_trial_always_runs_and_no_budget_runs_everything():
    runner, search = _search(1)
    winner, _ = search.run(FEASIBLE)
    assert len(runner.ran) == 1 and winner is not None
    runner, search = _search(None)
    _, notes = search.run(FEASIBLE)
    assert len(runner.ran) > 2 and not any("budget" in n for n in notes)


def test_a_budgeted_profile_is_cached_under_its_own_name():
    assert SearchOptions(budget_s=600).key() == {"budget": "600s"} and SearchOptions().key() == {}


def test_duration_parsing():
    assert [_duration(v) for v in ("90s", "10m", "1h", "45", None)] == [90, 600, 3600, 45, None]
    for bad in ("abc", "0", "10d"):
        with pytest.raises(typer.BadParameter):
            _duration(bad)
