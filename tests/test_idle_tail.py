"""The idle tail: the card's idle draw and wind-down after a trial's last request, measured without moving any
number the trial already reports."""

from __future__ import annotations

import pytest

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.measure import Telemetry, _idle_summary, _Sample, _summarize
from rigtune.calibrate.objectives import Constraints, enough_level
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload
from rigtune.models import OBJECTIVES
from rigtune.pipeline import SearchOptions, idle_tail_for


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _decay(t0: float, tail_s: float, step: float = 0.1):
    """300 W until t0, a straight fall to 60 W over the next second, then 60 W: (t, W) after t0."""
    out = []
    for k in range(1, int(round(tail_s / step)) + 1):
        dt = k * step
        out.append((t0 + dt, 300.0 - 240.0 * min(1.0, dt)))
    return out


def test_idle_and_hold_from_a_scripted_trace():
    got = _idle_summary(_decay(10.0, 3.0), 10.0, 13.0, 300.0, counter_j=None)
    assert got.idle_power_w == pytest.approx(60.0) == got.idle_power_sampled_w
    assert got.hold_joules == pytest.approx(120.0)  # the triangle above idle: 240 W falling to 0 over 1 s
    assert got.hold_s == pytest.approx(0.9)  # within 10% of the 240 W drop
    assert not got.truncated and not got.below_idle and got.energy_source == "trapezoid"
    assert got.tail_s == pytest.approx(3.0)


def test_the_counter_gives_idle_and_hold_from_one_source():
    # The counter reads 62 W over the final second where the samples say 60, and 120 J above that over the tail.
    got = _idle_summary(_decay(10.0, 3.0), 10.0, 13.0, 300.0, counter_j=120.0 + 62.0 * 3.0, counter_idle_w=62.0)
    assert got.energy_source == "nvml_counter" and got.idle_power_w == 62.0
    assert got.idle_power_sampled_w == pytest.approx(60.0)  # both recorded
    assert got.hold_joules == pytest.approx(120.0)
    # Without the final second's own reading the counter is not mixed with sampled idle: the trapezoid stands.
    assert _idle_summary(_decay(10.0, 3.0), 10.0, 13.0, 300.0, counter_j=999.0).energy_source == "trapezoid"


def test_noise_is_not_clamped_into_a_hold():
    # Flat 60 W with +-5 W noise and no wind-down: clamping each point at idle would invent a hold.
    noisy = [(10.0 + 0.1 * k, 60.0 + (5.0 if k % 2 else -5.0)) for k in range(1, 31)]
    got = _idle_summary(noisy, 10.0, 13.0, 60.0, counter_j=None)
    # Clamped per point the noise alone would read as ~3 s x 2.5 W = 7.5 J; signed, only the idle estimate's own
    # error (59.5 W from the final second's 11 samples) is left.
    assert got.hold_joules < 2.5
    low = _idle_summary(_decay(10.0, 3.0), 10.0, 13.0, 300.0, counter_j=150.0, counter_idle_w=62.0)
    assert low.below_idle and low.hold_joules == 0.0  # 150 J < 62 W x 3 s: recorded as 0, and flagged


def test_a_tail_too_short_for_the_wind_down_is_flagged():
    got = _idle_summary(_decay(10.0, 1.0), 10.0, 11.0, 300.0, counter_j=None)
    assert got.truncated  # still falling through the last second, so its mean is not idle
    assert got.idle_power_w > 100


def test_too_few_samples_record_nothing():
    assert _idle_summary([(10.1, 80.0)], 10.0, 13.0, 300.0, None) is None


def test_the_mark_splits_the_window_and_leaves_the_active_part_alone():
    clock = Clock(100.0)
    active = [_Sample(t=100.0 + 0.1 * k, power_w=300.0, mem_mb=1000.0) for k in range(6)]
    tail = [_Sample(t=t, power_w=p, mem_mb=1000.0) for t, p in _decay(100.5, 3.0)]
    tel = Telemetry(LlmtraceHooks(gpu_ids=[]), clock=clock)
    tel._source, tel._samples = "pynvml", active + tail
    clock.t = 100.5
    tel.mark()
    clock.t = 103.5
    tel.stop()
    alone = _summarize(active, "pynvml")
    assert tel.summary.energy_j == alone.energy_j == pytest.approx(150.0)  # 300 W over 0.5 s, tail excluded
    assert tel.summary.power_w == alone.power_w == 300.0 and tel.summary.samples == 6
    assert tel.idle.idle_power_w == pytest.approx(60.0) and tel.idle.hold_joules == pytest.approx(120.0)
    assert tel.idle.tail_s == pytest.approx(3.0)


# --------------------------------------------------------------------------- run_trial

CLOCK = Clock()


class _ScriptedTelemetry(Telemetry):
    """The real Telemetry over a scripted trace: 300 W until the window is marked, 60 W after, sampled every
    0.1 s on a fake clock. A run_trial that let the tail into a level's numbers would read the 60 W."""

    marks = 0

    def __init__(self, hooks, pid=None, interval_ms=100):
        super().__init__(hooks, pid, interval_ms, clock=CLOCK)

    def start(self):
        self._source, self._t0 = "pynvml", CLOCK()

    def mark(self):
        type(self).marks += 1
        super().mark()

    def _stop_sampling(self):
        split = self._mark[0] if self._mark else float("inf")
        n = int(round((CLOCK() - self._t0) / 0.1))
        ts = [self._t0 + 0.1 * k for k in range(n + 1)]
        return [_Sample(t=t, power_w=300.0 if t <= split + 1e-9 else 60.0) for t in ts], False


def _drive(base_url, hooks, workload, concurrency, timeout, counter=None):
    async def run():
        CLOCK.t += 2.0  # every level keeps the card busy for 2 s
        t = {1: 0.05, 4: 0.1, 8: 0.9}[concurrency]
        return [M.RequestOutcome(ok=True, ttft_s=t, duration_s=t + 1.0, tokens=100, token_source="usage")] * 8
    return run()


def _trial(monkeypatch, **kw):
    _ScriptedTelemetry.marks = 0
    slept = []

    def sleep(s):
        slept.append(s)
        CLOCK.sleep(s)

    monkeypatch.setattr(M, "Telemetry", _ScriptedTelemetry)
    monkeypatch.setattr(M, "_drive", _drive)
    wl = Workload(n_prompts=8, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4, 8))
    m = M.run_trial("http://x", LlmtraceHooks(gpu_ids=[0]), wl, warmup=False, counter=TokenCounter(),
                    sleep=sleep, **kw)
    return m, slept


def test_no_tail_costs_no_sleep_and_records_nothing(monkeypatch):
    m, slept = _trial(monkeypatch)
    assert slept == [] and _ScriptedTelemetry.marks == 0
    assert m.idle_power_w is None and m.hold_joules is None and m.hold_after_concurrency is None


def test_the_tail_is_measured_once_after_the_last_level(monkeypatch):
    plain, _ = _trial(monkeypatch)
    m, slept = _trial(monkeypatch, idle_tail_s=3.0)
    assert slept == [2.0, 1.0] and _ScriptedTelemetry.marks == 1  # the final second is read on its own
    assert m.hold_after_concurrency == 8  # levels run 1, 4, 8 when every one is measured
    assert m.idle_power_w == pytest.approx(60.0) and m.idle_tail_s == pytest.approx(3.0)
    for c, level in plain.by_concurrency.items():  # per-level energy is what it was without the tail
        assert level.joules_per_token == pytest.approx(600.0 / 800)  # 300 W for 2 s over 800 tokens
        assert m.by_concurrency[c].joules_per_token == pytest.approx(level.joules_per_token)
        assert m.by_concurrency[c].power_w == pytest.approx(level.power_w)
    assert m.joules_per_token == pytest.approx(plain.joules_per_token)


def test_with_an_early_stop_the_tail_follows_the_level_that_ends_the_trial(monkeypatch):
    m, slept = _trial(monkeypatch, idle_tail_s=3.0, enough=enough_level("balanced", Constraints(ttft_ceiling_ms=500)))
    assert sorted(m.by_concurrency, key=int) == ["4", "8"]  # 8 broke the ceiling, 4 met it
    assert slept == [2.0, 1.0] and m.hold_after_concurrency == 4


def test_only_efficiency_trials_carry_a_tail():
    """Throughput, balanced and latency trials keep their timing and every number exactly as before."""
    for objective in OBJECTIVES:
        expected = 3.0 if objective == "efficiency" else 0.0
        assert idle_tail_for(objective, SearchOptions()) == expected, objective
        assert idle_tail_for(objective, SearchOptions(all_levels=True)) == expected, objective


# --------------------------------------------------------------------------- end to end

# The trials a balanced calibration of the fake engine below runs on 8729349, the code this feature was built on:
# (stage, configuration). Regenerate only if the search itself changes, never because of the energy work: run
# _calibrate("balanced", ...) below on the new search and paste [(r.stage, r.config.key()) for r in profile.calibration_table].
BASE_BALANCED = [
    ("quant", "vllm/bf16/ctx4096/b64/gmu0.95"), ("quant", "vllm/bf16/ctx4096/b64/gmu0.90"),
    ("quant", "vllm/fp8/ctx4096/b64/gmu0.95"), ("quant", "vllm/fp8/ctx4096/b64/gmu0.90"),
    ("memory", "vllm/fp8/ctx8192/b64/gmu0.95"), ("memory", "vllm/fp8/ctx2048/b64/gmu0.95"),
    ("memory", "vllm/bf16/ctx8192/b64/gmu0.95"), ("memory", "vllm/bf16/ctx2048/b64/gmu0.95"),
    ("batch", "vllm/fp8/ctx4096/b16/gmu0.90"), ("batch", "vllm/fp8/ctx4096/b256/gmu0.90"),
    ("batch", "vllm/bf16/ctx4096/b16/gmu0.90"), ("batch", "vllm/bf16/ctx4096/b256/gmu0.90"),
    ("spec", "vllm/fp8/ctx4096/b64/gmu0.90/sd=ngram:4"), ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb2048"),
    ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb8192"), ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb16384"),
    ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb2048/sd=ngram:4"),
    ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb8192/sd=ngram:4"),
    ("prefill", "vllm/fp8/ctx4096/b64/gmu0.90/pb16384/sd=ngram:4"), ("confirm", "vllm/fp8/ctx4096/b64/gmu0.90"),
    ("confirm", "vllm/fp8/ctx4096/b64/gmu0.90/sd=ngram:4"), ("confirm", "vllm/fp8/ctx4096/b64/gmu0.90/pb2048"),
]


class _Proc:
    pid = None

    def __init__(self, ok):
        self.ok = ok

    def wait_ready(self, timeout):
        return self.ok

    def tail_log(self, n):
        return ""

    def returncode(self):
        return 1

    def stop(self):
        pass


def _calibrate(objective, hw, spec, monkeypatch):
    """calibrate() through the real SubprocessTrialRunner, with the engine launch and the HTTP drive faked."""
    import rigtune.calibrate.search as S
    import rigtune.pipeline as pl
    from tests.test_objectives_and_search import FakeRunner

    fake, launched, calls = FakeRunner(), {}, []

    def run_trial(base_url, hooks, workload, **kw):
        calls.append(kw)
        return fake.run(launched["cfg"], "x").metrics

    monkeypatch.setattr(S, "run_trial", run_trial)
    monkeypatch.setattr(S, "device_used_mb", lambda ids: None)
    candidates, reg = pl.select(hw, spec, force="vllm")
    plan = pl.prepare_and_plan(hw, spec, candidates, reg)

    def launch(cfg, model, port, log_path=None):
        launched["cfg"] = cfg
        return _Proc(not (cfg.gpu_memory_utilization or 0) >= 0.95)  # FakeRunner's simulated out-of-memory

    monkeypatch.setattr(reg["vllm"], "launch", launch)
    return pl.calibrate(hw, spec, objective, plan, reg, log_dir=None), calls


@pytest.mark.usefixtures("no_network")
def test_a_balanced_calibration_runs_the_trials_it_ran_before_and_measures_no_tail(hw_a100, spec, monkeypatch):
    profile, calls = _calibrate("balanced", hw_a100, spec, monkeypatch)
    assert [(r.stage, r.config.key()) for r in profile.calibration_table] == BASE_BALANCED
    assert calls and all(kw.get("idle_tail_s", 0.0) == 0.0 for kw in calls)  # no idle sleep, on any trial
    assert all(r.metrics.idle_power_w is None and r.metrics.idle_tail_s is None for r in profile.calibration_table)


@pytest.mark.usefixtures("no_network")
def test_an_efficiency_calibration_asks_every_trial_for_the_tail(hw_a100, spec, monkeypatch):
    _, calls = _calibrate("efficiency", hw_a100, spec, monkeypatch)
    assert calls and all(kw["idle_tail_s"] == 3.0 for kw in calls)
