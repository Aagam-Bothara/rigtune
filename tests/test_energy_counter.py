"""The NVML energy counter is read beside the sampled power, and never replaces the number every objective scores."""

from __future__ import annotations

import sys
import types

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.measure import Telemetry, TelemetrySummary
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload


class NotSupported(Exception):
    pass


def fake_nvml(counter_mj=None, power_mw: float = 250_000.0):
    """Module-shaped stand-in for pynvml. `counter_mj` lists what successive counter reads return (per device);
    None makes the card answer NotSupported, as a driver that does not report the counter does."""
    reads = iter(counter_mj or [])
    nv = types.ModuleType("pynvml")
    nv.NVML_CLOCK_SM = 1
    nv.inits = 0
    nv.shutdowns = 0

    def init():
        nv.inits += 1

    def shutdown():
        nv.shutdowns += 1

    def counter(handle):
        if counter_mj is None:
            raise NotSupported("NVML_ERROR_NOT_SUPPORTED")
        return next(reads)

    nv.nvmlInit = init
    nv.nvmlShutdown = shutdown
    nv.nvmlDeviceGetHandleByIndex = lambda i: f"gpu{i}"
    nv.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=2 * 2**30)
    nv.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=90)
    nv.nvmlDeviceGetPowerUsage = lambda h: power_mw
    nv.nvmlDeviceGetClockInfo = lambda h, kind: 1800
    nv.nvmlDeviceGetTotalEnergyConsumption = counter
    return nv


def _telemetry(monkeypatch, nv) -> Telemetry:
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    monkeypatch.setattr(Telemetry, "_start_llmtrace", lambda self: False)  # llmtrace may be installed here
    return Telemetry(LlmtraceHooks(gpu_ids=[0]), interval_ms=5)


def test_a_rising_counter_gives_the_window_its_energy(monkeypatch):
    nv = fake_nvml(counter_mj=[5_000_000, 5_250_000])
    with _telemetry(monkeypatch, nv) as tel:
        pass
    assert tel.summary.counter_energy_j == 250.0  # 250 000 mJ between start and stop
    assert tel.summary.source == "pynvml"
    assert nv.inits == nv.shutdowns  # the counter's own NVML session is closed again


def test_a_card_without_the_counter_keeps_the_trapezoid(monkeypatch):
    with _telemetry(monkeypatch, fake_nvml(counter_mj=None)) as tel:
        pass
    assert tel.summary.counter_energy_j is None


def test_a_counter_that_goes_backwards_is_not_trusted(monkeypatch):
    with _telemetry(monkeypatch, fake_nvml(counter_mj=[9_000_000, 1_000])) as tel:  # driver reload mid-window
        pass
    assert tel.summary.counter_energy_j is None
    assert M._counter_delta_j(10.0, 10.0) is None and M._counter_delta_j(None, 5.0) is None


def test_no_gpu_means_no_counter(monkeypatch):
    nv = fake_nvml(counter_mj=[1, 2])
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    with Telemetry(LlmtraceHooks(gpu_ids=[]), interval_ms=5) as tel:
        pass
    assert tel.summary.counter_energy_j is None and nv.inits == 0


class _ScriptedTelemetry:
    """Stands in for Telemetry inside run_trial: every level window reports the same trapezoid and counter."""

    def __init__(self, hooks, pid=None, interval_ms=100):
        self.summary = TelemetrySummary(source="pynvml", power_w=300.0, energy_j=600.0, samples=20,
                                        counter_energy_j=630.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
    async def run():
        return [M.RequestOutcome(ok=True, ttft_s=0.05, duration_s=1.0, tokens=100, token_source="usage")] * 4
    return run()


def test_joules_per_token_stays_the_sampled_integral(monkeypatch):
    """The counter is recorded beside J/tok, never in place of it, so no recorded or scored number moves."""
    monkeypatch.setattr(M, "Telemetry", _ScriptedTelemetry)
    monkeypatch.setattr(M, "_drive", _fake_drive)
    wl = Workload(n_prompts=4, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4))
    m = M.run_trial("http://x", LlmtraceHooks(gpu_ids=[0]), wl, warmup=False, counter=TokenCounter())
    for level in m.by_concurrency.values():
        assert level.joules_per_token == 600.0 / 400 and level.counter_energy_j == 630.0
    assert m.joules_per_token == 1200.0 / 800 and m.counter_energy_j == 1260.0
