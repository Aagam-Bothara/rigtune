"""--kv-tier auto: a tier is tried only when the leader's working set outgrows the GPU pool, sized from it,
and judged like any other change. Fakes only: no engine, no GPU."""

from __future__ import annotations

import math

import pytest
from typer.testing import CliRunner

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
import rigtune.memory as memmod
from rigtune import kvtier
from rigtune.backends import registry
from rigtune.calibrate.datasets import Session, Step
from rigtune.calibrate.objectives import Constraints
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import TURN_OVERHEAD, Workload, get_workload, workload_from_file
from rigtune.layout import _strip, tp_candidates
from rigtune.memlog import MeasuredMemory
from rigtune.memory import host_tier_budget, working_set_bytes
from rigtune.memory import plan as memory_plan
from rigtune.models import Config, GiB, MemoryObservation, TrialMetrics, TrialResult
from rigtune.pipeline import (
    PlanResult,
    SearchOptions,
    calibrate,
    combination_fits,
    gpu_pool_bytes,
    tier_variants_fn,
)
from tests.conftest import make_hw
from tests.test_session_replay import _file

PER_TOKEN = 2 * 28 * 8 * 128 * 2  # Llama 3.2 3B in bf16: 114,688 B a token
POOL = 10 * GiB  # the GPU pool the fake leaders logged
ALL_VLLM = frozenset({"cpu", "hma", "shm"})
ALL_SGLANG = frozenset({"hicache", "file", "file_cap"})


@pytest.fixture
def engines(monkeypatch, tmp_home):
    monkeypatch.setattr(vl, "_offload_features", lambda: ALL_VLLM)
    monkeypatch.setattr(vl, "_shm_free", lambda: 500 * GiB)
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL_SGLANG)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(memmod, "disk_tier_budget", lambda root: 1000 * GiB)
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: None)  # a local disk, whatever this machine has
    return tmp_home


def _plan(hw, spec, pm, backend="vllm"):
    return PlanResult(hw=hw, spec=spec, candidates=[backend], prepared={backend: pm})


def _cfg(backend="vllm", **kw):
    fields = {"ctx": 8192, "batch": 64, "gpu_memory_utilization": 0.9 if backend == "vllm" else 0.88, **kw}
    return Config(backend=backend, quant="bf16", **fields)


def _leader(tokens, reuse=(0.70, 0.72), pool_mb=POOL / 2**20, cfg=None, pool=64):
    m = TrialMetrics(tok_s=1000.0, ttft_ms=100.0, ttft_p95_ms=150.0, tpot_ms=20.0, requests=48,
                     output_tokens=4800, concurrency=8, working_set_tokens=tokens, effective_pool=pool,
                     prefix_reuse=reuse[0], reuse_expected=reuse[1], sessions=128)
    memory = MemoryObservation(measured=MeasuredMemory(kv_mb=pool_mb)) if pool_mb else None
    return TrialResult(config=cfg or _cfg(), stage="batch", metrics=m, memory=memory)


def _fn(hw, spec, pm, backend="vllm", workload=None, counter=None):
    return tier_variants_fn(hw, registry(report_reuse=True), _plan(hw, spec, pm, backend), "throughput",
                            Constraints(), workload, counter)


def test_a_working_set_that_fits_costs_no_trial(engines, spec, prepared_vllm):
    out, note = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(50_000))  # 5.3 GiB against 10 GiB
    assert out == [] and "0.53x the GPU pool (10.0 GiB, measured)" in note and "reuse 0.70 of 0.72" in note
    assert note.startswith("no KV-cache tier tried on vllm")


def test_a_reuse_shortfall_alone_never_opens_the_gate(engines, spec, prepared_vllm):
    # 0.75x the pool and 24 points short of expected: 1.25x the working set would be 0.94x the pool, below
    # the 1.2x a tier needs, so the shortfall could never have produced a trial and is not a gate.
    out, note = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(70_000, reuse=(0.50, 0.74)))
    assert out == [] and note == ("no KV-cache tier tried on vllm: working set 70,000 tokens (7.5 GiB, 64 "
                                  "sessions) is 0.75x the GPU pool (10.0 GiB, measured), reuse 0.50 of 0.74 "
                                  "expected")


def test_one_host_point_at_one_and_a_quarter_times_the_working_set(engines, spec, prepared_vllm):
    out, note = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(100_000))  # 10.7 GiB: 1.07x the pool
    want = math.floor(1.25 * 100_000 * PER_TOKEN / GiB)  # 13 GiB, under the 35 GiB host budget
    assert [c.kv_tier for c in out] == [f"host:{want}"]  # the host holds W: no disk point
    assert out[0].model_copy(update={"kv_tier": None}) == _cfg()  # nothing else changes
    assert note.startswith("KV-cache tier tried on vllm (host:13)")


def test_a_shared_system_prompt_is_counted_once(engines, spec, prepared_vllm):
    """64 live sessions that all start with the same 5.5k-token preamble: the cache holds it once."""
    preamble = " ".join(["policy"] * 5500)
    sessions = [Session(id=f"s{i}", system=preamble, steps=(Step(user=f"question {i}"),)) for i in range(64)]
    wl = Workload(sessions=True, source="file:unused", session_list=sessions, n_prompts=64,
                  prompts=[s.first_turn for s in sessions], fitted=True)
    words = TokenCounter(encode=lambda s: len(s.split()), name="words")
    once = 5500 + TURN_OVERHEAD
    recorded = 64 * 6000  # what replay records: every session's last prompt and answer, preamble included
    unique = recorded - 63 * once  # 36,492 tokens
    fn = _fn(make_hw("rtx4090"), spec, prepared_vllm, workload=wl, counter=words)
    out, note = fn(_leader(recorded))
    assert f"working set {unique:,} tokens" in note and f"a {once:,}-token shared prefix counted once" in note
    assert out == [] and f"{unique * PER_TOKEN / POOL:.2f}x the GPU pool" in note  # 0.39x: it fits
    plain, _ = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(recorded))
    assert plain  # without the correction the same trial looked like 4.1x the pool


def test_vllm_offers_no_disk_point_and_says_why(engines, spec, prepared_vllm):
    hw = make_hw("rtx4090")  # 48 GiB free: a 35 GiB host budget
    out, note = _fn(hw, spec, prepared_vllm)(_leader(380_000))  # 40.6 GiB: the host cannot hold it
    assert [c.kv_tier for c in out] == [f"host:{host_tier_budget(hw) // GiB}"]
    assert "4.06x the GPU pool" in note and "no disk tier: vLLM's disk tier has no size limit in 0.29" in note


def test_sglang_adds_a_capped_disk_point_when_host_ram_cannot_hold_the_working_set(engines, spec,
                                                                                     prepared_vllm):
    hw = make_hw("rtx4090")
    pm = prepared_vllm.model_copy(update={"backend": "sglang"})
    room = host_tier_budget(hw, 10 * GiB)  # SGLang keeps 10 GiB free itself
    out, note = _fn(hw, spec, pm, "sglang")(_leader(380_000, cfg=_cfg("sglang")))
    disk = math.ceil(1.25 * 380_000 * PER_TOKEN / GiB)
    assert [c.kv_tier for c in out] == [f"host:{room // GiB}", f"host:{room // GiB}+disk:{disk}"]


def test_no_disk_point_on_a_network_filesystem_or_without_space(engines, spec, prepared_vllm, monkeypatch):
    hw = make_hw("rtx4090")
    pm = prepared_vllm.model_copy(update={"backend": "sglang"})
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: "rigtune's cache dir /workspace is on a fuse.mfs "
                                                          "filesystem")
    out, note = _fn(hw, spec, pm, "sglang")(_leader(380_000, cfg=_cfg("sglang")))
    assert [c.tier_gib[1] for c in out] == [0] and "fuse.mfs" in note
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: None)
    monkeypatch.setattr(memmod, "disk_tier_budget", lambda root: 10 * GiB)
    out, note = _fn(hw, spec, pm, "sglang")(_leader(380_000, cfg=_cfg("sglang")))
    assert len(out) == 1 and "GiB of disk is not free" in note


def test_dev_shm_is_named_when_it_binds(engines, spec, prepared_vllm, monkeypatch):
    hw = make_hw("rtx4090")
    monkeypatch.setattr(vl, "_shm_free", lambda: 64 * 2**20)  # Docker's default /dev/shm
    out, note = _fn(hw, spec, prepared_vllm)(_leader(100_000))
    assert out == [] and "the 0 GiB of /dev/shm, where vLLM keeps its CPU tier, free for a tier" in note
    monkeypatch.setattr(vl, "_offload_features", lambda: frozenset())  # an engine without the connector
    out, note = _fn(hw, spec, prepared_vllm)(_leader(380_000))
    assert out == [] and "OffloadingConnector" in note


def test_a_tier_under_one_gib_is_not_offered(engines, spec, prepared_vllm):
    out, note = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(5_000, pool_mb=100.0))
    assert out == [] and "under the 1 GiB step" in note


def test_without_a_measured_pool_the_plan_stands_in(engines, spec, prepared_vllm):
    hw = make_hw("rtx4090")
    be = registry()["vllm"]
    leader = _leader(100_000, pool_mb=None)
    g, source = gpu_pool_bytes(leader, hw, prepared_vllm, be.memory_model(hw))
    assert source == "planned"
    workspace = be.memory_model(hw).runtime_workspace
    assert g == int(0.9 * 24 * GiB) - prepared_vllm.weights_for("bf16") - workspace
    _, note = _fn(hw, spec, prepared_vllm)(leader)
    assert "planned)" in note


def test_a_leader_without_a_working_set_is_left_alone(engines, spec, prepared_vllm):
    out, note = _fn(make_hw("rtx4090"), spec, prepared_vllm)(_leader(None))
    assert out == [] and "no working set" in note
    tiered = _leader(100_000, cfg=_cfg(kv_tier="host:8"))
    assert _fn(make_hw("rtx4090"), spec, prepared_vllm)(tiered) == ([], None)


def test_a_hybrid_model_is_sized_in_the_pools_own_units(engines, spec):
    from tests.test_hybrid_memory import qwen35

    hw = make_hw("rtx4090")
    pm, mm = qwen35("vllm"), registry()["vllm"].memory_model(hw)
    cfg = _cfg(ctx=16384, batch=16)
    out, note = _fn(hw, spec, pm)(_leader(64 * 6000, cfg=cfg))
    w = working_set_bytes(pm, cfg, 64 * 6000, 64, mm)
    every_layer_full = pm.arch.kv_bytes_per_token(2) * 64 * 6000
    assert w < every_layer_full / 2  # 8 of 32 layers keep every token; the rest one state per session
    assert f"({w / GiB:.1f} GiB, 64 sessions)" in note and f"{w / POOL:.2f}x the GPU pool" in note
    assert [c.kv_tier for c in out] == [f"host:{int(1.25 * w // GiB)}"]


def test_sglang_counts_a_second_models_host_pool(engines, spec, prepared_vllm):
    hw = make_hw("rtx4090")
    pm = prepared_vllm.model_copy(update={"backend": "sglang"})
    fn = _fn(hw, spec, pm, "sglang")
    room = host_tier_budget(hw, 10 * GiB)
    plain, _ = fn(_leader(380_000, cfg=_cfg("sglang")))
    assert plain[0].kv_tier == f"host:{room // GiB}"
    eagle = _cfg("sglang", spec_decode="eagle3:lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B:3")
    halved, _ = fn(_leader(380_000, cfg=eagle))
    assert halved[0].kv_tier == f"host:{int(room / 2 // GiB)}"


def test_combinations_cannot_carry_a_tier_past_the_host(engines, spec, prepared_vllm):
    hw = make_hw("rtx4090")
    fits = combination_fits(hw, registry(), _plan(hw, spec, prepared_vllm))
    base = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    assert fits(base) and fits(base.model_copy(update={"kv_tier": "host:20"}))
    assert not fits(base.model_copy(update={"kv_tier": "host:200"}))  # 35 GiB free for a tier
    assert not fits(base.model_copy(update={"kv_tier": "host:20", "tp": 2}))  # the engine's own rule
    assert not fits(base.model_copy(update={"kv_tier": "host:20+disk:40"}))  # vLLM's disk tier is not offered


class TierRunner:
    """A session trial on a 10 GiB pool. A tier's gain grows with how far the working set overflows it, as
    the design predicts: none where it fits, 20% from twice the pool on."""

    def __init__(self, tokens=380_000):
        self.tokens, self.calls = tokens, []

    def gain(self, cfg):
        w = self.tokens * PER_TOKEN * (0.5 if cfg.kv_dtype != "auto" else 1.0)
        return 1 + 0.2 * min(1.0, max(0.0, w / POOL - 1)) if cfg.kv_tier else 1.0

    def run(self, cfg, stage):
        self.calls.append((stage, cfg.key()))
        if (cfg.gpu_memory_utilization or 0) >= 0.95:
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="CUDA OOM")
        tok = (1.35 if cfg.quant == "fp8" else 1.0) * (1 + 0.5 * math.log2(cfg.batch)) * 100 * self.gain(cfg)
        m = TrialMetrics(tok_s=tok, ttft_ms=100.0, ttft_p95_ms=150.0, tpot_ms=1000 / tok, requests=48,
                         output_tokens=4800, concurrency=8, working_set_tokens=self.tokens, effective_pool=64,
                         sessions=128)
        return TrialResult(config=cfg, stage=stage, metrics=m,
                           memory=MemoryObservation(measured=MeasuredMemory(kv_mb=POOL / 2**20)))


def _calibrate(tmp_path, spec, prepared_vllm, runner, opts):
    hw = make_hw("rtx4090")
    reg = registry(report_reuse=True)
    wl = workload_from_file(_file(tmp_path), get_workload("chat"))
    be = reg["vllm"]
    kept = memory_plan(hw, prepared_vllm, be.candidate_configs(hw, prepared_vllm, min_ctx=wl.min_ctx),
                       be.memory_model(hw))
    plan_ = PlanResult(hw=hw, spec=spec, candidates=["vllm"], prepared={"vllm": prepared_vllm},
                       feasible={"vllm": kept})
    return calibrate(hw, spec, "throughput", plan_, reg, workload=wl, runner=runner, options=opts)


def test_a_tier_is_adopted_where_the_working_set_overflows_the_pool(engines, tmp_path, spec, prepared_vllm):
    runner = TierRunner(tokens=380_000)  # 4.06x the pool in bf16
    p = _calibrate(tmp_path, spec, prepared_vllm, runner, SearchOptions(kv_tier="auto", kv_quant=False))
    tier_trials = [k for s, k in runner.calls if s == "tier"]
    assert len(tier_trials) == 1 and "/tier=host" in tier_trials[0]  # vLLM: host only
    assert p.config.kv_tier is not None and p.options["tier"] == "auto"
    assert "--kv-transfer-config" in p.launch_args
    assert any(n.startswith("KV-cache tier tried on vllm") for n in p.notes)


def test_a_tier_on_a_working_set_that_barely_overflows_ties_and_is_not_adopted(engines, tmp_path, spec,
                                                                               prepared_vllm):
    runner = TierRunner(tokens=95_500)  # 1.02x the pool: a tier of 12 GiB is tried and gains 0.4%
    p = _calibrate(tmp_path, spec, prepared_vllm, runner, SearchOptions(kv_tier="auto", kv_quant=False))
    assert [k for s, k in runner.calls if s == "tier"] and p.config.kv_tier is None


def test_under_a_budget_the_tier_stage_comes_after_the_memory_sweep(engines, tmp_path, spec, prepared_vllm):
    runner = TierRunner()
    _calibrate(tmp_path, spec, prepared_vllm, runner,
               SearchOptions(kv_tier="auto", kv_quant=False, budget_s=10**9))
    stages = [s for s, _ in runner.calls]
    assert "tier" in stages and stages.index("tier") > max(i for i, s in enumerate(stages) if s == "memory")


def test_off_runs_no_tier_trial_and_keys_nothing(engines, tmp_path, spec, prepared_vllm):
    off = TierRunner()
    p = _calibrate(tmp_path, spec, prepared_vllm, off, SearchOptions())
    assert off.calls and not any(s == "tier" or "tier=" in k for s, k in off.calls)
    assert "tier" not in p.options and not any("KV-cache tier" in n for n in p.notes)
    assert "tier" not in SearchOptions().key("balanced")
    assert SearchOptions(kv_tier="auto").key()["tier"] == "auto"


def test_auto_on_a_flat_workload_says_so_and_tries_nothing(engines, spec, prepared_vllm):
    hw = make_hw("rtx4090")
    reg = registry()
    be = reg["vllm"]
    kept = memory_plan(hw, prepared_vllm, be.candidate_configs(hw, prepared_vllm), be.memory_model(hw))
    plan_ = PlanResult(hw=hw, spec=spec, candidates=["vllm"], prepared={"vllm": prepared_vllm},
                       feasible={"vllm": kept})
    runner = TierRunner()
    p = calibrate(hw, spec, "throughput", plan_, reg, workload=get_workload("chat"), runner=runner,
                  options=SearchOptions(kv_tier="auto"))
    assert not any(s == "tier" for s, _ in runner.calls)
    assert any("--kv-tier auto needs a session file" in n for n in p.notes)


def test_the_stage_runs_once_per_engine_and_logs_the_change():
    calls = []

    def fn(leader):
        calls.append(leader.config.key())
        return [leader.config.model_copy(update={"kv_tier": "host:16"})], "tried"

    runner = TierRunner()
    search = StagedSearch(objective="throughput", runner=runner, tier_variants=fn)
    base = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    search._run(base, "batch")
    search.stage_tier("vllm")
    search.stage_tier("vllm")
    assert len(calls) == 1 and [s for s, _ in runner.calls] == ["batch", "tier"]
    assert search._variant_log[-1][0] == "tier" and "tried" in search.notes


def test_a_tier_trial_that_runs_out_of_memory_is_not_retried_lower():
    class Oom:
        def __init__(self):
            self.calls = []

        def run(self, cfg, stage):
            self.calls.append(cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="out of memory at start-up; failed to start (rc=1)")

    runner = Oom()
    search = StagedSearch(objective="throughput", runner=runner)
    tier = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9,
                  kv_tier="host:16")
    search._run(tier, "tier")
    assert runner.calls == [tier.key()]
    assert any("is not retried with less GPU memory" in n for n in search.notes)
    search._run(tier.model_copy(update={"kv_tier": None}), "batch")  # a GPU-only trial still is
    assert len(runner.calls) == 3 and "gmu0.85" in runner.calls[-1]


def test_multi_gpu_layouts_carry_no_tier():
    cfg = _cfg(kv_tier="host:40")
    assert _strip(cfg).kv_tier is None
    assert tp_candidates(cfg, 2) and all(c.kv_tier is None and c.tp == 2 for c in tp_candidates(cfg, 2))


def test_a_layout_calibration_skips_the_tier_stage(engines, monkeypatch, spec):
    import rigtune.layout as layout
    import rigtune.pipeline as pl
    from rigtune.models import Profile

    hw = make_hw("a100")
    hw.gpus.append(hw.gpus[0].model_copy(update={"index": 1}))
    asked = []

    def resolve(*a, options=None, **k):
        asked.append(options.kv_tier)
        return Profile(rigtune_version="x", hardware_hash="h", hardware=hw, model_id=spec.hf_id,
                       objective="throughput", backend="vllm", backend_version=None, config=_cfg(),
                       launch_args=[])

    monkeypatch.setattr(pl, "resolve_profile", resolve)
    monkeypatch.setattr(pl, "select", lambda *a, **k: ([], {}))
    monkeypatch.setattr(layout, "calibrate_layout", lambda hw, objective, base, *a, **k: base.model_copy())
    monkeypatch.setattr(pl.profile_cache, "load", lambda *a, **k: None)
    monkeypatch.setattr(pl.profile_cache, "save", lambda p: None)
    p = pl._resolve_layout(spec, "throughput", None, True, get_workload("chat"), None, None, hw, None, "off",
                           SearchOptions(kv_tier="auto"), "replicas")
    assert asked == ["off"] and p.options["tier"] == "auto"
    assert any("not tried under --layout replicas" in n for n in p.notes)


def test_the_cli_refuses_auto_without_a_session_file():
    from rigtune.cli import app

    result = CliRunner().invoke(app, ["bench", "m", "--kv-tier", "auto"])
    assert result.exit_code != 0 and "session file" in result.output
    result = CliRunner().invoke(app, ["bench", "m", "--kv-tier", "maybe"])
    assert result.exit_code != 0
