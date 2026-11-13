"""A KV-cache tier on Config: absent from every key until set, counted as a strategy in tie-breaks."""

from __future__ import annotations

import pytest

from rigtune.calibrate.objectives import Constraints, pick
from rigtune.calibrate.search import StagedSearch
from rigtune.models import Config, Profile, TrialMetrics, TrialResult, format_kv_tier, parse_kv_tier


def test_keys_are_unchanged_without_a_tier():
    cfg = Config(backend="vllm", quant="fp8", ctx=8192, batch=64, gpu_memory_utilization=0.9,
                 spec_decode="ngram:4")
    assert cfg.key() == "vllm/fp8/ctx8192/b64/gmu0.90/sd=ngram:4"
    assert "kv_tier" in cfg.model_dump() and cfg.model_dump()["kv_tier"] is None


def test_a_tier_is_in_the_key():
    cfg = Config(backend="vllm", quant="fp8", batch=64, kv_tier=format_kv_tier(16))
    assert cfg.kv_tier == "host:16" and cfg.key().endswith("/tier=host16")
    both = cfg.model_copy(update={"kv_tier": format_kv_tier(48, 120)})
    assert both.key().endswith("/tier=host48+disk120") and both.tier_gib == (48, 120)
    assert cfg.tier_gib == (16, 0) and Config(backend="vllm", quant="fp8").tier_gib == (0, 0)


@pytest.mark.parametrize("bad", ["disk:10", "host:0", "host:x", "host:4+host:4", "ram:4", "host:4+disk"])
def test_malformed_tiers_are_refused(bad):
    with pytest.raises(ValueError):
        parse_kv_tier(bad)


def test_a_profile_saved_before_tiers_still_loads():
    old = {"rigtune_version": "0.2.1", "hardware_hash": "h", "model_id": "m", "objective": "throughput",
           "backend": "vllm", "backend_version": "0.29.0", "launch_args": ["vllm", "serve", "m"],
           "hardware": {"os": "Linux", "python": "3.11", "cpu": {}},
           "config": {"backend": "vllm", "quant": "fp8", "ctx": 4096, "batch": 64,
                      "gpu_memory_utilization": 0.9}}
    p = Profile.model_validate(old)
    assert p.config.kv_tier is None and p.config.key() == "vllm/fp8/ctx4096/b64/gmu0.90"


def _trial(cfg: Config, tok_s: float) -> TrialResult:
    m = TrialMetrics(tok_s=tok_s, ttft_ms=100.0, ttft_p95_ms=150.0, tpot_ms=20.0, requests=32,
                     output_tokens=4096, concurrency=8)
    return TrialResult(config=cfg, stage="tier", metrics=m)


def test_a_tier_that_only_ties_loses():
    base = Config(backend="vllm", quant="fp8", ctx=8192, batch=64, gpu_memory_utilization=0.9)
    tier = base.model_copy(update={"kv_tier": format_kv_tier(32)})
    winner, _ = pick([_trial(tier, 1010.0), _trial(base, 1000.0)], "throughput", Constraints())
    assert winner.config.kv_tier is None  # 1% is noise: the tier holds 32 GiB of host RAM for nothing
    winner, _ = pick([_trial(tier, 1150.0), _trial(base, 1000.0)], "throughput", Constraints())
    assert winner.config.kv_tier == "host:32"


def test_the_tier_is_a_change_combinations_can_carry_and_undo():
    base = Config(backend="vllm", quant="fp8", ctx=8192, batch=64)
    tier = base.model_copy(update={"kv_tier": "host:32", "prefill_budget": 8192})
    delta = StagedSearch._delta(base, tier)
    assert delta == {"prefill_budget": 8192, "kv_tier": "host:32"}
    assert StagedSearch._apply(base, {"kv_tier": "host:32"}).kv_tier == "host:32"
