"""Speculation first, the other stages per speculation setting, engines that cannot catch up skipped, and a known
start-up failure left out of the search."""

from __future__ import annotations

from typing import List

import pytest

from rigtune.calibrate.search import StagedSearch
from rigtune.models import Config
from rigtune.pipeline import calibrate, combination_fits, prepare_and_plan, select
from tests.test_contenders import SGLANG, ScoredRunner
from tests.test_objectives_and_search import FakeRunner

VLLM = Config(backend="vllm", quant="fp8", ctx=16384, batch=16, gpu_memory_utilization=0.95)
SGL = SGLANG.model_copy(update={"quant": "fp8", "batch": 16})
SPECS = {"suffix:24": 1.35, "draft:small:4": 1.15, "ngram_gpu:4": 1.15}


def rtx4090_like(c: Config) -> float:
    """SGLang in fp8 leads at first and has no speculation. On vLLM, suffix decoding is the best method alone,
    and the int8 cache and a 16k prefill budget pay only with the draft model (neutral or worse without it)."""
    if c.backend == "sglang":
        return 781.0 * (0.96 if c.kv_dtype != "auto" else 1.0)
    tok = 670.0 * SPECS.get(c.spec_decode or "", 1.0)
    draft = (c.spec_decode or "").startswith("draft")
    if c.kv_dtype == "int8":
        tok *= 1.2 if draft else 1.0
    if c.prefill_budget == 16384:
        tok *= 1.12 if draft else 0.99
    return tok


def kv(c: Config) -> List[Config]:
    if c.kv_dtype != "auto":
        return []
    return [c.model_copy(update={"kv_dtype": "int8" if c.backend == "vllm" else "fp8_e5m2"})]


def spec(c: Config) -> List[Config]:
    if c.backend != "vllm" or c.spec_decode:
        return []
    return [c.model_copy(update={"spec_decode": s}) for s in SPECS]


def prefill(c: Config) -> List[Config]:
    return [c.model_copy(update={"prefill_budget": b}) for b in (2048, 16384) if c.prefill_budget != b]


def search(spec_first: bool) -> tuple:
    runner = ScoredRunner(rtx4090_like)
    s = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                     prefill_variants=prefill, spec_first=spec_first)
    winner, notes = s.run([VLLM, SGL])
    return winner, notes, runner


def test_speculation_first_finds_what_only_pays_with_the_draft_model():
    winner, notes, runner = search(spec_first=True)
    assert winner.config.spec_decode == "draft:small:4"
    assert (winner.config.kv_dtype, winner.config.prefill_budget) == ("int8", 16384)
    stages = [s for s, c in runner.ran if c.backend == "vllm"]
    assert stages.index("spec") < stages.index("prefill")  # speculation before the settings that depend on it


def test_an_engine_left_behind_with_nothing_more_to_offer_is_not_tuned():
    _, notes, runner = search(spec_first=True)
    assert [s for s, c in runner.ran if c.backend == "sglang"] == ["quant"]  # only its first trial
    assert any("sglang was not tuned further: vllm led it by" in n for n in notes)


def test_sglangs_large_prefill_budgets_are_left_out_on_a_24_gb_card(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec)
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    small = hw_a100.model_copy(update={"gpus": [g.model_copy(update={"vram_total_bytes": 24 * 2**30})
                                                for g in hw_a100.gpus]})
    assert small.gpu.vram_total_bytes == 24 * 2**30
    sgl = plan.feasible["sglang"][0][0].model_copy(update={"prefill_budget": 8192})
    vll = plan.feasible["vllm"][0][0].model_copy(update={"prefill_budget": 8192})
    assert combination_fits(hw_a100, reg, plan)(sgl)
    assert not combination_fits(small, reg, plan)(sgl)
    assert combination_fits(small, reg, plan)(vll)  # the rule is SGLang's alone


def test_calibration_runs_speculation_before_the_prefill_stage(hw_a100, spec, no_network):
    candidates, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    stages = [t.stage for t in calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner()).calibration_table]
    assert "spec" in stages and "prefill" in stages and stages.index("spec") < stages.index("prefill")


# The trials the search runs on rtx4090_like, choosing engines after the batch stage. SGLang offers no speculation
# there, so choosing after speculation (--speculative all) must measure exactly these too.
BEFORE = [
    ("quant", "vllm/fp8/ctx16384/b16/gmu0.95"), ("quant", "sglang/fp8/ctx8192/b16/gmu0.90"),
    ("spec", "vllm/fp8/ctx16384/b16/gmu0.95/sd=suffix:24"), ("spec", "vllm/fp8/ctx16384/b16/gmu0.95/sd=draft:small:4"),
    ("spec", "vllm/fp8/ctx16384/b16/gmu0.95/sd=ngram_gpu:4"),
    ("prefill", "vllm/fp8/ctx16384/b16/gmu0.95/pb2048/sd=suffix:24"),
    ("prefill", "vllm/fp8/ctx16384/b16/gmu0.95/pb16384/sd=suffix:24"),
    ("kv", "vllm/fp8/ctx16384/b16/gmu0.95/kvint8/sd=suffix:24"),
    ("combine", "vllm/fp8/ctx16384/b16/gmu0.95/pb2048/kvint8/sd=suffix:24"),
    ("combine", "vllm/fp8/ctx16384/b16/gmu0.95/pb16384/kvint8/sd=suffix:24"),
    ("prefill", "vllm/fp8/ctx16384/b16/gmu0.95/pb2048/sd=draft:small:4"),
    ("prefill", "vllm/fp8/ctx16384/b16/gmu0.95/pb16384/sd=draft:small:4"),
    ("kv", "vllm/fp8/ctx16384/b16/gmu0.95/pb16384/kvint8/sd=draft:small:4"),
    ("combine", "vllm/fp8/ctx16384/b16/gmu0.95/kvint8/sd=draft:small:4"),
    ("combine", "vllm/fp8/ctx16384/b16/gmu0.95/pb2048/kvint8/sd=draft:small:4"),
]


@pytest.mark.parametrize("after_spec", [False, True])
def test_when_sglang_offers_no_speculation_choosing_after_it_measures_the_same(after_spec):
    runner = ScoredRunner(rtx4090_like)
    s = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                     prefill_variants=prefill, spec_first=True, decide_after_spec=after_spec)
    _, notes = s.run([VLLM, SGL])
    assert [(st, c.key()) for st, c in runner.ran] == BEFORE
    assert any("sglang was not tuned further" in n for n in notes)
    # choosing after speculation, vLLM no longer gets through for what SGLang lacks: it leads on its own
    assert any("which sglang does not" in n for n in notes) is not after_spec


EAGLE = "eagle3:head:3"


def sglang_speculates(c: Config) -> float:
    """The 4090 prediction: SGLang fp8 at 781 tok/s, 1000 with an EAGLE3 head, against vLLM fp8 at 670 and 904
    with suffix decoding, its best method there (901 measured). The other settings change nothing here."""
    if c.backend == "sglang":
        return 1000.0 if c.spec_decode == EAGLE else 781.0
    return 670.0 * SPECS.get(c.spec_decode or "", 1.0)


def spec_both(c: Config) -> List[Config]:
    if c.spec_decode:
        return []
    return [c.model_copy(update={"spec_decode": EAGLE})] if c.backend == "sglang" else spec(c)


def test_both_engines_speculate_before_either_is_dropped():
    runner = ScoredRunner(sglang_speculates)
    s = StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec_both)],
                     prefill_variants=prefill, spec_first=True, decide_after_spec=True)
    winner, notes = s.run([VLLM, SGL])
    assert winner.config.backend == "sglang" and winner.config.spec_decode == EAGLE
    # stage 1, then speculation on both engines in stage 1's order, before anything else
    assert [(st, c.backend) for st, c in runner.ran][:6] == [
        ("quant", "vllm"), ("quant", "sglang"), ("spec", "sglang"), ("spec", "vllm"), ("spec", "vllm"),
        ("spec", "vllm")]
    assert not any("which sglang does not" in n or "which vllm does not" in n for n in notes)
    # Choosing after the batch stage, SGLang (781 against 670) offered nothing vLLM lacked, so vLLM was never tuned.
    old = ScoredRunner(sglang_speculates)
    StagedSearch(objective="throughput", runner=old, variant_stages=[("kv", kv), ("spec", spec_both)],
                 prefill_variants=prefill, spec_first=True).run([VLLM, SGL])
    assert [st for st, c in old.ran if c.backend == "vllm"] == ["quant"]
