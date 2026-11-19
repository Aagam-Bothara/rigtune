"""Once speculation has been measured on every engine (--speculative all), the leading engine is tuned first.

The shape of the A100 calibration of Qwen3.5-9B (18 Nov 2026): vLLM with the model's MTP head at 3 tokens led at
1023 tok/s and SGLang with it at 937, inside the contender band. vLLM refuses an fp8 KV cache with MTP and SGLang
does not, so SGLang offered one more stage and was tuned first: 7 trials, its fp8 cache 2% slower. The hour ran
out before vLLM's leader got a prefill trial or its confirm round."""

from __future__ import annotations

from typing import List

from rigtune.calibrate.search import StagedSearch
from rigtune.models import Config, TrialResult
from tests.test_contenders import ScoredRunner

VLLM = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.95)
SGL = Config(backend="sglang", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.93)
SPEED = {("vllm", None): 579.0, ("vllm", "mtp:1"): 877.0, ("vllm", "mtp:3"): 1023.0,
         ("sglang", None): 605.0, ("sglang", "mtp:1"): 809.0, ("sglang", "mtp:3"): 937.0}


def a100_like(c: Config) -> float:
    tok = SPEED[(c.backend, c.spec_decode)]
    if c.kv_dtype != "auto":
        tok *= 0.98
    if c.backend == "vllm" and c.prefill_budget == 2048:
        tok *= 1.04
    return tok


def kv(c: Config) -> List[Config]:
    if c.kv_dtype != "auto" or (c.backend == "vllm" and (c.spec_decode or "").startswith("mtp")):
        return []  # vLLM refuses an fp8 KV cache with MTP
    return [c.model_copy(update={"kv_dtype": "fp8_e5m2"})]


def spec(c: Config) -> List[Config]:
    return [] if c.spec_decode else [c.model_copy(update={"spec_decode": s}) for s in ("mtp:1", "mtp:3")]


def prefill(c: Config) -> List[Config]:
    return [c.model_copy(update={"prefill_budget": b}) for b in (2048, 8192) if c.prefill_budget != b]


class TimedRunner(ScoredRunner):
    """Every trial takes 100 s on a clock the search reads."""

    def __init__(self, score):
        super().__init__(score)
        self.t = 0.0

    def clock(self) -> float:
        return self.t

    def run(self, cfg: Config, stage: str) -> TrialResult:
        self.t += 100.0
        return super().run(cfg, stage)


def _search(runner, budget_s=None, after_spec=True) -> StagedSearch:
    kw = {"budget_s": budget_s, "clock": runner.clock} if budget_s is not None else {}
    return StagedSearch(objective="throughput", runner=runner, variant_stages=[("kv", kv), ("spec", spec)],
                        prefill_variants=prefill, spec_first=True, decide_after_spec=after_spec, **kw)


def _after_spec(runner) -> List[tuple]:
    ran = [(st, c.backend) for st, c in runner.ran]
    last = max(i for i, (st, _) in enumerate(ran) if st == "spec")
    return ran[last + 1:]


def test_the_leading_engine_is_tuned_first_once_every_engine_has_speculated():
    runner = TimedRunner(a100_like)
    winner, _ = _search(runner).run([VLLM, SGL])
    tuned = _after_spec(runner)
    assert tuned[0] == ("prefill", "vllm")
    assert (winner.config.backend, winner.config.spec_decode, winner.config.prefill_budget) == ("vllm", "mtp:3", 2048)


def test_a_contender_the_tuned_leader_leaves_outside_the_band_is_then_skipped():
    # SGLang came within 10% after speculation (937 against 1023), but vLLM's prefill trial took it to 1064, 12%
    # ahead, and SGLang offers nothing vLLM's engine lacks: the rule that skipped an engine tuned second skips it.
    runner = TimedRunner(a100_like)
    winner, notes = _search(runner).run([VLLM, SGL])
    assert all(b == "vllm" for _, b in _after_spec(runner))
    assert any("sglang came within 10% of the leading vllm after the speculation stage" in n for n in notes)
    assert any("sglang was not tuned further: vllm led it by 12% once tuned" in n for n in notes)


def test_a_budget_that_runs_out_mid_tuning_spends_it_on_the_leader():
    runner = TimedRunner(a100_like)
    s = _search(runner, budget_s=1000.0)
    winner, notes = s.run([VLLM, SGL])
    tuned = _after_spec(runner)
    assert tuned and all(b == "vllm" for _, b in tuned)
    assert any("calibration budget" in n for n in notes)
    assert winner.config.prefill_budget == 2048 and winner.metrics.tok_s > 1023.0


def test_without_a_budget_it_runs_a_subset_of_the_old_trials_and_picks_the_same():
    # The order SGLang first ran these 20 trials; leader first runs 10 of them, and the same configuration wins.
    runner = TimedRunner(a100_like)
    winner, _ = _search(runner).run([VLLM, SGL])
    old = ['vllm/bf16/ctx8192/b64/gmu0.95', 'sglang/bf16/ctx8192/b64/gmu0.93',
           'sglang/bf16/ctx8192/b64/gmu0.93/sd=mtp:1', 'sglang/bf16/ctx8192/b64/gmu0.93/sd=mtp:3',
           'vllm/bf16/ctx8192/b64/gmu0.95/sd=mtp:1', 'vllm/bf16/ctx8192/b64/gmu0.95/sd=mtp:3',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb2048/sd=mtp:3', 'sglang/bf16/ctx8192/b64/gmu0.93/pb8192/sd=mtp:3',
           'sglang/bf16/ctx8192/b64/gmu0.93/kvfp8_e5m2/sd=mtp:3',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb2048/kvfp8_e5m2/sd=mtp:3',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb8192/kvfp8_e5m2/sd=mtp:3',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb2048/sd=mtp:1', 'sglang/bf16/ctx8192/b64/gmu0.93/pb8192/sd=mtp:1',
           'sglang/bf16/ctx8192/b64/gmu0.93/kvfp8_e5m2/sd=mtp:1',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb2048/kvfp8_e5m2/sd=mtp:1',
           'sglang/bf16/ctx8192/b64/gmu0.93/pb8192/kvfp8_e5m2/sd=mtp:1',
           'vllm/bf16/ctx8192/b64/gmu0.95/pb2048/sd=mtp:3', 'vllm/bf16/ctx8192/b64/gmu0.95/pb8192/sd=mtp:3',
           'vllm/bf16/ctx8192/b64/gmu0.95/pb2048/sd=mtp:1', 'vllm/bf16/ctx8192/b64/gmu0.95/pb8192/sd=mtp:1']
    ran = [c.key() for _, c in runner.ran]
    assert len(ran) == 10 and set(ran) <= set(old)
    assert winner.config.key() == 'vllm/bf16/ctx8192/b64/gmu0.95/pb2048/sd=mtp:3'
