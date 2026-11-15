"""With --eval-set, the exact configuration served is certified, its KV cache and speculation included, or a
certified fallback is served, or nothing."""
import pytest

import rigtune.pipeline as pipeline
from rigtune.calibrate.objectives import Constraints, pick
from rigtune.calibrate.search import StagedSearch
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.conftest import make_hw
from tests.test_accuracy_gate import Clock, GradingRunner, Script, _feasible, _flip, _gate, _search

REF = "1" * 300
KV = ("kv", lambda c: [c.model_copy(update={"kv_dtype": "fp8"})] if c.kv_dtype == "auto" else [])
SPEC = ("spec", lambda c: [c.model_copy(update={"spec_decode": f"ngram:{k}"}) for k in (2, 3, 4, 5)]
        if not c.spec_decode else [])


class PerConfig(GradingRunner):
    """Answers depend on the exact configuration running: answers(cfg) gives its grades."""

    def __init__(self, gate, answers, **kw):
        super().__init__(gate, **kw)
        self.answers = answers

    def run(self, cfg, stage):
        self.gate.answer.grades[(cfg.backend, cfg.quant)] = self.answers(cfg)
        self.gate.answer.second[(cfg.backend, cfg.quant)] = self.answers(cfg)
        return super().run(cfg, stage)


def _run(answers, variants=(KV,), confirm_top=3, script=None, **kw):
    script = script or Script({("vllm", "bf16"): REF})
    gate = _gate(300, script, clock=kw.get("clock"))
    search = _search(gate, PerConfig(gate, answers, clock=kw.pop("clock", None)), variant_stages=list(variants),
                     confirm_top=confirm_top, **kw)
    winner, notes = search.run(_feasible(("vllm", "bf16")))
    return gate, search, winner, notes, script


def test_a_fast_lossy_kv_configuration_falls_back_to_the_certified_baseline_re_measured():
    gate, search, winner, notes, _ = _run(lambda c: "0" * 300 if c.kv_dtype == "fp8" else REF)
    assert winner is not None and winner.config.kv_dtype == "auto"
    assert winner.stage == "confirm"  # the fallback's speed is a fresh run, not the search's best of several
    assert any("final configuration" in n and "refused" in n for n in notes)
    assert any("It is a fallback" in n for n in notes) and gate.final_record.reason.endswith("a fallback")


def test_disabling_confirmation_cannot_bypass_final_quality_checks():
    _, _, winner, _, _ = _run(lambda c: "0" * 300 if c.kv_dtype == "fp8" else REF, confirm_top=0)
    assert winner.config.kv_dtype == "auto"


def test_an_ungraded_new_configuration_is_not_certified():
    gate = _gate(300, Script({("vllm", "bf16"): REF}))
    cfg = Config(backend="vllm", quant="bf16", batch=16)
    gate.plan([("vllm", "bf16")])
    gate.record("fake://vllm/bf16", "m", cfg)
    gate.decide()
    assert gate.certified(cfg)  # the reference's own stage-1 configuration
    assert not gate.certified(cfg.model_copy(update={"spec_decode": "ngram:3"}))
    assert gate.certified(cfg.model_copy(update={"power_limit_w": 250}))  # a power cap changes no answer


# ---- unanswered items (the critic's repro_gate.py)


def test_one_unanswered_item_on_the_final_configuration_costs_an_item_not_the_certificate():
    gate, _, winner, _, _ = _run(lambda c: _flip(REF, [5], "-") if c.kv_dtype == "fp8" else REF)
    assert winner.config.kv_dtype == "fp8"
    rec = gate.final_record
    assert (rec.lost, rec.gained, rec.paired) == (1, 0, 300)  # counted as lost, over the reference's items


def test_an_item_the_reference_missed_drops_out_instead_of_blocking_every_configuration():
    gate, _, winner, _, _ = _run(lambda c: _flip(REF, [5], "-"))
    assert winner.config.kv_dtype == "fp8"
    assert gate.final_record.paired == 299


def test_unanswered_items_still_count_against_a_lossy_configuration():
    lossy = _flip(REF, range(10), "-")  # ten unanswered where the reference was right: ten lost
    gate, _, winner, notes, _ = _run(lambda c: lossy if c.kv_dtype == "fp8" else REF)
    assert winner.config.kv_dtype == "auto"
    refused = next(r for r in gate.records() if r.config and r.admitted is False)
    assert refused.lost == 10


def test_weights_admitted_in_stage_1_with_an_unanswered_item_are_served():
    script = Script({("vllm", "bf16"): REF, ("vllm", "fp8"): _flip(REF, [5], "-")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate), confirm_top=3)
    winner, _ = search.run(_feasible(("vllm", "bf16"), ("vllm", "fp8")))
    assert winner.config.quant == "fp8" and gate.final_record.admitted


# ---- cost


def test_refused_settings_are_not_graded_again_and_the_attempts_are_capped():
    lossy = "0" * 40 + "1" * 260
    gate, search, winner, notes, script = _run(lambda c: lossy if (c.kv_dtype == "fp8" or c.spec_decode) else REF,
                                               variants=(KV, SPEC))
    assert winner.config.kv_dtype == "auto" and not winner.config.spec_decode
    final_passes = len(script.passes) - 2  # stage 1 graded the reference twice
    assert final_passes <= search.final_attempts
    skipped = next(n for n in notes if n.startswith("accuracy gate: not graded for the final choice"))
    assert "shares a refused KV cache or speculation method" in skipped


@pytest.mark.parametrize("attempts", [0, 1])
def test_the_attempts_are_configurable_and_an_already_certified_configuration_costs_none(attempts):
    gate, search, winner, notes, script = _run(lambda c: "0" * 300 if c.kv_dtype == "fp8" else REF,
                                               confirm_top=0, final_attempts=attempts)
    assert winner is not None and winner.config.kv_dtype == "auto"
    assert len(script.passes) == 2 + attempts
    if attempts == 0:
        assert any("over the 0 final grading attempts" in n for n in notes)


def test_a_spent_budget_falls_back_to_a_certified_configuration_and_says_its_speed_is_from_the_search():
    clock = Clock()
    script = Script({("vllm", "bf16"): REF}, clock=clock, seconds_per_pass=10)
    gate = _gate(300, script, clock=clock)
    search = StagedSearch(objective="throughput", runner=GradingRunner(gate, clock=clock), gate=gate,
                          variant_stages=[KV], confirm_top=3, budget_s=200, clock=clock)
    winner, notes = search.run(_feasible(("vllm", "bf16")))
    assert winner is not None and winner.config.kv_dtype == "auto" and gate.final_record.admitted
    assert any("could not be re-measured, so the profile reports its speed from the search" in n for n in notes)


# ---- the stage itself


def _decided_gate(baseline):
    gate = _gate(300, Script({("vllm", "bf16"): REF}))
    gate.plan([("vllm", "bf16")])
    gate.record("fake://vllm/bf16", "m", baseline)
    gate.decide()
    return gate


def _result(cfg, tok_s, ttft, stage="batch"):
    return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(tok_s=tok_s, ttft_ms=ttft, requests=72,
                                                                     output_tokens=1000))


def test_a_leader_that_breaks_strict_latency_when_re_measured_leaves_a_certified_fallback():
    baseline = Config(backend="vllm", quant="bf16", ctx=4096, batch=16)
    candidate = baseline.model_copy(update={"ctx": 8192})
    gate = _decided_gate(baseline)

    class SlowerOnRemeasurement(GradingRunner):
        def run(self, cfg, stage):
            result = super().run(cfg, stage)
            ttft = 200 if cfg == candidate else 50
            result.metrics = TrialMetrics(tok_s=200, ttft_ms=ttft, requests=72, output_tokens=1000)
            return result

    cons = Constraints(ttft_ceiling_ms=100, strict=True)
    search = StagedSearch(objective="balanced", constraints=cons, gate=gate, runner=SlowerOnRemeasurement(gate))
    promising = _result(candidate, 200, 50)
    search.results = [_result(baseline, 100, 50, "quant"), promising]
    pool = search.stage_final_quality([promising])
    winner, _ = pick(pool, "balanced", cons)
    assert winner is not None and winner.config == baseline


def test_expansion_keeps_each_configuration_s_best_ranked_run():
    baseline = Config(backend="vllm", quant="bf16", ctx=4096, batch=16)
    gate = _decided_gate(baseline)
    search = StagedSearch(objective="throughput", gate=gate, runner=GradingRunner(gate))
    search.results = [_result(baseline, 90, 50, "quant"), _result(baseline, 110, 50, "memory")]
    chosen = search.stage_final_quality([])
    assert [r.metrics.tok_s for r in chosen] == [110]


# ---- layouts


def _gated_base(hw, spec, prepared):
    from rigtune.models import EvalSetInfo
    from tests.test_cache_and_pipeline import _profile
    base = _profile(hw, spec, prepared)
    base.eval_set = EvalSetInfo(name="eval", path="eval", items=300, sha256="a", max_accuracy_loss=3)
    return base


@pytest.mark.parametrize("kind", ["layout", "phases"])
def test_a_new_layout_cannot_inherit_a_single_engines_quality_certificate(kind, spec, prepared_vllm, monkeypatch):
    import rigtune.disagg as D
    from rigtune.backends import registry

    hw = make_hw("a100")
    hw.gpus = [hw.gpus[0].model_copy(update={"index": i}) for i in range(2)]
    base = _gated_base(hw, spec, prepared_vllm)
    with pytest.raises(RuntimeError, match="accuracy certification"):
        if kind == "layout":
            from rigtune.layout import calibrate_layout
            calibrate_layout(hw, "balanced", base, registry(), "auto")
        else:  # a pair that would be measured: no connector or memory reason to fall back first
            monkeypatch.setattr(D, "check_support", lambda *a, **kw: None)
            monkeypatch.setattr(D, "candidate_pairs", lambda *a, **kw: ["a pair"])
            D.calibrate_disaggregated(hw, spec, "balanced", base, registry(), phases="auto")


@pytest.mark.parametrize("kind", ["layout", "phases"])
def test_auto_on_one_gpu_with_an_eval_set_falls_back_to_the_certified_single_engine(kind, hw_a100, spec,
                                                                                    prepared_vllm):
    """The pre-check lets --layout auto / --phases auto through on one GPU; the step after the single-engine
    calibration must then serve that certified profile, as it did before, not refuse."""
    base = _gated_base(hw_a100, spec, prepared_vllm)
    if kind == "layout":
        from rigtune.layout import calibrate_layout
        out = calibrate_layout(hw_a100, "balanced", base, {}, "auto")
        assert out.replicas == 1 and out.config == base.config
    else:
        from rigtune.disagg import calibrate_disaggregated
        out = calibrate_disaggregated(hw_a100, spec, "balanced", base, {}, phases="auto")
        assert out.disagg is None and out.config == base.config
    assert out.eval_set == base.eval_set


def test_strict_mode_judges_a_certified_fallback_against_everything_measured():
    """--strict-constraints with a relative throughput floor: the certified fallback must still be measured against
    the faster configuration the gate refused, as it would be without the gate, not against itself alone."""
    lossy = "0" * 300

    class Speeds(PerConfig):
        def run(self, cfg, stage):
            res = super().run(cfg, stage)
            fast = cfg.kv_dtype == "fp8"
            m = TrialMetrics(tok_s=300 if fast else 100, ttft_ms=50, tpot_ms=5 if fast else 15, requests=32,
                             output_tokens=2048, concurrency=8)
            res.metrics = m.model_copy(update={"by_concurrency": {"8": m}})
            return res

    for strict in (True, False):
        gate = _gate(300, Script({("vllm", "bf16"): REF}))
        runner = Speeds(gate, lambda c: lossy if c.kv_dtype == "fp8" else REF)
        search = StagedSearch(objective="latency", runner=runner, gate=gate, variant_stages=[KV], confirm_top=3,
                              constraints=Constraints(ttft_ceiling_ms=500, strict=strict))
        winner, notes = search.run(_feasible(("vllm", "bf16")))
        if strict:  # 100 tok/s is under half of the 300 measured: the floor is missed, so nothing is served
            assert winner is None and any(n.startswith("strict constraints") for n in notes)
        else:
            assert winner is not None and winner.config.kv_dtype == "auto"


@pytest.mark.parametrize("layout, phases, gpus, refused", [
    ("replicas", "unified", 2, True), ("auto", "unified", 2, True), ("single", "auto", 2, True),
    ("single", "disaggregated", 1, True),
    ("auto", "unified", 1, False),  # --layout auto on one GPU serves one engine: the eval set still applies
])
def test_an_eval_set_with_a_multi_gpu_search_is_refused_before_calibration(layout, phases, gpus, refused,
                                                                           monkeypatch):
    hw = make_hw("a100")
    hw.gpus = [hw.gpus[0].model_copy(update={"index": i}) for i in range(gpus)]

    def select_(*args, **kwargs):
        raise RuntimeError("calibration started")

    monkeypatch.setattr(pipeline, "select", select_)
    with pytest.raises(pipeline.OptionRefused if refused else RuntimeError, match=None if refused else "started"):
        pipeline.resolve_profile(pipeline.ModelSpec(hf_id="m"), hw=hw, layout=layout, phases=phases,
                                 recalibrate=True, options=pipeline.SearchOptions(eval_set="gsm8k"))
