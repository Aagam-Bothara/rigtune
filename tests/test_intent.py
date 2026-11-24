"""Why to change a seed (rigtune.intent, recipe as seed S5): the shipped rules, when each holds, and their trials."""

from __future__ import annotations

import json

import pytest

from rigtune.calibrate.search import StagedSearch
from rigtune.intent import holds, load_rules, propose
from rigtune.models import ArchInfo, Config
from tests.test_inkling2 import INKLING2
from tests.test_objectives_and_search import FakeRunner

RECIPE = Config(backend="vllm", quant="ckpt-nvfp4", ctx=262144, batch=1024, gpu_memory_utilization=0.9,
                spec_decode="dspark:7", extra={"reasoning_parser": "inkling"})


def _inkling() -> ArchInfo:
    from rigtune.hfconfig import arch_from_config

    return arch_from_config(INKLING2)


def _all_offered(cfg):
    return ["mtp:3", "suffix:24", "ngram_gpu:4"]


def test_every_shipped_rule_has_its_reason_and_its_evidence():
    rules = load_rules()
    assert len(rules) >= 3
    for r in rules:
        assert r["because"] and r["evidence"].startswith("benchmarks/sessions/")


@pytest.mark.parametrize("prompt, users, expected", [
    (512, 8, {"inkling-short-prompts-token-budget"}),  # session 1's chat prompts
    (512, 32, {"inkling-short-prompts-token-budget"}),
    (512, 64, set()),  # measured -10% at 64 users
    (8192, 32, set()),  # session 2's long prompts: no measurement for it
])
def test_the_inkling_recipe_gets_the_change_its_traffic_was_measured_on(prompt, users, expected):
    found, notes = propose(RECIPE, _inkling(), 79.18, prompt, users, _all_offered)
    assert {p.rule for p in found} == expected and notes == []
    for p in found:  # the batch vLLM can run at that budget with 7 drafted tokens, stated
        assert (p.config.prefill_budget, p.config.batch) == (2048, 256)
    # Not on SGLang: the evidence is vLLM's.
    assert propose(RECIPE.model_copy(update={"backend": "sglang"}), _inkling(), 79.18, prompt, users,
                   _all_offered)[0] == []


def test_the_inkling_proposal_passes_rigtune_s_own_launch_check(monkeypatch):
    from rigtune.backends import vllm as vl
    from rigtune.backends.facts import EngineFacts
    from tests.test_inkling2 import _prepared

    monkeypatch.setattr(vl, "_vllm_runs_own_drafter", lambda: True)
    monkeypatch.setattr(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", version="0.35.0",
                                                                 spec_fields=["draft_sample_method"]))
    be, pm = vl.VllmBackend(), _prepared()
    found, _ = propose(RECIPE, _inkling(), 79.18, 512, 8, _all_offered,
                       refusal=lambda cfg: be.config_refusal(cfg, pm))
    assert [p.rule for p in found] == ["inkling-short-prompts-token-budget"]
    # The budget alone, at the recipe's batch, is what rigtune refuses: vLLM would run only 256 of 1,024.
    alone = RECIPE.model_copy(update={"prefill_budget": 2048})
    assert be.config_refusal(alone, pm)
    found, notes = propose(RECIPE, _inkling(), 79.18, 512, 8, _all_offered,
                           refusal=lambda cfg: "refused for the test")
    assert found == [] and any("would not launch (refused for the test)" in n for n in notes)


def test_a_model_s_unused_head_is_proposed_only_where_the_engine_offers_it():
    qwen = ArchInfo(architecture="Qwen3_5ForConditionalGeneration", num_layers=32, hidden_size=4096,
                    num_attention_heads=16, num_kv_heads=4, head_dim=256, vocab_size=248320, max_position_embeddings=262144,
                    mtp_layers=1)
    seed = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9)
    found, _ = propose(seed, qwen, 80, 512, 8, _all_offered)
    assert [(p.rule, p.config.spec_decode) for p in found] == [("qwen35-own-mtp-head-unused", "mtp:3")]
    found, notes = propose(seed, qwen, 80, 512, 8, lambda cfg: ["ngram_gpu:4", "mtp:1"])
    assert found == [] and any("not offered mtp:3 here (only mtp:1)" in n for n in notes)
    found, notes = propose(seed, qwen, 80, 512, 8, lambda cfg: None)
    assert found == [] and any("speculation is off" in n for n in notes)
    assert propose(seed, qwen, 80, 512, 16, _all_offered)[0] == []  # measured at 8 users
    # A seed that already drafts is left to its own drafter.
    assert propose(seed.model_copy(update={"spec_decode": "ngram_gpu:4"}), qwen, 80, 512, 8, _all_offered)[0] == []


def test_conditions_read_as_named():
    arch = _inkling()
    assert holds({"seed_spec": "any"}, RECIPE, arch, 80, 512, 8)
    assert not holds({"seed_spec": "none"}, RECIPE, arch, 80, 512, 8)
    assert holds({"max_users_at_least": 32, "max_users_at_most": 64}, RECIPE, arch, 80, 512, 32)
    assert not holds({"prefill_tokens_at_least": 4096}, RECIPE, arch, 80, 512, 32)


def test_a_rule_without_evidence_or_with_an_unknown_condition_is_refused(monkeypatch):
    import importlib.resources

    import rigtune.intent as intent

    def shipped(rules):
        class F:
            def joinpath(self, *a):
                return self

            def read_text(self, encoding=None):
                return json.dumps({"rules": rules})
        monkeypatch.setattr(importlib.resources, "files", lambda pkg: F())
        monkeypatch.setattr(intent, "resources", importlib.resources)

    good = {"id": "x", "when": {"seed_spec": "none"}, "try": {"batch": 32}, "because": "b", "evidence": "e"}
    for bad, why in [({**good, "evidence": ""}, "no evidence"), ({**good, "when": {"gpu_vendor": "amd"}}, "unknown"),
                     ({**good, "try": {"quant": "fp8"}}, "cannot change")]:
        shipped([bad])
        with pytest.raises(ValueError, match=why):
            load_rules()


def test_each_proposal_is_measured_right_after_the_seed_and_reported_against_it(hw_a100, prepared_vllm):
    from tests.test_seed_search import _feasible

    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    proposal = seed.model_copy(update={"batch": 128})  # FakeRunner: faster with the batch
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed,
                          seed_proposals=[("bigger-batch", proposal, "a reason")])
    _, notes = search.run(_feasible(hw_a100, prepared_vllm))
    # This seed's settings are a grid configuration the precision stage measured: it is not measured twice, and
    # that trial stands for it. The proposal is measured next.
    stages = [r.stage for r in search.results]
    assert "seed" not in stages and stages[stages.index("rule") - 1] == "quant"
    assert not any("did not run" in n for n in notes)
    note = next(n for n in notes if n.startswith("rule bigger-batch"))
    assert "a reason" in note and "against the seed's" in note and "+" in note


def test_proposals_are_not_tried_when_the_seed_did_not_run(hw_a100, prepared_vllm):
    from tests.test_seed_search import _feasible

    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.95)  # FakeRunner: OOM
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed,
                          seed_proposals=[("r", seed.model_copy(update={"batch": 128}), "why")])
    _, notes = search.run(_feasible(hw_a100, prepared_vllm))
    assert not any(r.stage == "rule" for r in search.results)
    assert any(n == "rule r not tried: the seed did not run" for n in notes)


def test_an_error_with_no_text_still_reads_as_a_line():
    from rigtune.calibrate.search import _error_head

    assert _error_head(None) == _error_head("") == _error_head("  \n ") == "no error text"
    assert _error_head("out of memory\ntraceback") == "out of memory"
