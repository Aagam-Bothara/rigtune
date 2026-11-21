"""The SGLang backend offers a checkpoint's own MTP head as the vLLM backend does (tests/test_vllm_own_head.py): when
the checkpoint holds one and SGLang, asked (rigtune.backends.own_head), does not say it builds no draft from it. Its
rules rename by architecture alone, so the checkpoint's own answer is what keeps a head that is not there from being
offered; and its KV pool is sized by the layers its draft reads, which the planner takes from it."""

from __future__ import annotations

import pytest

import rigtune.backends.sglang as sg
from rigtune import introspect
from rigtune.backends import own_head
from rigtune.backends.facts import EngineFacts
from rigtune.backends.sglang import SglangBackend, sglang_hybrid_kv
from tests.engine_trees import installed
from tests.test_hybrid_memory import qwen35
from tests.test_spec_mtp import HEAD_BYTES, SHARED_BYTES, WEIGHTS, scfg, specs

HEAD = introspect.MtpHead(layers=8, config_key="mtp_config.num_nextn_predict_layers", tensors=160, bytes=HEAD_BYTES,
                          shared_bytes=SHARED_BYTES)
INKLING = "InklingForConditionalGeneration"
PLANNED_AS_QWEN = (f"sglang: {INKLING}'s MTP head: its draft's KV cache is planned only on a model with "
                   "linear-attention layers, as Qwen3.5's is, and not otherwise, though SGLang reserves one; the "
                   "draft's copy of the embedding and output head is counted as a single-step draft holds it; neither "
                   "was checked for this family")


def _model(arch=None, mtp_layers=1):
    pm = qwen35("sglang", dict(WEIGHTS))
    if arch:
        pm.arch.architecture = arch
    pm.arch.mtp_layers = mtp_layers
    return pm


@pytest.fixture
def world(monkeypatch):
    """Make the checkpoint hold `head` (or none) and SGLang answer `verdict`; returns what was asked of each."""
    asked = {"checkpoint": [], "sglang": []}

    def set_(verdict, head=HEAD):
        def answer(model, revision):
            asked["sglang"].append(model)
            return verdict

        def inspect(spec, **kw):
            asked["checkpoint"].append(spec.hf_id)
            return introspect.ModelFacts(mtp=head)

        monkeypatch.setattr(sg, "_own_head", answer)
        monkeypatch.setattr(introspect, "inspect_model", inspect)
        return asked
    return set_


def test_a_head_of_any_family_sglang_has_a_draft_for_is_offered_at_sglangs_own_depth(
        world, tmp_path, tmp_home, hw_a100):
    world(own_head.Verdict(True, "sglang builds InklingForConditionalGenerationMTP", "InklingForConditionalGenerationMTP",
                           1))
    pm = _model(INKLING)
    with installed("sglang", "0.5.23", tmp_path):
        be = SglangBackend()
        assert be.find_second_models(pm) == [PLANNED_AS_QWEN]
        assert pm.spec_repos == {"mtp": pm.spec.hf_id}
        assert pm.arch.mtp_layers == 1  # SGLang's draft reads one layer, not the checkpoint's eight
        assert {"mtp:1", "mtp:3"} <= set(specs(be.spec_variants(scfg(batch=4), pm, hw_a100, True)))


def test_a_checkpoint_without_a_head_is_never_offered_one_whatever_sglang_renames(world, tmp_path, tmp_home):
    """SGLang renames Qwen3MoeForCausalLM to Qwen3MoeForCausalLMMTP with or without MTP weights."""
    asked = world(own_head.Verdict(True, "sglang builds Qwen3MoeForCausalLMMTP"), head=None)
    pm = _model("Qwen3MoeForCausalLM", mtp_layers=0)
    with installed("sglang", "0.5.23", tmp_path):
        assert SglangBackend().find_second_models(pm) == [] and pm.spec_repos == {}
    assert asked["sglang"] == []


def test_a_head_sglang_has_no_draft_for_is_not_tried_and_the_profile_says_why(world, tmp_path, tmp_home):
    world(own_head.Verdict(False, f"sglang has no MTP draft for {INKLING}"))
    pm = _model(INKLING)
    with installed("sglang", "0.5.23", tmp_path):
        notes = SglangBackend().find_second_models(pm)
    assert notes == [f"sglang: the checkpoint's MTP head not tried: sglang has no MTP draft for {INKLING}"]
    assert pm.spec_repos == {}


def test_a_head_sglang_could_not_be_asked_about_is_tried_with_a_note(world, tmp_path, tmp_home):
    world(own_head.Verdict(None, "sglang could not load the model's config (ImportError: x)"))
    pm = _model("Glm4MoeForCausalLM")  # a family SGLang runs on its single-layer worker
    with installed("sglang", "0.5.23", tmp_path):
        notes = SglangBackend().find_second_models(pm)
    assert pm.spec_repos == {"mtp": pm.spec.hf_id} and pm.arch.mtp_layers == 8  # no count from SGLang: the checkpoint's
    assert notes[0].startswith("sglang: the checkpoint's MTP head is tried although whether SGLang loads it could not")


def test_not_asking_sglang_reads_its_source_as_before(world, tmp_path, tmp_home):
    world(own_head.verdict(None))
    with installed("sglang", "0.5.23", tmp_path / "s"):
        qwen = _model()
        assert SglangBackend().find_second_models(qwen) == [] and qwen.spec_repos == {"mtp": qwen.spec.hf_id}
        other = _model(INKLING)
        assert SglangBackend().find_second_models(other)[0].endswith("its source is read for the Qwen3.5 family only")
        assert other.spec_repos == {}
    with installed("sglang", "0.5.23", tmp_path / "s2") as pkg:
        (pkg / "srt" / "configs" / "model_config.py").write_text("", encoding="utf-8")  # no Qwen3.5 rewrite
        bare = _model()
        notes = SglangBackend().find_second_models(bare)
        assert notes[0].endswith("and this SGLang's source has no Qwen3.5 MTP draft") and bare.spec_repos == {}


def test_a_head_whose_size_is_unknown_is_still_noted(world, tmp_path, tmp_home):
    sizeless = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=None)
    world(own_head.Verdict(True, "sglang builds Qwen3_5ForCausalLMMTP", "Qwen3_5ForCausalLMMTP", 1), head=sizeless)
    pm = _model()
    with installed("sglang", "0.5.23", tmp_path):
        notes = SglangBackend().find_second_models(pm)
    assert notes == [f"sglang: MTP head of {pm.spec.hf_id} not tried, its size could not be read"]


def test_sglang_is_asked_with_its_own_registry(monkeypatch, tmp_path):
    seen = {}

    def fake_ask(engine, python, package, model, revision, registered=None, **kw):
        seen.update(engine=engine, model=model, revision=revision, registered=registered)
        return own_head.Verdict(True, "yes"), None

    monkeypatch.setattr(own_head, "ask", fake_ask)
    with installed("sglang", "0.5.23", tmp_path):
        facts = EngineFacts(engine="sglang", version="0.5.23", archs=["InklingForConditionalGenerationMTP"])
        monkeypatch.setattr(sg, "_engine_facts", lambda: facts)
        assert sg._own_head("thinkingmachines/Inkling-Small", "abc1234").loads is True
    assert seen == {"engine": "sglang", "model": "thinkingmachines/Inkling-Small", "revision": "abc1234",
                    "registered": ["InklingForConditionalGenerationMTP"]}


@pytest.mark.parametrize("mtp_layers, added", [(1, 1), (0, 1), (2, 2)])
def test_the_kv_pool_counts_the_layers_sglang_s_draft_reads(mtp_layers, added):
    """DefaultPoolConfigurator scales the cell by 1 + draft layers / full-attention layers, the draft's layers being
    its num_nextn_predict_layers: Qwen3.5's 1 as before, another count when SGLang said so."""
    arch = qwen35().arch.model_copy(update={"mtp_layers": mtp_layers})
    plan = sglang_hybrid_kv(scfg(spec_decode="mtp:1"), arch, 80 * 2**30, "0.5.23")
    assert plan.full_layers == arch.full_attention_layers + added


def test_a_release_whose_second_model_behaviour_was_not_read_is_offered_no_head(world, monkeypatch, tmp_path,
                                                                                  tmp_home):
    """As sglang_mtp_archs did: no head before the releases whose second-model behaviour was read (DSPARK)."""
    asked = world(own_head.Verdict(True, "sglang builds Qwen3_5ForCausalLMMTP", "Qwen3_5ForCausalLMMTP", 1))
    with installed("sglang", "0.5.23", tmp_path):
        monkeypatch.setattr(sg, "feature", lambda name: frozenset({"ngram"}) if name == "spec_methods" else None)
        pm = _model()
        assert SglangBackend().find_second_models(pm) == [] and pm.spec_repos == {}
    assert asked == {"checkpoint": [], "sglang": []}


@pytest.mark.parametrize("verdict", [
    own_head.Verdict(True, "sglang builds InklingForConditionalGenerationMTP", "InklingForConditionalGenerationMTP", 8,
                     banded=True),  # Inkling-Small's mtp_config.local_layer_ids
    own_head.Verdict(True, "sglang builds MiMoV2MTP", "MiMoV2MTP", 1),
    own_head.Verdict(None, "registry does not list Step3p5MTP", "Step3p5MTP", 3),
    own_head.Verdict(None, "sglang could not load the model's config (ValueError: x)"),  # no class: by the target
])
def test_a_head_sglang_runs_on_its_multi_layer_worker_is_not_offered(verdict, world, tmp_path, tmp_home):
    world(verdict)
    pm = _model(INKLING)
    with installed("sglang", "0.5.23", tmp_path):
        notes = SglangBackend().find_second_models(pm)
    assert notes == [f"sglang: the checkpoint's MTP head not tried: SGLang runs {verdict.draft_arch or INKLING}'s MTP "
                     "head only on its multi-layer EAGLE worker, one model runner and one copy of the embedding and "
                     "output head per step, which rigtune does not plan yet"]
    assert pm.spec_repos == {}


def test_qwen35_is_offered_the_same_with_sglang_asked_or_not(world, tmp_path, tmp_home, hw_rtx4090):
    qwen_head = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=HEAD_BYTES,
                                   shared_bytes=SHARED_BYTES)

    def offered(verdict):
        world(verdict, head=qwen_head)
        pm = _model()
        with installed("sglang", "0.5.23", tmp_path):
            be = SglangBackend()
            notes = be.find_second_models(pm)
            got = [(v.spec_decode, v.gpu_memory_utilization, v.batch)
                   for v in be.spec_variants(scfg(batch=4), pm, hw_rtx4090, True)]
        return notes, pm.arch.mtp_layers, got

    asked = offered(own_head.Verdict(True, "sglang builds Qwen3_5ForCausalLMMTP", "Qwen3_5ForCausalLMMTP", 1))
    assert asked == offered(own_head.verdict(None)) and asked[0] == [] and ("mtp:3", 0.71, 4) in asked[2]
