"""The vLLM backend offers a checkpoint's own MTP head when the checkpoint holds one and vLLM, asked
(rigtune.backends.own_head), does not say it builds no draft from it; and plans its cache at every depth vLLM might
build. Until 20 Nov 2026 a list of four Qwen3.5 architectures read from vLLM's source decided, so Inkling's, DeepSeek's
and GLM's heads were never offered. That list is still what is checked when vLLM is not asked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rigtune.backends.vllm as vl
from rigtune import introspect, speculative
from rigtune.backends import own_head
from rigtune.backends.facts import EngineFacts
from rigtune.backends.vllm import VllmBackend, vllm_hybrid_kv, vllm_hybrid_kv_any_depth
from rigtune.memory import kv_cache_bytes
from tests.engine_trees import installed
from tests.test_hybrid_memory import qwen35
from tests.test_spec_mtp import HEAD_BYTES, SHARED_BYTES, WEIGHTS, specs, vcfg

HEAD = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=HEAD_BYTES,
                          shared_bytes=SHARED_BYTES)
INKLING = "InklingForConditionalGeneration"
PLANNED_AS_QWEN = (f"vllm: {INKLING}'s MTP head is planned as Qwen3.5's is (each depth a full-attention layer, and on "
                   "a model without linear or sliding-window layers no cache of its own), and the copy of the "
                   "embedding and output head it may build at load is not planned; neither was checked for this "
                   "family")


def _model(arch=None, mtp_layers=1):
    pm = qwen35(weights=dict(WEIGHTS))
    if arch:
        pm.arch.architecture = arch
    pm.arch.mtp_layers = mtp_layers
    return pm


@pytest.fixture
def world(monkeypatch):
    """Make the checkpoint hold `head` (or none) and vLLM answer `verdict`; returns what was asked of each."""
    asked = {"checkpoint": [], "vllm": []}

    def set_(verdict, head=HEAD):
        def answer(model, revision):
            asked["vllm"].append(model)
            return verdict

        def inspect(spec, **kw):
            asked["checkpoint"].append(spec.hf_id)
            return introspect.ModelFacts(mtp=head)

        monkeypatch.setattr(vl, "_own_head", answer)
        monkeypatch.setattr(introspect, "inspect_model", inspect)
        return asked
    return set_


YES = own_head.Verdict(True, "vllm builds InklingMTPModel with 8 MTP layers", "InklingMTPModel", 8)


def test_a_head_of_any_family_vllm_has_a_draft_for_is_offered_and_what_is_unchecked_is_said(
        world, tmp_path, tmp_home, hw_rtx4090):
    world(YES)
    pm = _model(INKLING)
    with installed("vllm", "0.33.0", tmp_path):
        be = VllmBackend()
        assert be.find_second_models(pm) == [PLANNED_AS_QWEN]
        assert pm.spec_repos == {"mtp": pm.spec.hf_id}
        assert {"mtp:1", "mtp:3"} <= set(specs(be.spec_variants(vcfg(), pm, hw_rtx4090, True)))


def test_qwen35_is_offered_as_before(world, tmp_path, tmp_home, hw_rtx4090):
    world(own_head.Verdict(True, "vllm builds Qwen3_5MTP with 1 MTP layer", "Qwen3_5MTP", 1))
    pm = _model()
    with installed("vllm", "0.33.0", tmp_path):
        be = VllmBackend()
        assert be.find_second_models(pm) == []
        assert (pm.spec_repos, pm.spec_bytes) == ({"mtp": pm.spec.hf_id}, {"mtp": HEAD_BYTES, "mtp_shared": SHARED_BYTES})
        assert specs(be.spec_variants(vcfg(), pm, hw_rtx4090, True))[-2:] == ["mtp:1", "mtp:3"]


def test_a_checkpoint_without_a_head_never_asks_vllm(world, tmp_path, tmp_home):
    asked = world(YES, head=None)
    pm = _model(INKLING, mtp_layers=0)
    with installed("vllm", "0.33.0", tmp_path):
        assert VllmBackend().find_second_models(pm) == [] and pm.spec_repos == {}
    assert asked == {"checkpoint": [pm.spec.hf_id], "vllm": []}  # vLLM takes seconds to import: not for nothing


def test_a_head_vllm_has_no_draft_for_is_not_tried_and_the_profile_says_why(world, tmp_path, tmp_home):
    world(own_head.Verdict(False, f"vllm has no MTP draft for {INKLING}"))
    pm = _model(INKLING)
    with installed("vllm", "0.33.0", tmp_path):
        notes = VllmBackend().find_second_models(pm)
    assert notes == [f"vllm: the checkpoint's MTP head not tried: vllm has no MTP draft for {INKLING}"]
    assert pm.spec_repos == {}


def test_a_head_vllm_could_not_be_asked_about_is_tried_with_a_note(world, tmp_path, tmp_home):
    world(own_head.Verdict(None, "vllm could not load the model's config (ValueError: gated)"))
    pm = _model(INKLING)
    with installed("vllm", "0.33.0", tmp_path):
        notes = VllmBackend().find_second_models(pm)
    assert pm.spec_repos == {"mtp": pm.spec.hf_id}
    assert notes == ["vllm: the checkpoint's MTP head is tried although whether vLLM loads it could not be told "
                     "(vllm could not load the model's config (ValueError: gated)); a start that fails costs one "
                     "trial", PLANNED_AS_QWEN]


def test_not_asking_vllm_reads_its_source_as_before(world, tmp_path, tmp_home):
    """Probing off (RIGTUNE_ENGINE_PROBE=off) or vLLM's Python not found: Qwen3.5 only, read from vLLM's source."""
    world(own_head.verdict(None))
    with installed("vllm", "0.33.0", tmp_path / "v"):
        qwen = _model()
        assert VllmBackend().find_second_models(qwen) == [] and qwen.spec_repos == {"mtp": qwen.spec.hf_id}
        other = _model(INKLING)
        assert VllmBackend().find_second_models(other) == [
            "vllm: the checkpoint's MTP head not tried: vLLM was not asked whether it loads one (the engine was not "
            "asked (probing is off, or its Python was not found)), and its source is read for the Qwen3.5 family "
            "only"]
        assert other.spec_repos == {}
    with installed("vllm", "0.33.0", tmp_path / "bare") as pkg:
        (pkg / "model_executor" / "models" / "qwen3_5_mtp.py").unlink()  # a release without the draft class
        bare = _model()
        notes = VllmBackend().find_second_models(bare)
        assert notes[0].endswith("and this vLLM's source has no Qwen3.5 MTP draft") and bare.spec_repos == {}


def test_vllm_is_asked_with_its_own_registry(monkeypatch, tmp_path):
    seen = {}

    def fake_ask(engine, python, package, model, revision, registered=None, **kw):
        seen.update(engine=engine, model=model, revision=revision, registered=registered)
        return own_head.Verdict(True, "yes"), None

    monkeypatch.setattr(own_head, "ask", fake_ask)
    with installed("vllm", "0.33.0", tmp_path):
        facts = EngineFacts(engine="vllm", version="0.33.0", archs=["InklingMTPModel", "LlamaForCausalLM"])
        monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
        assert vl._own_head("thinkingmachines/Inkling-Small", "abc1234").loads is True
    assert seen == {"engine": "vllm", "model": "thinkingmachines/Inkling-Small", "revision": "abc1234",
                    "registered": ["InklingMTPModel", "LlamaForCausalLM"]}


# ---------------------------------------------------------------- the variants


@pytest.mark.parametrize("loads", [True, None])  # None: vLLM gave its count but its registry lacks the class
def test_three_tokens_are_not_offered_on_a_two_layer_head(loads, world, tmp_path, tmp_home, hw_rtx4090):
    """vLLM refuses more tokens than layers unless a multiple of them (config/speculative.py, 1600-1611)."""
    world(own_head.Verdict(loads, "vllm builds XMTP with 2 MTP layers", "XMTP", 2))
    pm = _model(INKLING)
    with installed("vllm", "0.33.0", tmp_path):
        be = VllmBackend()
        be.find_second_models(pm)
        offered = specs(be.spec_variants(vcfg(), pm, hw_rtx4090, True))
    assert "mtp:1" in offered and "mtp:3" not in offered


def test_the_fp8_output_head_is_offered_for_the_family_it_was_read_for_only(world, monkeypatch, tmp_path, tmp_home,
                                                                            hw_rtx4090):
    facts = EngineFacts(engine="vllm", version="0.33.0", spec_fields=["draft_lm_head_quantization"])
    with installed("vllm", "0.33.0", tmp_path):
        monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
        be = VllmBackend()
        for arch, offered in ((None, True), (INKLING, False)):
            world(own_head.Verdict(True, "yes", "D", 1))
            pm = _model(arch)
            be.find_second_models(pm)
            heads = [v.spec_head_quant for v in be.spec_variants(vcfg(), pm, hw_rtx4090, True)]
            assert ("fp8" in heads) is offered, arch


# ---------------------------------------------------------------- the cache planned at every depth


def _hybrid(full, linear, mtp):
    return qwen35().arch.model_copy(update={"full_attention_layers": full, "linear_layers": linear, "mtp_layers": mtp})


def _bytes(arch, cfg, tokens, plan):
    return kv_cache_bytes(SimpleNamespace(arch=arch), cfg, int(tokens), plan)


@pytest.mark.parametrize("ctx, batch", [(4096, 64), (32768, 256), (8192, 16)])
def test_every_depth_is_planned_and_the_plan_bounds_each_at_every_token_count(ctx, batch):
    """With 24 linear layers, 11 full layers pad the linear ones to 33 pages and 12 to 24: one more MTP layer can
    plan less, and which depth needs more depends on the token count (the planner sizes a quarter of ctx x batch)."""
    arch, cfg = _hybrid(10, 24, 2), vcfg(spec_decode="mtp:3", batch=batch, ctx=ctx)
    one, two = (vllm_hybrid_kv(cfg, arch.model_copy(update={"mtp_layers": m})) for m in (1, 2))
    assert (one.full_layers, two.full_layers) == (11, 12) and one.state_bytes > two.state_bytes  # not monotonic
    plan = vllm_hybrid_kv_any_depth(cfg, arch)
    for share in (0.05, 0.25, 0.5, 1.0):
        tokens = share * ctx * batch
        assert _bytes(arch, cfg, tokens, plan) >= max(_bytes(arch, cfg, tokens, one), _bytes(arch, cfg, tokens, two))


@pytest.mark.parametrize("spec, mtp", [(None, 2), ("ngram_gpu:4", 2), ("mtp:1", 1)])
def test_without_a_multi_layer_head_the_plan_is_unchanged(spec, mtp):
    arch, cfg = _hybrid(10, 24, mtp), vcfg(spec_decode=spec)
    assert vllm_hybrid_kv_any_depth(cfg, arch) == vllm_hybrid_kv(cfg, arch)


def test_the_family_list_the_fallback_reads_is_qwen35s():
    assert speculative.MTP_ARCHS == {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                                     "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"}
