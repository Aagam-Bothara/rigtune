import pytest

from rigtune import pairing as p, speculative as sp
from rigtune.backends import vllm as vl, sglang as sg
from rigtune.models import Config
from tests.test_pairing import pair, target


def gemma(unified=True, assistant=False):
    arch = "Gemma4" + ("Unified" if unified else "") + ("AssistantForCausalLM" if assistant else "ForConditionalGeneration")
    return pair(config={"architectures": [arch], "model_type": "gemma4_unified_assistant" if assistant else "gemma4_unified",
                        "vocab_size": 262144}) if assistant else target(config={"architectures": [arch], "vocab_size": 262144})


def test_parse_round_trip_assist():
    assert sp.parse(sp.assist("google/gemma-4-12B-it-assistant", 1)) == ("assist", "google/gemma-4-12B-it-assistant", 1)


def test_vllm_config_is_mtp_with_model_and_k_1():
    assert sp.vllm_config("assist:org/assistant:1", revision="a" * 40) == {
        "method": "mtp", "model": "org/assistant", "num_speculative_tokens": 1, "revision": "a" * 40}


def test_sglang_flags_are_nextn_never_eagle3():
    args = sp.sglang_config("assist:org/assistant:3", draft_revision="b" * 40)
    assert args[1] == "NEXTN" and "EAGLE3" not in args
    assert args[args.index("--speculative-num-draft-tokens") + 1] == "4"


def test_unified_assistant_only_for_unified_target():
    assert p.check_pair(gemma(), gemma(assistant=True), "assistant", 4096).status == "ok"
    assert p.check_pair(gemma(False), gemma(assistant=True), "assistant", 4096).status == "refused"


def test_published_gemma4_configs_with_null_nested_architectures_pair():
    # The shapes of google/gemma-4-12B-it and its assistant as published (read on the A40 pod, 16 Nov 2026):
    # architectures at the top level, and text_config repeating the key as null.
    t = target(config={"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
                       "text_config": {"architectures": None, "model_type": "gemma4_unified_text",
                                       "vocab_size": 262144}})
    a = pair(config={"architectures": ["Gemma4UnifiedAssistantForCausalLM"], "model_type": "gemma4_unified_assistant",
                     "text_config": {"architectures": None, "model_type": "gemma4_unified_text", "vocab_size": 262144}})
    v = p.check_pair(t, a, "assistant", 4096)
    assert "architecture_unread" not in v.reasons and v.status == "ok", v.reasons
    assert p.text_config(t.config)["model_type"] == "gemma4_unified_text"  # a non-null nested value still wins


def test_non_gemma_target_gets_no_assist():
    assert p.check_pair(target(), gemma(assistant=True), "assistant", 4096).status == "refused"


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_assistant_is_offered_only_with_all(backend, monkeypatch, prepared_vllm, hw_a100):
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 33))
    monkeypatch.setattr(sg, "feature", lambda name: {"draft", "eagle3"} if name == "spec_methods" else None)
    pm = prepared_vllm
    pm.backend = backend
    pm.drafts_mode = "verified"
    pm.spec_pairs = [{"kind": "assistant", "repo": "org/assistant", "params": 10_000, "size_bytes": 20_000}]
    pm.spec_revisions["org/assistant"] = "a" * 40
    pm.spec_bytes["org/assistant"] = 20_000
    cfg = Config(backend=backend, quant="bf16", batch=16, gpu_memory_utilization=.9)
    be = vl.VllmBackend() if backend == "vllm" else sg.SglangBackend()
    assert not any(v.spec_decode.startswith("assist:") for v in be.spec_variants(cfg, pm, hw_a100))
    assert any(v.spec_decode.startswith("assist:") for v in be.spec_variants(cfg, pm, hw_a100, True))


def test_conformance_can_request_verified_pairs(prepared_vllm, hw_a100, tmp_path, monkeypatch):
    from rigtune.conformance import Conformance, CheckResult
    prepared_vllm.spec.revision = "a" * 40
    class Backend(vl.VllmBackend):
        def find_second_models(self, model):
            return []
        def find_pairs(self, model, ctx=4096, **kw):
            model.spec_pairs = [{"kind": "assistant", "repo": "org/assistant"}]
            return ["checked"]
        def spec_variants(self, cfg, model, *a, **kw):
            return [cfg.model_copy(update={"spec_decode": "assist:org/assistant:1"})] if model.spec_pairs else []
    run = Conformance("vllm", Backend(), hw_a100, {"small": prepared_vllm}, tmp_path, drafts="verified")
    monkeypatch.setattr(run, "_spec_check", lambda cid, *a: CheckResult(cid, "assistant", "pass"))
    try:
        assert [r.id for r in run.c06_speculative()] == ["C06-assist"]
    finally:
        run.http.close()
