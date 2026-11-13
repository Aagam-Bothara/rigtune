"""vLLM's GPU n-gram lookup and suffix decoding as speculative-decoding candidates."""

from __future__ import annotations

import importlib.util
import json

import pytest

from rigtune import speculative


def test_the_new_methods_become_vllm_speculative_configs():
    assert speculative.vllm_config("ngram_gpu:4") == {
        "method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    assert speculative.vllm_config(speculative.suffix()) == {"method": "suffix", "num_speculative_tokens": 24}
    assert speculative.parse("suffix") == ("suffix", None, 24) and speculative.parse("ngram_gpu") == ("ngram_gpu", None, 4)
    with pytest.raises(ValueError):
        speculative.parse("eagle3:4")


def test_versions_gate_them():
    assert speculative.vllm_supports_ngram_gpu("0.29.0") and not speculative.vllm_supports_ngram_gpu("0.28.1")
    assert not speculative.vllm_supports_suffix("0.28.1")


def test_suffix_decoding_needs_arctic_inference(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "arctic_inference" else real(name, *a))
    assert not speculative.vllm_supports_suffix("0.29.0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "arctic_inference" else real(name, *a))
    assert speculative.vllm_supports_suffix("0.29.0")


@pytest.mark.parametrize("version, arctic, expected", [
    ("0.29.0", True, ["ngram_gpu:4", "suffix:24", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.29.0", False, ["ngram_gpu:4", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.28.1", True, ["ngram:4", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]),
    ("0.11.0", False, ["ngram:4"]),  # no separate draft model before 0.12
])
def test_vllm_offers_what_the_installed_release_supports(monkeypatch, prepared_vllm, version, arctic, expected):
    from rigtune.backends import get_backend
    from rigtune.models import Config

    monkeypatch.setattr(speculative, "_vllm_version", lambda: version)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if arctic else None) if name == "arctic_inference" else real(name, *a))
    cfg = Config(backend="vllm", quant="bf16", batch=64)
    model = prepared_vllm.model_copy(update={"spec": prepared_vllm.spec.model_copy(update={"hf_id": "Qwen/Qwen2.5-3B-Instruct"})})
    assert [c.spec_decode for c in get_backend("vllm").spec_variants(cfg, model)] == expected


# config.json of each head as read on the Hub (the fields the check uses).
LMSYS_LLAMA31_8B = {"architectures": ["LlamaForCausalLMEagle3"], "vocab_size": 128256, "hidden_size": 4096,
                    "num_hidden_layers": 1, "draft_vocab_size": 32000}
ANGELSLIM_QWEN3_8B = {"architectures": ["LlamaForCausalLMEagle3"], "vocab_size": 151936, "hidden_size": 4096}
YUHUILI_LLAMA31_8B = {"architectures": ["LlamaForCausalLM"], "vocab_size": 128256, "hidden_size": 4096}


def _arch(vocab: int, hidden: int = 4096):
    from rigtune.models import ArchInfo

    return ArchInfo(architecture="LlamaForCausalLM", num_layers=32, hidden_size=hidden, num_attention_heads=32,
                    num_kv_heads=8, head_dim=128, vocab_size=vocab)


def test_an_eagle3_spec_names_its_head_and_becomes_a_vllm_config():
    s = speculative.eagle3("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    assert s == "eagle3:lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B:3"
    assert speculative.parse(s) == ("eagle3", "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B", 3)
    assert speculative.parse(speculative.eagle3("a/b", 5)) == ("eagle3", "a/b", 5)
    assert speculative.vllm_config(s) == {"method": "eagle3", "model": "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
                                          "num_speculative_tokens": 3}
    with pytest.raises(ValueError):
        speculative.parse("eagle3:3")  # no head


def test_a_head_is_used_only_when_its_config_shows_an_eagle3_head_that_fits():
    llama, qwen3 = _arch(128256), _arch(151936)
    assert speculative.verified_eagle3_head(LMSYS_LLAMA31_8B, llama)
    assert speculative.verified_eagle3_head(ANGELSLIM_QWEN3_8B, qwen3)
    assert not speculative.verified_eagle3_head(YUHUILI_LLAMA31_8B, llama)  # plain LlamaForCausalLM
    assert not speculative.verified_eagle3_head({**ANGELSLIM_QWEN3_8B}, _arch(152064))  # another vocabulary
    assert not speculative.verified_eagle3_head(LMSYS_LLAMA31_8B, _arch(128256, hidden=3072))
    assert not speculative.verified_eagle3_head({**LMSYS_LLAMA31_8B, "architectures": ["Eagle3Speculator"]}, llama)
    assert not speculative.verified_eagle3_head(None, llama) and not speculative.verified_eagle3_head({}, llama)


def test_finding_a_head_reads_only_the_listed_repositories():
    from rigtune.models import ModelSpec

    asked = []

    def fetch(repo):
        asked.append(repo)
        return {"lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B": LMSYS_LLAMA31_8B}.get(repo)

    sizes = {"lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B": 850_000_000}.get
    for hf_id in ("unsloth/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"):
        assert speculative.find_eagle3_head(ModelSpec(hf_id=hf_id), _arch(128256), fetch_config=fetch,
                                            sizes=sizes) == ("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B", 850_000_000)
    assert speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), _arch(152064),
                                        fetch_config=fetch, sizes=sizes) is None  # not listed: nothing asked
    assert asked == ["lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"] * 2
    # A listed head whose config no longer fits is not used.
    assert speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen3-8B"), _arch(151936),
                                        fetch_config=lambda r: YUHUILI_LLAMA31_8B, sizes=sizes) is None
    assert speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen3-8B"), _arch(151936),
                                        fetch_config=lambda r: ANGELSLIM_QWEN3_8B,
                                        sizes=lambda r: None) == ("AngelSlim/Qwen3-8B_eagle3", None)


def test_a_hub_that_cannot_be_asked_is_told_apart_from_no_head_fitting():
    from rigtune.models import ModelSpec

    def offline(repo):
        raise OSError("no network")

    with pytest.raises(speculative.HubLookupError, match="no network"):
        speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen3-8B"), _arch(151936), fetch_config=offline,
                                     sizes=lambda r: 1)

    def no_size(repo):
        raise OSError("rate limited")

    # the head fits but its size cannot be read: found, size unknown
    assert speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen3-8B"), _arch(151936),
                                        fetch_config=lambda r: ANGELSLIM_QWEN3_8B,
                                        sizes=no_size) == ("AngelSlim/Qwen3-8B_eagle3", None)
    assert speculative.eagle3_heads_for("meta-llama/Llama-3.3-70B-Instruct") == ()  # no head whose config was read


HEAD = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"


@pytest.fixture
def llama31(prepared_vllm):
    return prepared_vllm.model_copy(update={
        "spec": prepared_vllm.spec.model_copy(update={"hf_id": "unsloth/Meta-Llama-3.1-8B-Instruct"}),
        "spec_repos": {"eagle3": HEAD}})


def test_vllm_is_offered_the_same_eagle3_head_under_all(monkeypatch, llama31):
    from rigtune.backends import get_backend
    from rigtune.models import Config

    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")
    monkeypatch.setattr(speculative, "vllm_supports_suffix", lambda version=None: False)
    be = get_backend("vllm")
    cfg = Config(backend="vllm", quant="fp8", batch=16, gpu_memory_utilization=0.95)
    before = ["ngram_gpu:4", "draft:unsloth/Llama-3.2-1B-Instruct:4"]
    assert [c.spec_decode for c in be.spec_variants(cfg, llama31)] == before  # --speculative on: unchanged
    out = be.spec_variants(cfg, llama31, None, True)
    assert [c.spec_decode for c in out] == before + [f"eagle3:{HEAD}:3"]
    assert {c.gpu_memory_utilization for c in out} == {0.95}  # vLLM's planner is unchanged
    args = be.launch_spec(out[-1], llama31, 1).args
    assert json.loads(args[args.index("--speculative-config") + 1]) == {
        "method": "eagle3", "model": HEAD, "num_speculative_tokens": 3}
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.28.1")  # not checked before 0.29
    assert not any((c.spec_decode or "").startswith("eagle3") for c in be.spec_variants(cfg, llama31, None, True))
    assert speculative.vllm_supports_eagle3("0.29.0") and not speculative.vllm_supports_eagle3("0.28.1")


def test_eagle3_is_kept_off_the_fp8_cache_that_broke_the_draft_model_on_ampere(monkeypatch, hw_a100, llama31):
    from rigtune.backends import registry
    from rigtune.backends.vllm import AMPERE_FP8_KV
    from rigtune.models import Config
    from rigtune.pipeline import PlanResult, combination_fits, kv_variants_fn

    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")
    reg = registry()
    be = reg["vllm"]
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    on_fp8 = cfg.model_copy(update={"kv_dtype": AMPERE_FP8_KV})
    assert not any((c.spec_decode or "").startswith("eagle3") for c in be.spec_variants(on_fp8, llama31, None, True))
    assert any((c.spec_decode or "").startswith("draft") for c in be.spec_variants(on_fp8, llama31, None, True))
    lead = cfg.model_copy(update={"spec_decode": f"eagle3:{HEAD}:3"})
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "flashinfer" else real(name, *a))
    plan = PlanResult(hw=hw_a100, spec=llama31.spec, candidates=["vllm"], prepared={"vllm": llama31})
    assert be.kv_dtypes(hw_a100) == [AMPERE_FP8_KV]
    assert kv_variants_fn(hw_a100, reg, plan)(lead) == []
    assert kv_variants_fn(hw_a100, reg, plan)(cfg.model_copy(update={"spec_decode": "draft:x:4"}))  # as before
    fits = combination_fits(hw_a100, reg, plan)
    assert not fits(lead.model_copy(update={"kv_dtype": AMPERE_FP8_KV})) and fits(lead)


def test_vllm_looks_the_head_up_only_under_all_and_notes_an_unreachable_hub(monkeypatch, hw_rtx4090, llama3b_arch,
                                                                             no_network):
    import rigtune.backends.vllm as vl
    from rigtune.backends import registry
    from rigtune.models import ModelSpec
    from rigtune.pipeline import prepare_and_plan

    asked = []
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")
    monkeypatch.setattr(vl, "hf_weight_options", lambda *a, **k: ({"bf16": 1, "fp8": 1}, {}))
    monkeypatch.setattr(speculative, "find_eagle3_head", lambda spec, arch: asked.append(spec.hf_id) or (HEAD, None))
    reg, llama = registry(), ModelSpec(hf_id="unsloth/Meta-Llama-3.1-8B-Instruct")
    assert prepare_and_plan(hw_rtx4090, llama, ["vllm"], reg).prepared["vllm"].spec_repos == {}
    assert asked == []  # --speculative on: no lookup
    plan = prepare_and_plan(hw_rtx4090, llama, ["vllm"], reg, speculative_all=True)
    assert plan.prepared["vllm"].spec_repos == {"eagle3": HEAD} and plan.notes == []
    assert prepare_and_plan(hw_rtx4090, ModelSpec(hf_id="Qwen/Qwen2.5-3B-Instruct"), ["vllm"], reg,
                            speculative_all=True).prepared["vllm"].spec_repos == {}
    assert asked == [llama.hf_id]  # a model with no listed head asks nothing

    def offline(*a, **k):
        raise speculative.HubLookupError("could not read: no network")

    monkeypatch.setattr(speculative, "find_eagle3_head", offline)
    plan = prepare_and_plan(hw_rtx4090, llama, ["vllm"], reg, speculative_all=True)
    assert plan.notes and "the Hub could not be asked" in plan.notes[0]
