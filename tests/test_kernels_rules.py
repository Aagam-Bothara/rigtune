"""Rules for kernel settings an engine would refuse, or that are what it does anyway.

Each rule holds only for the releases it was read in. Outside them, or with a version that cannot be read, a rule
says nothing: the setting is neither refused nor called the same as the default."""

from __future__ import annotations

import pytest

import rigtune.backends.vllm as vl
from rigtune import kernels
from rigtune.models import Config
from tests.kernel_facts import vllm_facts

A40, RTX4090, T4, H100 = 46068.0, 24564.0, 15360.0, 81559.0  # MiB, as nvidia-smi reports each card
UNREAD = ["0.28.0", "0.34.0", "1.0.0", "", None, "main", "0.5"]


def P(text: str):
    return kernels.parse(text)


# ---- vLLM

@pytest.mark.parametrize("version", ["0.29.0", "0.31.2", "0.33.0", "0.33.1.dev7+g1a2b3c"])
def test_o3_equals_default(version):
    assert "default optimisation level" in kernels.vllm_same_as_default(P("O=3"), version)
    assert "default optimisation level" in kernels.vllm_same_as_default(P("O=2"), version)
    assert kernels.vllm_same_as_default(P("O=1"), version) is None
    assert kernels.vllm_same_as_default(P("O=0"), version) is None


def test_default_graph_mode_is_the_default_only_at_the_default_level():
    assert "default CUDA-graph mode" in kernels.vllm_same_as_default(P("cg=FULL_AND_PIECEWISE"), "0.33.0")
    both = kernels.vllm_same_as_default(P("cg=FULL_AND_PIECEWISE,O=3"), "0.33.0")
    assert "default CUDA-graph mode" in both and "default optimisation level" in both  # each rule is named
    assert kernels.vllm_same_as_default(P("cg=FULL_AND_PIECEWISE,O=1"), "0.33.0") is None  # O1 has another mode
    assert kernels.vllm_same_as_default(P("cg=FULL_AND_PIECEWISE,attn=TRITON_ATTN"), "0.33.0") is None


@pytest.mark.parametrize("text", ["cg=FULL", "cg=FULL_DECODE_ONLY", "cg=PIECEWISE", "cg=NONE", "cgmax=64", "gp=1"])
def test_full_on_a_backend_without_full_support_is_left_to_be_tried(text):
    """FULL becomes FULL_AND_PIECEWISE or FULL_DECODE_ONLY by how the model is compiled, which is not known here:
    no equivalence is claimed, and the trial's own record of what the engine changed decides."""
    for version in ("0.29.0", "0.33.0"):
        assert kernels.vllm_same_as_default(P(text), version, attention="FLASHINFER", capability=(8, 6)) is None


def test_flashinfer_spec_is_piecewise_before_0_33():
    same = kernels.vllm_same_as_default(P("cg=PIECEWISE"), "0.29.0", attention="FLASHINFER", speculating=True,
                                        capability=(8, 6))
    assert same and "0.29" in same and "FlashInfer" in same
    for other in ({"speculating": False}, {"attention": "TRITON_ATTN"}, {"attention": None}, {"capability": (10, 0)},
                  {"capability": None}):
        kw = {"attention": "FLASHINFER", "speculating": True, "capability": (8, 6), **other}
        assert kernels.vllm_same_as_default(P("cg=PIECEWISE"), "0.29.0", **kw) is None


@pytest.mark.parametrize("version", ["0.30.0", "0.32.1", "0.33.0"])
def test_flashinfer_spec_keeps_full_on_0_33_below_sm90(version):
    """0.33 keeps full decode graphs there, and the releases between were not read: PIECEWISE is a real change."""
    assert kernels.vllm_same_as_default(P("cg=PIECEWISE"), version, attention="FLASHINFER", speculating=True,
                                        capability=(8, 6)) is None


@pytest.mark.parametrize("attn, kv, refused", [
    ("FLEX_ATTENTION", "fp8_e5m2", True), ("FLASH_ATTN", "fp8_e5m2", True), ("FLASH_ATTN", "int8_per_token_head", True),
    ("FLASHINFER", "int8_per_token_head", True), ("TRITON_ATTN", "fp8_e5m2", False),
    ("TRITON_ATTN", "int8_per_token_head", False), ("FLASHINFER", "fp8_e5m2", False), ("FLASH_ATTN", "auto", False),
    ("TRITON_FLASH_ATTN", "fp8_e5m2", False),  # a backend whose list was not read: nothing is said
])
def test_attention_backend_and_cache_type(attn, kv, refused):
    for version in ("0.29.0", "0.33.0"):
        reason = kernels.vllm_refusal(P(f"attn={attn}"), kv, version)
        assert (reason is not None) == refused
        if refused:
            assert reason.startswith(f"vLLM's {attn} attention does not take a {kv} KV cache")


@pytest.mark.parametrize("version", UNREAD)
def test_unknown_version_is_unknown_never_equivalent_never_refused(version):
    assert kernels.vllm_same_as_default(P("O=3"), version) is None
    assert kernels.vllm_same_as_default(P("cg=FULL_AND_PIECEWISE"), version) is None
    assert kernels.vllm_refusal(P("attn=FLEX_ATTENTION"), "fp8_e5m2", version) is None
    assert kernels.sglang_default_decode_cap(A40, 64, 1, version) is None
    assert kernels.sglang_same_as_default(P("cgmax=32"), A40, 64, 1, version) is None


def test_missing_facts_refuse_with_not_asked(prepared_vllm, monkeypatch):
    cfg = Config(backend="vllm", quant="bf16", kv_dtype="fp8_e5m2", kernels="attn=FLEX_ATTENTION")
    monkeypatch.setattr(vl, "_engine_facts", lambda: None)
    assert vl.VllmBackend().config_refusal(cfg, prepared_vllm) == kernels.NOT_ASKED  # asked first, never guessed
    monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts("0.33.0"))
    assert vl.VllmBackend().config_refusal(cfg, prepared_vllm).startswith("vLLM's FLEX_ATTENTION attention")
    monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts("0.40.0"))
    assert vl.VllmBackend().config_refusal(cfg, prepared_vllm) is None  # a release no rule was read for


def test_rule_exception_is_raised_not_a_refusal(prepared_vllm, monkeypatch):
    """An error inside a rule is a defect to see, not a reason to drop the setting."""
    def broken(settings, kv_dtype, version):
        raise KeyError("a rule that fails")

    monkeypatch.setattr(kernels, "vllm_refusal", broken)
    monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts())
    cfg = Config(backend="vllm", quant="bf16", kernels="attn=TRITON_ATTN")
    with pytest.raises(KeyError):
        vl.VllmBackend().config_refusal(cfg, prepared_vllm)


def test_plans_without_kernels_identical(prepared_vllm, monkeypatch):
    """No rule is consulted for a config without kernel settings, whatever the release."""
    monkeypatch.setattr(kernels, "vllm_refusal", lambda *a: pytest.fail("a rule ran without kernel settings"))
    monkeypatch.setattr(kernels, "vllm_unconfirmed", lambda *a: pytest.fail("the probe was consulted"))
    monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts())
    for kv in ("auto", "fp8_e5m2"):
        assert vl.VllmBackend().config_refusal(Config(backend="vllm", quant="bf16", kv_dtype=kv), prepared_vllm) is None


# ---- SGLang

def test_tier_table_per_release():
    cap = kernels.sglang_default_decode_cap
    assert (cap(RTX4090, 256, 1, "0.5.19"), cap(RTX4090, 256, 1, "0.5.23")) == (24, 48)
    assert (cap(RTX4090, 256, 4, "0.5.19"), cap(RTX4090, 256, 4, "0.5.23")) == (80, 160)
    assert (cap(A40, 256, 1, "0.5.19"), cap(A40, 256, 1, "0.5.23")) == (32, 32)  # the same on both: the low tier now
    assert (cap(T4, 256, 1, "0.5.19"), cap(T4, 256, 1, "0.5.23")) == (8, 8)
    assert (cap(H100, 512, 1, "0.5.19"), cap(H100, 512, 1, "0.5.23")) == (256, 256)
    assert cap(200 * 1024, 1024, 1, "0.5.23") == 512
    assert cap(0, 64, 1, "0.5.23") is None  # a card whose memory is not known has no known default


def test_sglang_default_cap_is_min_of_tier_and_batch_on_0_5_23():
    assert kernels.sglang_default_decode_cap(A40, 16, 1, "0.5.23") == 16
    assert kernels.sglang_default_decode_cap(A40, 64, 1, "0.5.23") == 32
    assert "SGLang picks" in kernels.sglang_same_as_default(P("cgmax=16"), A40, 16, 1, "0.5.23")
    assert kernels.sglang_same_as_default(P("cgmax=32"), A40, 16, 1, "0.5.23") is None  # above the clamp: a change


def test_sglang_default_cap_ignores_batch_on_0_5_19():
    assert kernels.sglang_default_decode_cap(A40, 16, 1, "0.5.19") == 32
    assert kernels.sglang_same_as_default(P("cgmax=32"), A40, 16, 1, "0.5.19")
    assert kernels.sglang_same_as_default(P("cgmax=16"), A40, 16, 1, "0.5.19") is None


def test_equivalent_setting_is_not_launched_and_note_names_the_rule():
    """Raising the cap to what the card already gets is no trial; raising it past that is one."""
    same = kernels.sglang_same_as_default(P("cgmax=32"), A40, 64, 1, "0.5.23")
    assert same == "cgmax=32 is the decode cap SGLang picks for this card and batch"
    assert kernels.sglang_same_as_default(P("cgmax=64"), A40, 64, 1, "0.5.23") is None
    assert kernels.sglang_same_as_default(P("cgmax=32,attn=triton"), A40, 64, 1, "0.5.23") is None
