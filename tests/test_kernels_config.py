"""Config.kernels: one spelling, one key, rendered for vLLM only when the engine probe confirmed each setting.

With kernels unset, every config key, saved config and launch is what it was (tests/test_kernels_off.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rigtune.backends.vllm as vl
from rigtune import kernels
from rigtune.models import Config
from tests.kernel_facts import VLLM_KERNEL, vllm_facts

GOLDEN = Path(__file__).parent / "golden" / "kernels" / "vllm-args.json"
CFG = Config(backend="vllm", quant="bf16", ctx=4096, batch=32, gpu_memory_utilization=0.9)
STRINGS = ["attn=TRITON_ATTN", "cg=FULL", "cg=FULL,attn=TRITON_ATTN", "cgmax=64", "cgsizes=8+1+4+2", "O=1",
           "mode=STOCK_TORCH_COMPILE", "gp=1", "pass.fuse_norm_quant=1,pass.eliminate_noops=0",
           "O=1,cg=PIECEWISE,cgmax=128,pass.fuse_act_quant=true"]


def _with(kern: str, **update) -> Config:
    return CFG.model_copy(update={**update, "kernels": kernels.canonical(kern)})


def _launch(cfg: Config, prepared, monkeypatch, facts=None):
    monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts() if facts is None else facts)
    return vl.VllmBackend().launch_spec(cfg, prepared, 8000)


def test_canonical_order_and_round_trip():
    assert kernels.canonical(" cg=FULL , attn=TRITON_ATTN ") == "attn=TRITON_ATTN,cg=FULL"
    assert kernels.canonical("cgsizes=8+1+4+2+8") == "cgsizes=1+2+4+8"
    assert kernels.canonical("cgmax=064,gp=TRUE,pass.x=off") == "cgmax=64,gp=1,pass.x=0"
    for text in STRINGS:
        once = kernels.canonical(text)
        assert kernels.canonical(once) == once and kernels.format_settings(kernels.parse(once)) == once


@pytest.mark.parametrize("text", ["warp=9", "attn", "attn=", "=FULL", "", "cg=FULL,,O=1", "cg=FULL,cg=NONE",
                                  "cgmax=-4", "cgmax=6.5", "O=two", "gp=maybe", "cgsizes=1+x", "cgsizes=0+4",
                                  "attn=TRITON ATTN", "attn=../x", "pass.=1"])
def test_unknown_key_rejected(text):
    with pytest.raises(ValueError):
        kernels.parse(text)
    with pytest.raises(ValueError):
        Config(backend="vllm", quant="bf16", kernels=text)


@pytest.mark.parametrize("key", kernels.RESERVED)
def test_phase_b_keys_rejected_until_built(key):
    with pytest.raises(ValueError, match="not built yet"):
        kernels.parse(f"{key}=x")


def test_key_unchanged_when_unset():
    assert CFG.kernels is None and "kern" not in CFG.key()
    assert CFG.key() == "vllm/bf16/ctx4096/b32/gmu0.90"
    assert "kernels" not in CFG.model_dump(mode="json") and "kernels" not in json.loads(CFG.model_dump_json())
    assert Config.model_validate(CFG.model_dump(mode="json")) == CFG


def test_key_same_for_any_spelling_order():
    a = Config(backend="vllm", quant="bf16", kernels="cg=FULL,attn=TRITON_ATTN,cgsizes=4+1")
    b = Config(backend="vllm", quant="bf16", kernels="cgsizes=1+4, attn=TRITON_ATTN ,cg=FULL")
    assert a.kernels == b.kernels == "attn=TRITON_ATTN,cg=FULL,cgsizes=1+4" and a.key() == b.key()
    assert a.key().endswith("/kern=attn=TRITON_ATTN,cg=FULL,cgsizes=1+4")
    again = Config.model_validate_json(a.model_dump_json())  # what a saved profile hands back
    assert again == a and again.key() == a.key()


def test_compilation_config_json_is_sorted_and_compact():
    cc = kernels.vllm_compilation_config(kernels.parse("pass.fuse_norm_quant=1,cgsizes=4+1,cg=FULL,gp=0,cgmax=64"))
    assert cc == ('{"cudagraph_capture_sizes":[1,4],"cudagraph_mode":"FULL","max_cudagraph_capture_size":64,'
                  '"pass_config":{"fuse_norm_quant":true},"use_inductor_graph_partition":false}')
    assert " " not in cc and kernels.vllm_compilation_config(kernels.parse("attn=TRITON_ATTN,O=1")) is None


def test_attention_backend_and_cc_are_rendered(prepared_vllm, monkeypatch):
    spec = _launch(_with("cg=FULL,attn=TRITON_ATTN,O=1"), prepared_vllm, monkeypatch)
    args = spec.args
    assert args[args.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert args[args.index("--compilation-config") + 1] == '{"cudagraph_mode":"FULL"}'  # one argument, no quoting
    assert args[args.index("--optimization-level") + 1] == "1"
    assert spec.env["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"


def test_kernels_attn_beats_extra_and_kv_attention(prepared_vllm, monkeypatch):
    forced = _with("attn=TRITON_ATTN", kv_dtype="fp8_e5m2", extra={"attention_backend": "FLEX_ATTENTION"})
    args = _launch(forced, prepared_vllm, monkeypatch).args
    assert args.count("--attention-backend") == 1 and args[args.index("--attention-backend") + 1] == "TRITON_ATTN"
    without = CFG.model_copy(update={"extra": {"attention_backend": "FLEX_ATTENTION"}})
    args = _launch(without, prepared_vllm, monkeypatch).args
    assert args[args.index("--attention-backend") + 1] == "FLEX_ATTENTION"  # as before, with no kernel setting


@pytest.mark.parametrize("kern, facts, reason", [
    ("attn=TURBOQUANT", vllm_facts(), "this vLLM has no attention backend TURBOQUANT"),
    ("cg=BREAKABLE", vllm_facts(), "this vLLM has no CUDA-graph mode BREAKABLE"),
    ("mode=FAST", vllm_facts(), "this vLLM has no compilation mode FAST"),
    ("O=4", vllm_facts(), "this vLLM has no optimisation level 4"),
    ("pass.enable_sp=1", vllm_facts(), "this vLLM has no compilation pass enable_sp"),
    ("cgmax=64", vllm_facts(kernel={**VLLM_KERNEL, "compilation_fields": ["mode"]}),
     "this vLLM's compilation settings have no max_cudagraph_capture_size"),
    ("cg=FULL", vllm_facts(flags=("--attention-backend",)), "this vLLM has no --compilation-config"),
    ("O=1", vllm_facts(flags=("--attention-backend",)), "this vLLM has no --optimization-level"),
    ("attn=TRITON_ATTN", vllm_facts(flags=("--max-num-seqs",)), "this vLLM has no --attention-backend"),
    ("attnd=fa3", vllm_facts(), "vLLM has no kernel setting attnd"),
])
def test_unconfirmed_key_is_not_rendered_and_plan_says_why(kern, facts, reason, prepared_vllm, monkeypatch):
    monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
    backend = vl.VllmBackend()
    assert backend.config_refusal(_with(kern), prepared_vllm) == reason
    with pytest.raises(ValueError, match="ha(s|ve) no"):
        backend.launch_spec(_with(kern), prepared_vllm, 8000)


@pytest.mark.parametrize("facts", [None, vllm_facts(kernel=None), vllm_facts(flags=None)])
def test_probe_off_offers_nothing_with_a_note(facts, prepared_vllm, monkeypatch):
    """With no answer the setting is refused as not asked: never rendered on a guess, never called unsupported."""
    monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
    backend = vl.VllmBackend()
    reason = backend.config_refusal(_with("cg=FULL"), prepared_vllm)
    assert reason == kernels.NOT_ASKED and "not asked" in reason and "support" not in reason
    assert backend.config_refusal(CFG, prepared_vllm) is None  # and a config without kernels asks nothing


def test_rendered_arguments_match_the_golden(prepared_vllm, monkeypatch):
    base = _launch(CFG, prepared_vllm, monkeypatch)
    got = {}
    for text in STRINGS:
        spec = _launch(_with(text), prepared_vllm, monkeypatch)
        added = [a for a in spec.args if a not in base.args]
        changed = {k: v for k, v in spec.env.items() if base.env.get(k) != v}
        got[kernels.canonical(text)] = {"added_args": added, "env": changed}
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))
