"""Config.kernels rendered for SGLang: by the flags' current names, and only what its own parser reported.

With kernels unset the launch is what it was and the probe's kernel flags are never consulted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rigtune.backends.sglang as sg
from rigtune import kernels
from rigtune.backends.facts import EngineFacts
from rigtune.models import Config
from tests.kernel_facts import sglang_facts

GOLDEN = Path(__file__).parent / "golden" / "kernels" / "sglang-args.json"
CFG = Config(backend="sglang", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.88)
STRINGS = ["cgmax=64", "cg=breakable", "cgp=disabled,cgpmax=8192", "attn=triton", "attnp=fa3,attnd=flashinfer",
           "tc=1,tcmax=16", "tc=0", "cg=full,cgmax=128,attn=flashinfer"]


def _with(kern: str) -> Config:
    return CFG.model_copy(update={"kernels": kernels.canonical(kern)})


def _backend(monkeypatch, facts) -> sg.SglangBackend:
    monkeypatch.setattr(sg, "_engine_facts", lambda: facts)
    return sg.SglangBackend()


def _added(kern: str, prepared, monkeypatch, facts=None) -> list:
    backend = _backend(monkeypatch, sglang_facts() if facts is None else facts)
    base = backend.launch_spec(CFG, prepared, 8000).args
    args = backend.launch_spec(_with(kern), prepared, 8000).args
    assert args[:len(base)] == base  # the settings are added after everything the launch already had
    return args[len(base):]


def test_cgmax_reaches_the_command_line(prepared_vllm, monkeypatch):
    assert _added("cgmax=64", prepared_vllm, monkeypatch) == ["--cuda-graph-max-bs-decode", "64"]


def test_renders_decode_cap_by_non_deprecated_name(prepared_vllm, monkeypatch):
    for version in ("0.5.19", "0.5.23"):
        added = _added("cgmax=64,cgpmax=4096", prepared_vllm, monkeypatch, sglang_facts(version))
        assert added == ["--cuda-graph-max-bs-decode", "64", "--cuda-graph-max-bs-prefill", "4096"]


def test_never_renders_cuda_graph_max_bs_alias(prepared_vllm, monkeypatch):
    """0.5.19 still takes --cuda-graph-max-bs; 0.5.23 does not. Neither launch is given it."""
    for text in STRINGS:
        assert "--cuda-graph-max-bs" not in _added(text, prepared_vllm, monkeypatch, sglang_facts("0.5.19"))


def test_torch_compile_off_adds_nothing(prepared_vllm, monkeypatch):
    assert _added("tc=0", prepared_vllm, monkeypatch) == []
    assert _added("tc=1,tcmax=16", prepared_vllm, monkeypatch) == ["--enable-torch-compile",
                                                                   "--torch-compile-max-bs", "16"]
    reason = _backend(monkeypatch, sglang_facts()).config_refusal(_with("tc=0,tcmax=16"), prepared_vllm)
    assert reason == "tcmax sets a size for torch.compile, which tc=0 leaves off"


def test_choice_missing_from_probe_is_refused(prepared_vllm, monkeypatch):
    old, new = _backend(monkeypatch, sglang_facts("0.5.19")), None
    assert old.config_refusal(_with("cgp=tc_piecewise"), prepared_vllm) is None  # 0.5.19 has it
    new = _backend(monkeypatch, sglang_facts("0.5.23"))
    reason = new.config_refusal(_with("cgp=tc_piecewise"), prepared_vllm)
    assert reason == ("this SGLang's --cuda-graph-backend-prefill does not take tc_piecewise "
                      "(it takes full, breakable, disabled)")
    with pytest.raises(ValueError, match="does not take tc_piecewise"):
        new.launch_spec(_with("cgp=tc_piecewise"), prepared_vllm, 8000)


@pytest.mark.parametrize("kern, facts, reason", [
    ("cgmax=64", sglang_facts(drop=("--cuda-graph-max-bs-decode",)), "this SGLang has no --cuda-graph-max-bs-decode"),
    ("tc=1", sglang_facts(drop=("--enable-torch-compile",)), "this SGLang has no --enable-torch-compile"),
    ("O=1", sglang_facts(), "SGLang has no kernel setting O"),
    ("pass.fuse_norm_quant=1", sglang_facts(), "SGLang has no kernel setting pass.fuse_norm_quant"),
])
def test_flag_missing_from_probe_is_refused(kern, facts, reason, prepared_vllm, monkeypatch):
    assert _backend(monkeypatch, facts).config_refusal(_with(kern), prepared_vllm) == reason


@pytest.mark.parametrize("facts", [None, EngineFacts(engine="sglang", version="0.5.23")])
def test_flags_none_offers_nothing(facts, prepared_vllm, monkeypatch):
    backend = _backend(monkeypatch, facts)
    reason = backend.config_refusal(_with("cgmax=64"), prepared_vllm)
    assert reason == kernels.NOT_ASKED and "support" not in reason
    with pytest.raises(ValueError, match="not asked"):
        backend.launch_spec(_with("cgmax=64"), prepared_vllm, 8000)
    assert backend.config_refusal(CFG, prepared_vllm) is None


def test_sglang_goldens_unmodified(prepared_vllm, monkeypatch):
    """Without kernels the launch is the same whatever the probe reports, down to no answer at all."""
    launches = [_backend(monkeypatch, facts).launch_spec(CFG, prepared_vllm, 8000)
                for facts in (sglang_facts(), sglang_facts("0.5.19"), None)]
    assert all((s.args, s.env) == (launches[0].args, launches[0].env) for s in launches)
    assert not any(a.startswith(("--cuda-graph", "--attention-backend", "--enable-torch")) for a in launches[0].args)
    assert "kern" not in CFG.key()


def test_rendered_arguments_match_the_golden(prepared_vllm, monkeypatch):
    got = {kernels.canonical(text): _added(text, prepared_vllm, monkeypatch) for text in STRINGS}
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))
