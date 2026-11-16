"""With no kernel setting, nothing is different: the same key, the same saved config, the same launch."""

from __future__ import annotations

import rigtune.backends.vllm as vl
from rigtune.models import Config
from tests.kernel_facts import vllm_facts

CONFIGS = [
    Config(backend="vllm", quant="bf16", ctx=4096, batch=32, gpu_memory_utilization=0.9),
    Config(backend="vllm", quant="fp8", ctx=8192, batch=64, gpu_memory_utilization=0.92, kv_dtype="fp8_e5m2",
           prefill_budget=8192, spec_decode="ngram_gpu:4"),
    Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9,
           extra={"attention_backend": "TRITON_ATTN"}),
]
KEYS = ["vllm/bf16/ctx4096/b32/gmu0.90",
        "vllm/fp8/ctx8192/b64/gmu0.92/pb8192/kvfp8_e5m2/sd=ngram_gpu:4",
        "vllm/bf16/ctx4096/b16/gmu0.90/attention_backend=TRITON_ATTN"]


def test_keys_and_dumps_are_what_they_were():
    assert [c.key() for c in CONFIGS] == KEYS
    for cfg in CONFIGS:
        assert "kernels" not in cfg.model_dump(mode="json")
        assert Config.model_validate({**cfg.model_dump(mode="json"), "kernels": None}).key() == cfg.key()


def test_goldens_unmodified_launch_and_env(prepared_vllm, monkeypatch):
    """A launch without kernel settings is the same whatever the probe says about them, and never asks."""
    backend = vl.VllmBackend()
    specs = []
    for facts in (vllm_facts(), vllm_facts(kernel=None)):
        monkeypatch.setattr(vl, "_engine_facts", lambda facts=facts: facts)
        specs.append([backend.launch_spec(c, prepared_vllm, 8000) for c in CONFIGS])
    assert [(s.args, s.env) for s in specs[0]] == [(s.args, s.env) for s in specs[1]]
    for spec in specs[0]:
        assert "--compilation-config" not in spec.args and "--optimization-level" not in spec.args
