"""Backend registry. New hardware = new class here, no core changes."""

from __future__ import annotations

from typing import Dict

from rigtune.backends.base import Backend, BaseBackend, LaunchSpec, LlmtraceHooks, Process, free_port


def registry() -> Dict[str, BaseBackend]:
    from rigtune.backends.llamacpp import LlamaCppCpuBackend, LlamaCppCudaBackend
    from rigtune.backends.sglang import SglangBackend
    from rigtune.backends.vllm import VllmBackend
    from rigtune.backends.vllm_cpu import VllmCpuBackend

    backends = [VllmBackend(), SglangBackend(), LlamaCppCudaBackend(), LlamaCppCpuBackend(), VllmCpuBackend()]
    return {b.name: b for b in backends}


def get_backend(name: str) -> BaseBackend:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(reg)}")
    return reg[name]


__all__ = [
    "Backend",
    "BaseBackend",
    "LaunchSpec",
    "LlmtraceHooks",
    "Process",
    "free_port",
    "registry",
    "get_backend",
]
