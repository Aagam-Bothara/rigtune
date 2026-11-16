"""SGLang's CUDA-graph memory is counted when its decode cap is raised above the card's default.

SGLang reserves 2 MB per unit of decode cap itself only when it works out the memory fraction; rigtune always passes
the fraction, so the plan counts what a raised cap adds. Unmeasured: the figure is SGLang's own reserve, read in its
source, and the doubling with a second model is a prior."""

from __future__ import annotations

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from rigtune import kernels
from rigtune.memory import estimate
from rigtune.models import GiB, MiB, Config
from tests.conftest import make_hw

A40_MIB = 46068.0
BASE = Config(backend="sglang", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.88)


def _version(hw, name: str, version: str):
    hw.backends[name].version = version
    return hw


def _bytes(kern: str, batch: int = 64, version: str = "0.5.23", second: bool = False, vram: float = A40_MIB) -> int:
    return kernels.sglang_graph_extra_bytes(kernels.parse(kern), vram, batch, 1, version, second_model=second)


def test_no_term_when_cgmax_unset(hw_a100, prepared_vllm):
    assert _bytes("cg=full") == 0 and _bytes("attn=triton,tc=1") == 0
    hw = _version(hw_a100, "sglang", "0.5.23")
    assert sg.sglang_graph_bytes(BASE, hw.gpu.vram_total_bytes, "0.5.23") == 0
    mm = sg.SglangBackend().memory_model(hw)
    plain = estimate(hw, prepared_vllm, BASE, mm)
    mm.extra_fn = None
    assert estimate(hw, prepared_vllm, BASE, mm) == plain  # the estimate is what it was without the term


def test_no_term_at_or_below_default_cap():
    assert _bytes("cgmax=32") == 0 and _bytes("cgmax=16") == 0  # an A40's default is 32
    assert _bytes("cgmax=16", batch=16) == 0  # 0.5.23 lowers the default to the batch


def test_two_mib_per_unit_above_default():
    assert _bytes("cgmax=64") == (64 - 32) * 2 * MiB
    assert _bytes("cgmax=128") == (128 - 32) * 2 * MiB
    assert _bytes("cgmax=64", batch=16) == (64 - 16) * 2 * MiB  # above the clamped default
    assert _bytes("cgmax=64", batch=16, version="0.5.19") == (64 - 32) * 2 * MiB  # 0.5.19 does not clamp


def test_doubled_with_second_model():
    assert _bytes("cgmax=64", second=True) == 2 * _bytes("cgmax=64")
    drafted = BASE.model_copy(update={"kernels": "cgmax=64", "spec_decode": "draft:org/small:4"})
    lookup = BASE.model_copy(update={"kernels": "cgmax=64", "spec_decode": "ngram:4"})
    total = int(A40_MIB * MiB)
    assert sg.sglang_graph_bytes(drafted, total, "0.5.23") == 2 * sg.sglang_graph_bytes(lookup, total, "0.5.23")


@pytest.mark.parametrize("version", [None, "", "0.6.0", "main"])
def test_unknown_release_counts_the_whole_cap(version):
    """Where the default is not known the plan is the careful one: every unit of the cap is counted."""
    assert _bytes("cgmax=64", version=version) == 64 * 2 * MiB


def test_vllm_gets_no_term(hw_a100, prepared_vllm):
    """vLLM profiles its graphs inside the memory fraction: its plan has no such term, with or without a setting."""
    mm = vl.VllmBackend().memory_model(hw_a100)
    assert mm.extra_fn is None
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    raised = cfg.model_copy(update={"kernels": "cgmax=128"})
    a, b = estimate(hw_a100, prepared_vllm, cfg, mm), estimate(hw_a100, prepared_vllm, raised, mm)
    assert (a.total, a.runtime_workspace, a.feasible) == (b.total, b.runtime_workspace, b.feasible)


def test_raised_cap_lowers_the_fitting_fraction(prepared_vllm):
    """The term reaches the plan: a raised cap takes exactly its bytes more, and can stop a tight config fitting."""
    hw = _version(make_hw("a100"), "sglang", "0.5.23")  # 80 GiB: default cap 256
    mm = sg.SglangBackend().memory_model(hw)
    wide = BASE.model_copy(update={"batch": 512})
    raised = wide.model_copy(update={"kernels": "cgmax=512"})
    plain, more = estimate(hw, prepared_vllm, wide, mm), estimate(hw, prepared_vllm, raised, mm)
    assert more.total - plain.total == (512 - 256) * 2 * MiB == more.runtime_workspace - plain.runtime_workspace
    assert (more.weights, more.kv_cache, more.budget) == (plain.weights, plain.kv_cache, plain.budget)
    tight = plain.total + 100 * MiB  # room for the plain config and not for 512 MiB of graphs
    assert estimate(hw, prepared_vllm, wide, mm, available=int(tight / 0.9) + GiB).feasible
    room = next(a for a in range(plain.total, plain.total + 4 * GiB, 16 * MiB)
                if estimate(hw, prepared_vllm, wide, mm, available=a).feasible)
    assert not estimate(hw, prepared_vllm, raised, mm, available=room).feasible
