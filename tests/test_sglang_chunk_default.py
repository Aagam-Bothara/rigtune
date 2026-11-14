"""SGLang's sliding-window pool with the radix cache off, on 0.5.19 and 0.5.23.

Two rules, read from both release tags (cited in rigtune/backends/sglang.py):
- 0.5.23 gives cards of 20 to 35 GiB a 4096-token prefill chunk where 0.5.19 gave 2048, and the pool holds two.
- With speculation, each request holds the eviction interval once per draft token, and two decode steps of draft
  tokens, where rigtune planned one interval and one token (both releases).
Expected numbers are written out by hand. Everything else about the plan must not move: dense models, the radix
cache, linear-attention layers, the A40, and every release before 0.5.23 without speculation.
"""

from __future__ import annotations

import pytest

from rigtune.backends.sglang import (
    SglangBackend,
    sglang_decode_alloc,
    sglang_default_chunk,
    sglang_hybrid_kv,
    sglang_release,
)
from rigtune.memory import kv_cache_bytes
from tests.conftest import make_hw
from tests.test_hybrid_memory import cfg, gemma4, qwen25, qwen35

GIB_MIB = 1024
SWA_BYTES = 327_680  # Gemma 4 12B: 40 sliding layers x 8 KV heads x 256 x 2 (K, V) x 2 B, per token
FULL_PART = 268_435_456  # its full-attention pool at ctx 4096 x 16 (test_hybrid_memory)


def hw(card: str, version):
    h = make_hw(card)
    h.backends["sglang"] = h.backends["sglang"].model_copy(update={"version": version})
    return h


@pytest.mark.parametrize("mib, version, chunk", [
    (16 * GIB_MIB, "0.5.23", 2048),  # under 20 GiB: 2048 at both
    (24564, "0.5.19", 2048),  # an RTX 4090 as nvidia-smi reports it
    (24564, "0.5.23", 4096),
    (23034, "0.5.23", 4096),  # an L4
    (24564, "0.5.21", 2048),  # 0.5.20 to 0.5.22 were not read: 0.5.19's rule
    (24564, "0.5.24", 4096),
    (24564, "0.6.0", 4096),
    (24564, "0.5.23.post1", 4096),
    (24564, None, 2048),  # unknown: the rule rigtune always had
    (24564, "1.0", 2048),  # not major.minor.patch
    (46068, "0.5.19", 4096),  # an A40: 4096 at both
    (46068, "0.5.23", 4096),
    (80 * GIB_MIB, "0.5.23", 8192),
    (141 * GIB_MIB, "0.5.19", 8192),
    (192 * GIB_MIB, "0.5.23", 16384),
    (0, "0.5.23", 4096),  # no GPU memory reported
])
def test_default_chunk_by_card_and_release(mib, version, chunk):
    assert sglang_default_chunk(mib * 2**20, version) == chunk


def test_release_parsing():
    assert sglang_release("0.5.10.post1") == (0, 5, 10)
    assert sglang_release("0.5.23") == (0, 5, 23)
    assert sglang_release("1.0") is None and sglang_release(None) is None


def _kv(model, c, h) -> int:
    mm = SglangBackend().memory_model(h)
    return kv_cache_bytes(model, c, mm.kv_tokens_fn(c), mm.hybrid_for(c, model.arch))


def test_gemma4_without_radix_cache_on_a_24gb_card_plans_two_bigger_chunks_on_0523():
    model, c = gemma4("sglang"), cfg("sglang", ctx=4096, batch=16, prefix_cache=False)
    # (1024 window + 128 eviction interval + 2 pages) x 16 + 2 chunks + 1 page.
    assert _kv(model, c, hw("rtx4090", "0.5.19")) == FULL_PART + SWA_BYTES * (1154 * 16 + 2 * 2048 + 1)
    assert _kv(model, c, hw("rtx4090", "0.5.23")) == FULL_PART + SWA_BYTES * (1154 * 16 + 2 * 4096 + 1)
    # An explicit --chunked-prefill-size replaces either default.
    c8k = c.model_copy(update={"prefill_budget": 8192})
    assert _kv(model, c8k, hw("rtx4090", "0.5.19")) == _kv(model, c8k, hw("rtx4090", "0.5.23")) \
        == FULL_PART + SWA_BYTES * (1154 * 16 + 2 * 8192 + 1)


def test_the_radix_cache_dense_and_linear_layers_ignore_the_release():
    old, new = hw("rtx4090", "0.5.19"), hw("rtx4090", "0.5.23")
    for model, c in [(gemma4("sglang"), cfg("sglang", ctx=4096, batch=16)),  # radix on: 0.8 x the full pool
                     (qwen35("sglang"), cfg("sglang", ctx=4096, batch=16, prefix_cache=False)),
                     (qwen35("sglang"), cfg("sglang", ctx=4096, batch=16)),
                     (qwen25("sglang"), cfg("sglang", ctx=4096, batch=16, prefix_cache=False))]:
        assert _kv(model, c, old) == _kv(model, c, new)


@pytest.mark.parametrize("spec, decode", [(None, 1), ("ngram:12", 24), ("eagle3:some/head:3", 8),
                                          ("draft:some/draft:3", 8), ("ngram:4", 20)])
def test_decode_allocation(spec, decode):
    # n-gram: top-k 10, steps max(1, draft // 10): 2 x max(10, 12) = 24, and for 4 draft tokens 2 x max(10, 4).
    assert sglang_decode_alloc(spec) == decode


@pytest.mark.parametrize("version", ["0.5.19", "0.5.23"])
def test_speculation_holds_an_eviction_interval_per_draft_token(version):
    model, h = gemma4("sglang"), hw("rtx4090", version)
    chunk = sglang_default_chunk(h.gpu.vram_total_bytes, version)
    base = cfg("sglang", ctx=4096, batch=16, prefix_cache=False)
    # n-gram, 12 draft tokens: 1024 + 128 x 12 + 1 page + 24 = 2585 tokens a request (was 1154).
    c = base.model_copy(update={"spec_decode": "ngram:12"})
    assert _kv(model, c, h) == FULL_PART + SWA_BYTES * (2585 * 16 + 2 * chunk + 1)
    # A chain of 3 (EAGLE3 head or draft model), 4 draft tokens: 1024 + 512 + 1 + 8 = 1545.
    for spec in ("eagle3:some/head:3", "draft:some/draft:3"):
        c = base.model_copy(update={"spec_decode": spec})
        assert _kv(model, c, h) == FULL_PART + SWA_BYTES * (1545 * 16 + 2 * chunk + 1)
    # With the radix cache the pool is a share of the full one, whatever the speculation.
    on = cfg("sglang", ctx=4096, batch=16)
    assert _kv(model, on.model_copy(update={"spec_decode": "ngram:12"}), h) == _kv(model, on, h)


def _old_swa_pool(c, arch, vram) -> int:
    """The pool rigtune planned before these rules (any release): (window + 130) x requests + 2 chunks + 1."""
    chunk = c.prefill_budget or sglang_default_chunk(vram, "0.5.19")
    return (arch.sliding_window + 130) * c.batch + 2 * chunk + 1


@pytest.mark.parametrize("card", ["a100", "rtx4090"])
def test_only_speculation_and_0523s_chunk_move_the_pool(card):
    """Over a grid of configurations: on 0.5.19 the pool is the old one unless a speculation setting is on; on
    0.5.23 it is 0.5.19's unless the card is in the 20-35 GiB tier with no explicit chunk."""
    model = gemma4("sglang")
    vram = make_hw(card).gpu.vram_total_bytes
    grid = [cfg("sglang", ctx=ctx, batch=b, prefix_cache=False, prefill_budget=pb, spec_decode=s)
            for ctx in (2048, 4096, 32768) for b in (16, 64, 256) for pb in (None, 2048, 8192)
            for s in (None, "ngram:12", "draft:x/y:3")]
    for c in grid:
        old = sglang_hybrid_kv(c, model.arch, vram, "0.5.19").swa_pool_tokens
        new = sglang_hybrid_kv(c, model.arch, vram, "0.5.23").swa_pool_tokens
        assert (old == _old_swa_pool(c, model.arch, vram)) is (c.spec_decode is None)
        moved = card == "rtx4090" and c.prefill_budget is None
        assert new - old == (2 * (4096 - 2048) if moved else 0)
