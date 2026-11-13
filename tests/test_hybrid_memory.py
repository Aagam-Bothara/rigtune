"""KV planning per layer type for hybrid-attention models.

Expected numbers are written out by hand from the engines' rules (cited in rigtune/backends/vllm.py and
sglang.py), not computed with the code under test.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from rigtune.backends.llamacpp import LlamaCppCudaBackend
from rigtune.backends.sglang import SglangBackend, sglang_default_chunk
from rigtune.backends.vllm import VllmBackend, vllm_default_batched_tokens, vllm_hybrid_kv
from rigtune.hfconfig import arch_from_config
from rigtune.memlog import parse_sglang, parse_vllm
from rigtune.memory import MemoryModel, estimate, kv_cache_bytes, plan
from rigtune.models import Config, GiB, ModelSpec, PreparedModel
from tests.conftest import LLAMA_3B_CONFIG, make_hw

FIXTURES = Path(__file__).parent / "fixtures"
STATE = 51_511_296  # Qwen3.5-9B recurrent state per sequence (tests/test_hybrid_layout.py)
# vLLM pads each layer's state to one attention page: 528-token blocks x 4 heads x 256 x 2 (K, V) x 2 B.
STATE_VLLM = 24 * 528 * 4096  # 51,904,512 B


def prepared(fixture: str, backend: str, hf_id: str, weights=None) -> PreparedModel:
    arch = arch_from_config(json.loads((FIXTURES / fixture).read_text(encoding="utf-8")))
    return PreparedModel(spec=ModelSpec(hf_id=hf_id), backend=backend, arch=arch, hf_path=hf_id,
                         weights_bytes=weights or {"bf16": 8 * GiB, "fp8": 4 * GiB})


def qwen35(backend="vllm", weights=None):
    return prepared("qwen3_5_9b_config.json", backend, "Qwen/Qwen3.5-9B", weights)


def gemma4(backend="vllm", weights=None):
    return prepared("gemma4_12b_it_config.json", backend, "google/gemma-4-12B-it", weights)


def qwen25(backend="vllm"):
    return prepared("qwen2_5_7b_instruct_config.json", backend, "Qwen/Qwen2.5-7B-Instruct")


def gemma3(backend: str, layers: int = 48, kv_heads: int = 8, head_dim: int = 256) -> PreparedModel:
    """The fields google/gemma-3-12b-it (48 layers) and -27b (62 layers, 16 heads of 128) carry: no layer_types."""
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"],
           "text_config": {"num_hidden_layers": layers, "num_key_value_heads": kv_heads, "head_dim": head_dim,
                           "hidden_size": 3840, "num_attention_heads": 16, "sliding_window": 1024,
                           "sliding_window_pattern": 6}}
    return PreparedModel(spec=ModelSpec(hf_id="google/gemma-3"), backend=backend, arch=arch_from_config(cfg),
                         hf_path="x", weights_bytes={"bf16": 8 * GiB})


def cfg(backend="vllm", ctx=4096, batch=64, **kw) -> Config:
    return Config(backend=backend, quant="bf16", ctx=ctx, batch=batch, gpu_memory_utilization=0.9, **kw)


def kv(model, c, mm) -> int:
    return kv_cache_bytes(model, c, mm.kv_tokens_fn(c), mm.hybrid_for(c, model.arch))


# ---------------------------------------------------------------- dense control: not one byte moves

QWEN25_CONFIG = {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 3584, "intermediate_size": 18944,
                 "max_position_embeddings": 32768, "num_attention_heads": 28, "num_hidden_layers": 28,
                 "num_key_value_heads": 4, "sliding_window": 131072, "use_sliding_window": False,
                 "tie_word_embeddings": False, "torch_dtype": "bfloat16", "vocab_size": 152064}
DENSE = {"qwen2.5-7b": (QWEN25_CONFIG, 7_615_616_512), "llama-3.2-3b": (LLAMA_3B_CONFIG, 3_212_749_824)}


def dense_rows(hwname: str, be, mname: str) -> list:
    """Every grid config's estimate, as the golden in tests/fixtures/dense_plan_golden.json was captured from
    rigtune 24b4e5e, before hybrid layouts existed (the same procedure, run on that tree)."""
    hw = make_hw(hwname)
    mm = be.memory_model(hw)
    config, params = DENSE[mname]
    weights = {"bf16": 2 * params, "fp16": 2 * params, "fp8": params, "awq": params // 2, "gptq": params // 2}
    model = PreparedModel(spec=ModelSpec(hf_id=mname), backend=be.name, arch=arch_from_config(config),
                          hf_path=mname, weights_bytes=weights)
    grid = {c.key(): c for m in (0, 16384, 32768) for c in be.candidate_configs(hw, model, min_ctx=m)}
    for c in list(grid.values()):
        for v in [c.model_copy(update={"kv_dtype": d}) for d in ("fp8", "fp8_e5m2")] \
                + [c.model_copy(update={"prefix_cache": False})] + be.prefill_variants(c):
            grid[v.key()] = v
    rows = []
    for k in sorted(grid):
        e = estimate(hw, model, grid[k], mm)
        rows.append([k, e.weights, e.kv_cache, e.runtime_workspace, e.safety_margin, e.total, e.budget, e.feasible])
    return rows


@pytest.mark.parametrize("group", sorted(json.loads((FIXTURES / "dense_plan_golden.json").read_text())["groups"]))
def test_dense_plans_are_byte_identical_to_before_hybrid_layouts(group, tmp_home):
    hwname, backend, mname = group.split("/")
    be = {"vllm": VllmBackend, "sglang": SglangBackend}[backend]()
    rows = dense_rows(hwname, be, mname)
    golden = json.loads((FIXTURES / "dense_plan_golden.json").read_text())["groups"][group]
    assert (len(rows), sum(r[-1] for r in rows)) == (golden["rows"], golden["feasible"])
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == golden["sha256"]


def test_dense_control_kv_matches_the_old_formula(hw_rtx4090):
    model, c = qwen25(), cfg()
    # 2 x 28 layers x 4 KV heads x 128 x 2 B = 57,344 B per token; 0.25 x 4096 x 64 = 65,536 tokens.
    assert estimate(hw_rtx4090, model, c, VllmBackend().memory_model(hw_rtx4090)).kv_cache == 57_344 * 65_536


# ---------------------------------------------------------------- vLLM


def test_vllm_qwen3_5_short_context_large_batch(hw_rtx4090):
    model, c = qwen35(), cfg(ctx=4096, batch=64)
    # 1,024 tokens a sequence round up to two 528-token blocks: 1,056 x 64 x 32,768 B (8 full layers, bf16).
    # State: 64 x 24 layers x 528 x 4,096 B (padded to an attention page).
    assert kv(model, c, VllmBackend().memory_model(hw_rtx4090)) == 1056 * 64 * 32_768 + 64 * STATE_VLLM
    assert kv_cache_bytes(model, c, 65_536) == 8 * GiB  # today's every-layer-full estimate


def test_vllm_qwen3_5_fp8_cache_doubles_the_block(hw_rtx4090):
    model, c = qwen35(), cfg(ctx=4096, batch=64, kv_dtype="fp8")
    # Attention page per token halves (2,048 B), so the block is 16 x cdiv(2,146,304, 16 x 2,048) = 1,056 tokens;
    # the state page stays 1,056 x 2,048 = 2,162,688 B.
    hk = vllm_hybrid_kv(c, model.arch)
    assert (hk.full_block, hk.state_bytes) == (1056, 24 * 2_162_688)
    assert kv(model, c, VllmBackend().memory_model(hw_rtx4090)) == 1056 * 64 * 16_384 + 64 * STATE_VLLM


def test_vllm_speculation_holds_a_state_per_token_and_widens_the_conv_state(hw_rtx4090):
    model, mm = qwen35(), VllmBackend().memory_model(hw_rtx4090)
    # ngram_gpu:4: a layer's state is 2,146,304 + 4 x 16,384 = 2,211,840 B, so the block is 16 x 34 = 544 tokens
    # and the padded state page 544 x 4,096 = 2,228,224 B; 5 states a sequence.
    c = cfg(ctx=4096, batch=16, spec_decode="ngram_gpu:4")
    assert kv(model, c, mm) == 1088 * 16 * 32_768 + 16 * 5 * 24 * 2_228_224
    # suffix:24: 2,539,520 B a layer, block 16 x 39 = 624, page 2,555,904 B, 25 states: ~1.5 GB a sequence.
    c = cfg(ctx=4096, batch=16, spec_decode="suffix:24")
    assert vllm_hybrid_kv(c, model.arch).full_block == 624
    assert kv(model, c, mm) == 1248 * 16 * 32_768 + 16 * 25 * 24 * 2_555_904


def test_vllm_gemma4_sliding_layers_cap_at_the_window_plus_in_flight_tokens(hw_rtx4090):
    mm = VllmBackend().memory_model(hw_rtx4090)
    model = gemma4()
    assert vllm_default_batched_tokens(hw_rtx4090) == 2048
    # ctx 4096 x 64: 1,024 tokens a sequence (= the window). Full layers: 16,384 B/token in 64-token blocks.
    # Sliding: 327,680 B/token x (1,024 x 64 + 2 x 2,048 in-flight tokens).
    assert kv(model, cfg(ctx=4096, batch=64), mm) == 16_384 * 65_536 + 327_680 * (65_536 + 4096)
    # ctx 32768 x 16: 8,192 a sequence; the window spans 1,024 + one 16-token block.
    assert kv(model, cfg(ctx=32768, batch=16), mm) == 16_384 * 131_072 + 327_680 * (1040 * 16 + 4096)
    assert kv_cache_bytes(model, cfg(ctx=32768, batch=16), 131_072) == 48 * GiB  # today's estimate
    # Short context, large prefill budget: 512 tokens a sequence + 2 x 16,384 in flight would be 40,960 sliding
    # tokens, but no sequence holds more than min(1,023 + 32,768, 2,048) rounded up + 1 block = 2,064.
    c = cfg(ctx=2048, batch=16, prefill_budget=16384)
    assert kv(model, c, mm) == 16_384 * 512 * 16 + 327_680 * 2064 * 16


def test_vllm_refuses_what_vllm_refuses_at_start_up(hw_rtx4090):
    """One request at max_model_len must fit (kv_cache_utils._check_enough_kv_cache_memory)."""
    mm = VllmBackend().memory_model(hw_rtx4090)
    model = gemma4(weights={"fp8": 11_959_730_224})
    case = Config(backend="vllm", quant="fp8", ctx=32768, batch=16, gpu_memory_utilization=0.9, prefill_budget=16384)
    # Per KV cache group (8 layers), in blocks of 1 MiB: full cdiv(32,768, 64) = 512; each of the 5 sliding
    # groups cdiv(min(1,023 + 2 x 16,384, 32,768), 16) + 1 = 2,049; plus the null block: 10,758 MiB.
    assert vllm_hybrid_kv(case, model.arch).min_bytes == (512 + 5 * 2049 + 1) * 2**20
    # The previous hybrid rules, with no in-flight term or start-up check, planned 7 GiB here and admitted it.
    e = estimate(hw_rtx4090, model, case, mm)
    assert not e.feasible and e.kv_cache >= (512 + 5 * 2049 + 1) * 2**20
    # At batch 1 the start-up requirement is what binds: with the default 2 x 2,048 in-flight tokens each sliding
    # group needs cdiv(1,023 + 4,096, 16) + 1 = 321 blocks, so 512 + 5 x 321 + 1 blocks of 1 MiB.
    one = case.model_copy(update={"batch": 1, "prefill_budget": None})
    assert kv(model, one, mm) == (512 + 5 * 321 + 1) * 2**20


def test_vllm_pads_sliding_layers_to_whole_groups():
    # gemma-3-27b: 62 layers, 10 full and 52 sliding; groups of 10 layers, so the sliding layers take 60 pages.
    hk = vllm_hybrid_kv(cfg(), gemma3("vllm", layers=62, kv_heads=16, head_dim=128).arch)
    assert (hk.full_layers, hk.sliding_layers) == (10, 60)


# ---------------------------------------------------------------- SGLang


def test_sglang_qwen3_5_radix_cache_holds_five_slots_a_request(hw_rtx4090):
    model, mm = qwen35("sglang"), SglangBackend().memory_model(hw_rtx4090)
    # Extra buffer under the overlap scheduler: 3 + 2 slots a request, + 1 padding slot: 81 x 51,511,296 B
    # = 4,172,414,976 B, inside the mamba share 0.9 / 1.9: x 19 / 9 = 8,808,431,616 B.
    assert abs(kv(model, cfg("sglang", ctx=4096, batch=16), mm) - 8_808_431_616) <= 1
    # With the overlap scheduler off the strategy resolves to no_buffer: 49 slots, x 19 / 9 = 5,328,557,397.3 B.
    c = cfg("sglang", ctx=4096, batch=16, extra={"disable_overlap_schedule": True})
    assert abs(kv(model, c, mm) - 5_328_557_398) <= 1


def test_sglang_qwen3_5_without_radix_cache(hw_rtx4090):
    model, c = qwen35("sglang"), cfg("sglang", ctx=4096, batch=16, prefix_cache=False)
    # max_mamba_cache_size = max_running_requests (16) + 1 padding slot; full KV 32,768 x 16,384 B.
    assert kv(model, c, SglangBackend().memory_model(hw_rtx4090)) == 536_870_912 + 875_692_032


def test_sglang_speculation_reserves_intermediate_states(hw_rtx4090):
    model, mm = qwen35("sglang"), SglangBackend().memory_model(hw_rtx4090)
    # n-gram, 12 draft tokens, radix cache off: 17 slots x (1 + 12) states = 221 x 51,511,296 B.
    c = cfg("sglang", ctx=4096, batch=16, prefix_cache=False, spec_decode="ngram:12")
    assert kv(model, c, mm) == 536_870_912 + 11_383_996_416
    # Radix cache on, D = 12: 5 x 16 + 1 slots + 12 x 17 intermediate = 285 x 51,511,296 = 14,680,719,360 B,
    # x 19 / 9 = 30,992,629,760 B (K = 81 solves _handle_max_mamba_cache's joint budget exactly).
    c = cfg("sglang", ctx=4096, batch=16, spec_decode="ngram:12")
    assert abs(kv(model, c, mm) - 30_992_629_760) <= 1
    # EAGLE3, 3 steps, 4 draft tokens: 81 + 68 = 149 x 51,511,296 = 7,675,183,104 B, x 19 / 9.
    c = cfg("sglang", ctx=4096, batch=16, spec_decode="eagle3:some/head:3")
    assert abs(kv(model, c, mm) - 16_203_164_331) <= 1


def test_sglang_gemma4_radix_cache_swa_ratio(hw_rtx4090):
    model, mm = gemma4("sglang"), SglangBackend().memory_model(hw_rtx4090)
    # Full pool 131,072 tokens; SWA pool 0.8 of it whatever the window: 104,857 tokens.
    assert kv(model, cfg("sglang", ctx=32768, batch=16), mm) == 2_147_483_648 + 34_359_541_760
    # Short contexts: the SWA need (65,536) exceeds 0.8 x 65,536, so the full pool grows to 81,920 tokens.
    assert kv(model, cfg("sglang", ctx=4096, batch=64), mm) == 1_342_177_280 + 21_474_836_480


def test_sglang_gemma4_without_radix_cache_reserves_the_window_per_request(hw_rtx4090):
    model, mm = gemma4("sglang"), SglangBackend().memory_model(hw_rtx4090)
    # SWA pool: (1024 + 128 eviction interval + 2 pages) x 16 + 2 x 2048 chunk (24 GiB GPU) + 1 = 22,561 tokens.
    assert sglang_default_chunk(hw_rtx4090.gpu.vram_total_bytes) == 2048
    c = cfg("sglang", ctx=4096, batch=16, prefix_cache=False)
    assert kv(model, c, mm) == 268_435_456 + 7_392_788_480
    c = c.model_copy(update={"prefill_budget": 8192})  # an explicit --chunked-prefill-size replaces the default
    assert kv(model, c, mm) == 268_435_456 + 327_680 * 34_849


def test_sglang_keeps_every_gemma3_layer_full(hw_rtx4090, caplog):
    """SGLang disables its hybrid SWA pool for Gemma 2 and 3 (model_overrides/gemma2_gemma3.py)."""
    model, mm = gemma3("sglang"), SglangBackend().memory_model(hw_rtx4090)
    assert model.arch.is_hybrid and mm.hybrid_for(cfg("sglang"), model.arch) is None
    # 2 x 48 layers x 8 heads x 256 x 2 B = 393,216 B/token x 65,536 tokens, as before.
    assert kv(model, cfg("sglang", ctx=4096, batch=64), mm) == 24 * GiB
    with caplog.at_level(logging.INFO, logger="rigtune.memory"):
        plan(hw_rtx4090, model, [cfg("sglang")], mm)
    assert "full KV cache on every layer of Gemma3ForConditionalGeneration" in caplog.text
    # vLLM does free Gemma 3's sliding layers: at 32k x 16, 8 full layers (65,536 B/token) x 131,072 tokens
    # and 40 sliding (327,680 B/token) x (1,040 x 16 + 4,096 in-flight), against 48 GiB every-layer-full.
    vllm_kv = kv(gemma3("vllm"), cfg(ctx=32768, batch=16), VllmBackend().memory_model(hw_rtx4090))
    assert vllm_kv == 65_536 * 131_072 + 327_680 * (1040 * 16 + 4096)


# ---------------------------------------------------------------- engines without verified rules


def test_backends_without_hybrid_rules_keep_todays_estimate(hw_rtx4090, caplog):
    be = LlamaCppCudaBackend()
    mm = be.memory_model(hw_rtx4090)
    model = qwen35("llamacpp-cuda")
    model.weights_bytes = {"Q4_K_M": 5 * GiB}
    c = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=33)
    assert kv(model, c, mm) == 131_072 * 4096 * 4  # every layer full, as before
    with caplog.at_level(logging.INFO, logger="rigtune.memory"):
        plan(hw_rtx4090, model, [c], mm)
    assert "hybrid KV rules are not verified" in caplog.text


def test_speculation_leaves_dense_and_sliding_only_models_alone(hw_rtx4090):
    mm = SglangBackend().memory_model(hw_rtx4090)
    for model in (qwen25("sglang"), gemma4("sglang")):
        plain = cfg("sglang", ctx=4096, batch=16)
        assert kv(model, plain.model_copy(update={"spec_decode": "ngram:12"}), mm) == kv(model, plain, mm)


def test_hybrid_plan_admits_long_context_and_refuses_large_batch(hw_rtx4090):
    """The two directions today's planner was wrong in, on a 24 GB card with 4 GiB of fp8 weights."""
    mm = VllmBackend().memory_model(hw_rtx4090)
    model = qwen35()
    long_ctx = Config(backend="vllm", quant="fp8", ctx=65536, batch=16, gpu_memory_utilization=0.95)
    big_batch = Config(backend="vllm", quant="fp8", ctx=512, batch=512, gpu_memory_utilization=0.95)
    kept = {c.key() for c, _ in plan(hw_rtx4090, model, [long_ctx, big_batch], mm)}
    # Long context: 16,384 tokens a sequence round up to 16,896; x 16 x 32 KiB = 8.25 GiB, + 16 states.
    assert long_ctx.key() in kept
    assert estimate(hw_rtx4090, model, long_ctx, mm).kv_cache == 16_896 * 16 * 32_768 + 16 * STATE_VLLM
    assert kv_cache_bytes(model, long_ctx, mm.kv_tokens_fn(long_ctx)) == 32 * GiB
    # 512 short sequences: 512 x 49.5 MiB of state is 24.8 GiB, though today's estimate was only 8 GiB.
    assert big_batch.key() not in kept
    assert kv_cache_bytes(model, big_batch, mm.kv_tokens_fn(big_batch)) == 8 * GiB
    old = MemoryModel(mm.runtime_workspace, mm.kv_tokens_fn, mm.device, mm.margin_fraction, mm.calibrated)
    assert {c.key() for c, _ in plan(hw_rtx4090, model, [long_ctx, big_batch], old)} == {big_batch.key()}


# ---------------------------------------------------------------- measured memory from engine logs


SGLANG_GEMMA4_LOG = """
Load weight end. type=Gemma4UnifiedForConditionalGeneration, dtype=torch.bfloat16, avail mem=14.10 GB, mem usage=8.12 GB.
Use sliding window memory pool. full_layer_tokens=40960, swa_layer_tokens=32768
Full KV Cache is allocated. dtype: torch.bfloat16, #tokens: 40960, K size: 0.31 GB, V size: 0.31 GB
SWA KV Cache is allocated. dtype: torch.bfloat16, #tokens: 32768, K size: 5.00 GB, V size: 5.00 GB
"""

SGLANG_QWEN35_LOG = """
Load weight end. type=Qwen3_5ForConditionalGeneration, dtype=torch.bfloat16, avail mem=12.00 GB, mem usage=9.50 GB.
Mamba Cache is allocated. max_mamba_cache_size: 93, conv_state size: 0.10GB, ssm_state size: 4.36GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 150000, K size: 2.29 GB, V size: 2.29 GB
"""

# Two ranks of a dense model (--tp-size 2): each logs its own half-size pool.
SGLANG_TP2_LOG = """
[2026-11-12 10:00:01 TP0] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=19.20 GB, mem usage=7.10 GB.
[2026-11-12 10:00:01 TP1] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=19.30 GB, mem usage=7.10 GB.
[2026-11-12 10:00:02 TP1] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 300000, K size: 2.12 GB, V size: 2.12 GB
[2026-11-12 10:00:02 TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 300000, K size: 2.12 GB, V size: 2.12 GB
"""

# Post-capture KV sizing with speculation: the pool lines give a virtual upper bound, the backed size follows.
SGLANG_POST_CAPTURE_LOG = """
Load weight end. type=Qwen3_5ForConditionalGeneration, dtype=torch.bfloat16, avail mem=12.00 GB, mem usage=9.50 GB.
Mamba Cache is allocated. max_mamba_cache_size: 40, conv_state size: 0.05GB, ssm_state size: 2.00GB intermediate_ssm_state_cache size: 1.50GB intermediate_conv_window_cache size: 0.02GB
KV Cache VA upper bound. dtype: torch.bfloat16, #tokens: 400000, K size: 6.10 GB, V size: 6.10 GB
Post-capture KV sizing: full_layer_tokens=150000, swa_layer_tokens=0
Post-capture KV sizing: KV cache allocated. dtype: torch.bfloat16, #tokens: 150000, KV size: 4.58 GB, avail mem=1.10 GB
"""


def test_sglang_log_counts_every_hybrid_pool():
    m = parse_sglang(SGLANG_GEMMA4_LOG)
    assert m.kv_mb == pytest.approx((0.31 + 0.31 + 5.00 + 5.00) * 1024)
    assert m.kv_tokens == 40960 and m.details["swa_kv_tokens"] == 32768
    m = parse_sglang(SGLANG_QWEN35_LOG)
    assert m.kv_mb == pytest.approx((0.10 + 4.36 + 2.29 + 2.29) * 1024)
    assert m.details["mamba_mb"] == pytest.approx(4.46 * 1024) and m.details["mamba_slots"] == 93
    assert m.kv_tokens == 150000 and m.weights_mb == pytest.approx(9.50 * 1024) and m.source == "sglang-log"


def test_sglang_log_reads_one_rank_under_tensor_parallelism():
    m = parse_sglang(SGLANG_TP2_LOG)
    assert m.kv_mb == pytest.approx(4.24 * 1024)  # rank 0's pool only, not 8.48 GB
    assert m.weights_mb == pytest.approx(7.10 * 1024) and m.kv_tokens == 300000


def test_sglang_log_post_capture_and_speculative_state():
    m = parse_sglang(SGLANG_POST_CAPTURE_LOG)
    # Backed KV (4.58 GB, not the 12.2 GB upper bound) plus every state buffer, intermediates included.
    assert m.details["mamba_mb"] == pytest.approx((0.05 + 2.00 + 1.50 + 0.02) * 1024)
    assert m.kv_mb == pytest.approx((4.58 + 0.05 + 2.00 + 1.50 + 0.02) * 1024)
    assert m.kv_tokens == 150000


def test_sglang_dense_single_rank_log_reads_as_before():
    m = parse_sglang("Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.20 GB, "
                     "mem usage=6.21 GB.\nKV Cache is allocated. #tokens: 123456, K size: 2.12 GB, V size: 2.12 GB\n")
    assert m.kv_mb == pytest.approx(4.24 * 1024) and m.kv_tokens == 123_456 and m.details == {}


def test_vllm_log_kv_tokens_line_still_parses():
    m = parse_vllm("Available KV cache memory: 9.81 GiB\nGPU KV cache size: 74,976 tokens, Maximum concurrency "
                   "for 4,096 tokens per request: 18.30x\n")
    assert m.kv_mb == pytest.approx(9.81 * 1024) and m.kv_tokens == 74_976


# ---------------------------------------------------------------- the note in the plan


def test_layout_note_only_when_the_estimate_falls_back(hw_rtx4090):
    from rigtune.memory import layout_note

    vllm_mm = VllmBackend().memory_model(hw_rtx4090)
    assert layout_note(qwen25(), vllm_mm) is None  # dense
    assert layout_note(qwen35(), vllm_mm) is None  # hybrid, rules verified
    assert layout_note(gemma3("vllm"), vllm_mm) is None  # vLLM frees Gemma 3's sliding layers
    assert "rules are not verified" in layout_note(qwen35("llamacpp-cuda"), LlamaCppCudaBackend().memory_model(
        hw_rtx4090))
    note = layout_note(gemma3("sglang"), SglangBackend().memory_model(hw_rtx4090))
    assert note.startswith("SGLang keeps a full KV cache on every layer of Gemma3ForConditionalGeneration")
    odd = qwen35()
    odd.arch = odd.arch.model_copy(update={"full_attention_layers": None, "linear_layers": 0, "linear_state_bytes": 0,
                                           "layout_note": "unrecognised layer types mamba"})
    assert layout_note(odd, vllm_mm).startswith("Qwen3_5ForConditionalGeneration: unrecognised layer types mamba")


def test_plan_command_notes_an_unrecognised_layout(hw_a100, monkeypatch):
    import rigtune.backends.vllm as vl
    import rigtune.hardware as hwmod
    from typer.testing import CliRunner

    from rigtune.cli import app

    config = json.loads((FIXTURES / "qwen3_5_9b_config.json").read_text(encoding="utf-8"))
    config["text_config"]["layer_types"] = ["mamba"] * 24 + ["full_attention"] * 8
    arch = arch_from_config(config)
    monkeypatch.setattr(hwmod, "probe", lambda: hw_a100)
    monkeypatch.setattr(vl, "load_arch", lambda spec, token=None: arch)
    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: {arch.architecture})
    r = CliRunner().invoke(app, ["plan", "Qwen/Qwen3.5-9B", "--backend", "vllm", "--json"])
    assert r.exit_code == 0, r.output
    assert "unrecognised layer types mamba" in json.loads(r.output)["_notes"]["vllm"]
    r = CliRunner().invoke(app, ["plan", "Qwen/Qwen3.5-9B", "--backend", "vllm"])
    assert r.exit_code == 0 and "note: " in r.output


def test_plan_command_has_no_notes_for_a_dense_model(hw_a100, monkeypatch):
    import rigtune.hardware as hwmod
    from typer.testing import CliRunner

    from rigtune.cli import app

    monkeypatch.setattr(hwmod, "probe", lambda: hw_a100)
    r = CliRunner().invoke(app, ["plan", "meta-llama/Llama-3.2-3B-Instruct", "--backend", "vllm", "--json"])
    assert r.exit_code == 0 and "_notes" not in json.loads(r.output)


def test_a_speculation_setting_refused_for_state_memory_is_noted(hw_rtx4090):
    from rigtune.memory import spec_state_notes

    mm = VllmBackend().memory_model(hw_rtx4090)
    model = qwen35(weights={"fp8": 9_653_100_528})
    base = Config(backend="vllm", quant="fp8", ctx=4096, batch=16, gpu_memory_utilization=0.95)
    suffix = base.model_copy(update={"spec_decode": "suffix:24"})
    assert estimate(hw_rtx4090, model, base, mm).feasible and not estimate(hw_rtx4090, model, suffix, mm).feasible
    notes = spec_state_notes(hw_rtx4090, model, mm, base, [suffix])
    # 16 sequences x 25 states x 24 layers x 2,555,904 B = 24,536,678,400 B (22.9 GiB).
    assert notes == ["vllm suffix:24 at batch 16 does not fit: its recurrent state alone is 22.9 GiB (400 states "
                     "of 58.5 MiB); the search does not lower the batch for a speculation setting"]
    dense = qwen25()
    plain = base.model_copy(update={"ctx": 4096})
    assert spec_state_notes(hw_rtx4090, dense, mm, plain, [plain.model_copy(update={"spec_decode": "suffix:24"})]) == []


def test_profile_notes_for_speculation_skip_dense_models(hw_rtx4090):
    from rigtune.pipeline import PlanResult, SearchOptions, spec_state_notes_for

    reg = {"vllm": VllmBackend()}
    pick = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.95)
    dense = PlanResult(hw=hw_rtx4090, spec=ModelSpec(hf_id="x"), candidates=["vllm"], prepared={"vllm": qwen25()})
    assert spec_state_notes_for(hw_rtx4090, reg, dense, pick, SearchOptions()) == []
    hybrid = PlanResult(hw=hw_rtx4090, spec=ModelSpec(hf_id="x"), candidates=["vllm"],
                        prepared={"vllm": qwen35(weights={"bf16": 19_306_201_056})})
    assert spec_state_notes_for(hw_rtx4090, reg, hybrid, pick, SearchOptions(speculative=False)) == []
