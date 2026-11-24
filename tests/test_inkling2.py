"""thinkingmachines/Inkling-2-Mini read from its config (tests/fixtures/inkling_2_mini_config.json, the Hub's config.json
of 20 Nov 2026 with the per-tensor quantization lists dropped; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigtune.hfconfig import arch_from_config, vllm_context_length
from rigtune.models import Config

INKLING2 = json.loads((Path(__file__).parent / "fixtures" / "inkling_2_mini_config.json").read_text("utf-8"))


def test_its_context_is_its_model_max_length():
    # text_config sets only model_max_length, which vLLM's config class copies to max_position_embeddings.
    assert arch_from_config(INKLING2).max_position_embeddings == 1_048_576


def test_its_layout_counts_each_loop_pass_s_caches():
    # 20 local layers (window 1,024) and 4 full (5, 11, 17, 23); layers 6-17 run 3 passes, each with its own caches:
    # full 5, 11 x 3, 17 x 3, 23 = 8 and 40 sliding, as vLLM and SGLang lay them out.
    a = arch_from_config(INKLING2)
    assert (a.num_layers, a.loop_extra_layers, a.full_attention_layers, a.sliding_layers, a.sliding_window) == \
        (24, 24, 8, 40, 1024)
    assert a.full_kv_bytes_per_token(2) == 8 * 2 * 8 * 128 * 2  # 32 KiB a token in bf16
    assert a.kv_bytes_per_token(2) == 48 * 2 * 8 * 128 * 2  # the every-layer-full fallback counts 48 caches too


def _text(**fields):
    return {"architectures": ["X"], "num_hidden_layers": 6, "hidden_size": 256, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 64, **fields}


INKLING2_ARCH = ["Inkling2LoopForConditionalGeneration"]


def test_local_layer_ids_alone_make_a_sliding_layout():
    a = arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1, 3, 4], sliding_window_size=512))
    assert (a.full_attention_layers, a.sliding_layers, a.sliding_window, a.loop_extra_layers) == (2, 4, 512, 0)
    # Without a window the ids say nothing that can be planned: every layer full, as before.
    assert not arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1])).is_hybrid
    # Only Inkling 2's are read: another family's (the first Inkling's) were not checked.
    assert not arch_from_config(_text(local_layer_ids=[0, 1, 3, 4], sliding_window_size=512)).is_hybrid


def test_sliding_layers_sized_apart_from_the_full_ones_are_not_planned_as_if_alike():
    for field in ({"swa_head_dim": 32}, {"swa_num_key_value_heads": 1}):
        a = arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1], sliding_window_size=512,
                                   **field))
        assert not a.is_hybrid and "are not modelled" in a.layout_note
    # Equal to the full layers' (Inkling-2-Mini's own: 128 and 8), as planned.
    assert arch_from_config(INKLING2).is_hybrid


def test_sglang_plans_it_as_every_layer_full_and_says_why(hw_a100):
    from rigtune.backends.sglang import SglangBackend

    arch = arch_from_config(INKLING2)
    mm = SglangBackend().memory_model(hw_a100)
    reason = mm.full_kv_reason(arch)
    assert "not modelled" in reason and "every layer full" in reason
    assert mm.hybrid_for(Config(backend="sglang", quant="bf16", ctx=4096, batch=16), arch) is None


# ---------------------------------------------------------------- vLLM main's cache for it (F3b)

# The first H100 session's five start-ups (benchmarks/sessions/2026-11/runpod-inkling-2-mini-h100nvl; vLLM
# 5a215c27fd76): vLLM's logged pool blocks, its concurrency at max_model_len, and the blocks a request holds that both
# give (pool / concurrency), for (spec, budget, max_model_len).
SESSION_1 = [  # name, spec, budget, max_model_len, pool blocks (logged), concurrency (logged), blocks a request
    ("recipe", "dspark:7", 8192, 262144, 22480, 1.93, 11671),
    ("nospec", None, 8192, 262144, 33432, 2.88, 11617),
    ("budget2048", "dspark:7", 2048, 262144, 28376, 4.65, 6103),
    ("ctx32k", "dspark:7", 8192, 32768, 22824, 2.82, 8087),
]


@pytest.mark.parametrize("name, spec, budget, ctx, pool, conc, per_request", SESSION_1)
def test_vllm_plans_it_by_the_rule_that_reproduced_session_1(name, spec, budget, ctx, pool, conc, per_request):
    arch = arch_from_config(INKLING2)
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=ctx, batch=16, gpu_memory_utilization=0.9,
                 spec_decode=spec, prefill_budget=budget)
    h = vl.vllm_inkling2_kv(cfg, arch, 8192)
    block = (9 if spec else 8) * 256 * 1024
    # One request at the context, plus the null block: what vLLM checks before it starts.
    assert h.min_bytes == (per_request + 1) * block
    assert round(pool / per_request, 2) == conc  # the logged concurrency, from the logged pool


def test_vllm_plans_a_batch_as_its_tokens_and_its_requests():
    from rigtune.memory import kv_cache_bytes

    arch, pm = arch_from_config(INKLING2), _prepared()
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=64, gpu_memory_utilization=0.9,
                 spec_decode="dspark:7", prefill_budget=2048)
    h = vl.vllm_inkling2_kv(cfg, arch, 8192)
    block = 9 * 256 * 1024
    # Each request: its conv windows (6 groups x 2 blocks of 16), its drafter window (cdiv(1032, 64) + 1 = 18), and the
    # sliding block its lookahead adds in each of 5 groups; the in-flight tokens (2 x 2,048) once, in the sliding groups
    # and the conv groups.
    assert h.request_bytes == (6 * 2 + 18 + 5) * block
    assert h.shared_bytes == 6 * 4096 * (block // 16)
    assert (h.sliding_layers, h.sliding_block, h.swa_extra_tokens) == (45, 64, 4096)
    kv = kv_cache_bytes(pm, cfg, 64 * 2048, h)  # 2,048 tokens each
    full = 64 * 2048 * 9 * 4096
    sliding = (64 * (1024 + 64) + 4096) * 45 * 4096  # a window in 64-token blocks, one more once past it
    assert kv == full + sliding + 64 * (6 * 2 + 18 + 5) * block + 6 * 4096 * (block // 16)
    # The start-up minimum stays the admission bound of one request at the context.
    assert h.min_bytes == ((4096 + 63) // 64 + 5 * 65 + 6 * 257 + 54 + 1) * block
    # A KV dtype the rule was not fitted at keeps the every-layer-full estimate.
    assert vl.vllm_inkling2_kv(cfg.model_copy(update={"kv_dtype": "fp8"}), arch, 8192) is None


def test_what_the_rule_was_not_fitted_at_is_refused_not_planned_wrong():
    pm, be = _prepared(), vl.VllmBackend()
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    assert be.config_refusal(cfg, pm) is None
    for other in ({"tp": 2}, {"kv_dtype": "fp8"}, {"kv_dtype": "fp8_e5m2"}):
        assert "only as it was measured" in be.config_refusal(cfg.model_copy(update=other), pm)
    # Another family is untouched.
    qwen = pm.model_copy(update={"arch": pm.arch.model_copy(update={"architecture": "Qwen3ForCausalLM"})})
    assert be.config_refusal(cfg.model_copy(update={"tp": 2}), qwen) is None


def test_its_warm_up_reserve_is_added_to_other_drafters_weights_not_in_their_place():
    pm = _prepared()
    pm.spec_repos["dspark"] = "someone/inkling2-dspark"
    pm.spec_bytes["someone/inkling2-dspark"] = 700_000_000
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=64, gpu_memory_utilization=0.9,
                 spec_decode="dspark:someone/inkling2-dspark:7")
    assert vl.vllm_mtp_extra_weights(cfg, pm) == 700_000_000 + vl.inkling2_warmup_reserve(cfg, pm.arch)
    plain = cfg.model_copy(update={"spec_decode": None})
    assert vl.vllm_mtp_extra_weights(plain, pm) == vl.inkling2_warmup_reserve(plain, pm.arch)


def test_a_cached_session_does_not_hold_the_in_flight_tokens(hw_a100):
    from rigtune.memory import kv_cache_bytes, working_set_bytes

    pm, mm = _prepared(), vl.VllmBackend().memory_model(hw_a100)
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    held = working_set_bytes(pm, cfg, 16 * 2048, 16, mm)
    running = kv_cache_bytes(pm, cfg, 16 * 2048, mm.hybrid_for(cfg, pm.arch))
    assert held < running  # the in-flight conv tokens (and the start-up floor) are a running batch's


def test_sglang_counts_its_own_drafter_s_layers():
    from rigtune.backends.sglang import sglang_drafter_layers

    cfg = Config(backend="sglang", quant="ckpt-nvfp4", ctx=4096, batch=16, spec_decode="dspark:7")
    assert sglang_drafter_layers(cfg, _prepared().model_copy(update={"backend": "sglang"})) == 3


def test_vllm_s_memory_model_takes_it_through_the_rule(hw_a100):
    arch = arch_from_config(INKLING2)
    mm = vl.VllmBackend().memory_model(hw_a100)
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    assert mm.full_kv_reason(arch) is None
    assert mm.hybrid_for(cfg, arch).request_bytes > 0  # vllm_inkling2_kv, not the Gemma-style rules


def test_the_warm_up_reserve_is_never_below_session_1_s():
    arch = arch_from_config(INKLING2)
    GiB = 1 << 30

    def reserve(batch, spec, budget):
        return vl.inkling2_warmup_reserve(Config(backend="vllm", quant="ckpt-nvfp4", ctx=262144, batch=batch,
                                                 spec_decode=spec, prefill_budget=budget), arch) / GiB
    # Held: the recipe 17.55 GiB (1,024 requests x 8 tokens), budget 2,048 5.74, no drafter 6.03, 32 requests 5.13.
    assert round(reserve(1024, "dspark:7", 8192), 2) == 17.55
    assert 5.74 <= reserve(1024, "dspark:7", 2048) <= 6.05  # the 3.14 GiB floor: 6.04 where 5.74 was held
    assert reserve(1024, None, 8192) >= 6.03 and reserve(32, "dspark:7", 8192) >= 5.13
    # rigtune's own batch cap is a lever: 64 requests at k 7 hold back a third of the recipe's.
    assert reserve(64, "dspark:7", 8192) < 17.55 / 2


def test_a_loop_s_passes_are_counted_whatever_the_layout():
    loop = {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": False}
    dense = arch_from_config(_text(loop_config=loop))
    assert (dense.loop_extra_layers, dense.is_hybrid) == (4, False)
    assert dense.kv_bytes_per_token(2) == (6 + 4) * 2 * 2 * 64 * 2
    # A layout that falls back to every layer full still keeps the passes.
    odd = arch_from_config(_text(loop_config=loop, layer_types=["full_attention", "sliding_attention"]))
    assert odd.layout_note and odd.loop_extra_layers == 4
    # A loop whose fields do not make one is ignored, as is one pass.
    for bad in ({**loop, "core_end": 9}, {**loop, "num_passes": "3"}, {**loop, "num_passes": 1}):
        assert arch_from_config(_text(loop_config=bad)).loop_extra_layers == 0


# ---------------------------------------------------------------- its own DSpark drafter (F4)

from rigtune import speculative as sp  # noqa: E402
from rigtune.backends import vllm as vl  # noqa: E402
from rigtune.backends.facts import EngineFacts  # noqa: E402
from rigtune.introspect import own_block_drafter  # noqa: E402
from rigtune.models import ModelSpec, PreparedModel  # noqa: E402


DRAFTER_BYTES = 640_000_000  # mtp.safetensors, ~0.64 GB (the recipe: "~20 GB on disk including the 0.6 GB draft")


def _prepared(**arch) -> PreparedModel:
    a = arch_from_config(INKLING2).model_copy(update={"own_drafter_bytes": DRAFTER_BYTES, **arch})
    return PreparedModel(spec=ModelSpec(hf_id="thinkingmachines/Inkling-2-Mini"), backend="vllm", arch=a,
                         hf_path="thinkingmachines/Inkling-2-Mini", weights_bytes={"ckpt-nvfp4": 19_526_019_168})


def test_its_mtp_config_declares_a_block_drafter_not_an_mtp_head():
    assert own_block_drafter(INKLING2) == (3, 7)  # 3 layers, block 8: an anchor and 7 drafts
    a = arch_from_config(INKLING2)
    assert (a.own_drafter_layers, a.own_drafter_tokens, a.mtp_layers) == (3, 7, 0)
    # Another family's mtp_config, or one without a usable block, declares none.
    assert own_block_drafter({**INKLING2, "architectures": ["InklingForConditionalGeneration"]}) is None
    for mtp in ({"num_hidden_layers": 3}, {"num_hidden_layers": 3, "block_size": 1}, {"block_size": 8}):
        assert own_block_drafter({**INKLING2, "mtp_config": mtp}) is None


def test_a_drafter_spec_without_a_repository_is_the_checkpoint_s_own():
    assert sp.parse("dspark:7") == ("dspark", None, 7) and sp.own_drafter("dspark:7")
    assert not sp.own_drafter("dspark:deepseek-ai/dspark_qwen3_8b_block7:7") and not sp.own_drafter("mtp:3")
    # vLLM takes the target as the draft's model when none is given; the recipe's sampling where asked for.
    assert sp.vllm_config("dspark:7", "abc") == {"method": "dspark", "num_speculative_tokens": 7, "revision": "abc"}
    assert sp.vllm_config("dspark:7", sample_method="probabilistic")["draft_sample_method"] == "probabilistic"
    with pytest.raises(ValueError, match="SGLang is not offered"):
        sp.sglang_config("dspark:7")
    with pytest.raises(ValueError):  # a DFlash drafter still names its repository
        sp.parse("dflash:15")


@pytest.fixture
def nightly(monkeypatch):
    """A vLLM whose source converts Inkling 2 for its drafter (main after PR #69028)."""
    monkeypatch.setattr(vl, "_vllm_runs_own_drafter", lambda: True)
    facts = EngineFacts(engine="vllm", version="0.35.0", spec_fields=["draft_sample_method"])
    monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)


def _h100(hw_a100):
    gpu = hw_a100.gpus[0].model_copy(update={"name": "NVIDIA H100 80GB HBM3", "compute_capability": (9, 0)})
    return hw_a100.model_copy(update={"gpus": [gpu]})


def test_vllm_offers_it_at_7_with_the_budget_for_8_tokens_a_request(nightly, hw_a100):
    # An A100's default budget is 2,048 tokens: 512 requests at 8 tokens a step need 4,096, and 7 would give 3,584.
    # A batch above the card's default budget // 8 raises the budget to batch x 8 (7 would give batch x 7). That many
    # requests' conv and drafter windows, and the in-flight conv tokens at such a budget, need a card far larger than an
    # A100: a 4 TiB one, for the budget arithmetic alone.
    pm, be = _prepared(), vl.VllmBackend()
    big = hw_a100.gpus[0].model_copy(update={"vram_total_bytes": 4 << 40, "vram_free_bytes": (4 << 40) - (1 << 30)})
    hw_a100 = hw_a100.model_copy(update={"gpus": [big]})
    batch = vl.vllm_default_batched_tokens(hw_a100) // 8 + 64
    base = Config(backend="vllm", quant="ckpt-nvfp4", ctx=256, batch=batch, gpu_memory_utilization=0.95)
    variants, notes = be._own_drafter_variants(base, pm, hw_a100, True, None)
    assert [v.spec_decode for v in variants] == ["dspark:7"]
    assert variants[0].prefill_budget == batch * 8 and f"max_num_batched_tokens {batch * 8}" in notes[0]
    assert vl.vllm_drafter_layers(variants[0], pm) == 0  # its layers are in vllm_inkling2_kv's groups
    assert vl.vllm_drafter_layers(variants[0].model_copy(update={"kv_dtype": "fp8"}), pm) == 3  # else full layers
    # Not under --speculative on.
    assert be._own_drafter_variants(base, pm, hw_a100, False, None) == ([], [])


def test_its_weights_fold_and_sampled_logits_are_planned(nightly, monkeypatch):
    pm = _prepared()
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=64, gpu_memory_utilization=0.9,
                 spec_decode="dspark:7")
    fold, logits = 201024 * 256 * 2, 64 * 7 * 201024 * 2  # 103 MB; 180 MB at 64 requests
    assert pm.arch.own_drafter_fold_bytes == fold
    warmup = vl.inkling2_warmup_reserve(cfg, pm.arch)
    assert vl.vllm_mtp_extra_weights(cfg, pm) == DRAFTER_BYTES + fold + logits + warmup
    # A vLLM the probe found without the field drafts greedily: no logits buffer.
    monkeypatch.setattr(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", version="0.35.0", spec_fields=[]))
    assert vl.vllm_mtp_extra_weights(cfg, pm) == DRAFTER_BYTES + fold + warmup


def test_a_drafter_of_unknown_size_or_a_quantized_copy_is_not_offered_it(nightly, hw_a100):
    base = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    offered, notes = vl.VllmBackend()._own_drafter_variants(base, _prepared(own_drafter_bytes=None), hw_a100, True,
                                                            None)
    assert offered == [] and "mtp.safetensors" in notes[0]
    awq = base.model_copy(update={"quant": "awq"})
    assert vl.VllmBackend()._own_drafter_variants(awq, _prepared(), hw_a100, True, None) == ([], [])


def test_its_budget_is_unstated_again_when_it_leads(nightly, hw_a100):
    # Else n-gram and the rest would be offered again at an explicit budget only the drafter needs.
    be = vl.VllmBackend()
    leader = Config(backend="vllm", quant="ckpt-nvfp4", ctx=512, batch=512, gpu_memory_utilization=0.95,
                    spec_decode="dspark:7", prefill_budget=4096)
    assert be._before_block_drafter(leader, _prepared(), hw_a100).prefill_budget is None
    chosen = leader.model_copy(update={"prefill_budget": 8192})  # a budget the prefill stage chose is kept
    assert be._before_block_drafter(chosen, _prepared(), hw_a100).prefill_budget == 8192


def test_sglang_says_why_it_is_not_offered_it(hw_a100):
    from rigtune.backends.sglang import SglangBackend

    cfg = Config(backend="sglang", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.88)
    pm = _prepared().model_copy(update={"backend": "sglang"})
    assert any("#51926" in n for n in SglangBackend().spec_notes(cfg, pm, hw_a100, True, None))


def test_its_size_is_read_from_the_hub_listing(monkeypatch):
    from rigtune import hfconfig

    monkeypatch.setattr(hfconfig, "fetch_config", lambda spec, token=None: INKLING2)
    listing = {"model-00001-of-00001.safetensors": 19_525_870_752, "mtp.safetensors": DRAFTER_BYTES}
    monkeypatch.setattr(hfconfig, "hub_weights", lambda spec, token=None: (None, 19_525_870_752, listing))
    a = hfconfig.load_arch(ModelSpec(hf_id="thinkingmachines/Inkling-2-Mini"))
    assert (a.own_drafter_bytes, a.checkpoint_bytes) == (DRAFTER_BYTES, 19_525_870_752)  # not in the index's files


def test_vllm_without_the_conversion_is_not_offered_it(hw_a100, monkeypatch):
    monkeypatch.setattr(vl, "_vllm_runs_own_drafter", lambda: False)
    base = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    offered, notes = vl.VllmBackend()._own_drafter_variants(base, _prepared(), _h100(hw_a100), True, None)
    assert offered == [] and "Inkling2DSparkConfig" in notes[0]


def test_its_launch_is_the_recipe_s_speculative_config(nightly, monkeypatch):
    # Probabilistic drafting wherever offered: also when the probe could not say (the field predates the drafter).
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=16, gpu_memory_utilization=0.9,
                 spec_decode="dspark:7", prefill_budget=2048)
    args = vl.VllmBackend().launch_spec(cfg, _prepared(), 8000).args
    payload = json.loads(args[args.index("--speculative-config") + 1])
    assert payload == {"method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}
    monkeypatch.setattr(vl, "_engine_facts", lambda: None)
    args = vl.VllmBackend().launch_spec(cfg, _prepared(), 8000).args
    assert json.loads(args[args.index("--speculative-config") + 1])["draft_sample_method"] == "probabilistic"
    # A vLLM without the field gets vLLM's own default (greedy), not a flag it would refuse.
    monkeypatch.setattr(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", version="0.35.0", spec_fields=[]))
    args = vl.VllmBackend().launch_spec(cfg, _prepared(), 8000).args
    assert "draft_sample_method" not in json.loads(args[args.index("--speculative-config") + 1])


def test_model_runner_v1_and_a_short_budget_refuse_it(nightly, monkeypatch):
    cfg = Config(backend="vllm", quant="ckpt-nvfp4", ctx=4096, batch=64, gpu_memory_utilization=0.9,
                 spec_decode="dspark:7", prefill_budget=256)
    assert "run only 32 of batch 64" in vl.VllmBackend().config_refusal(cfg, _prepared())  # 256 // 8 a request
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert "Model Runner V2" in vl.VllmBackend().config_refusal(cfg, _prepared())


@pytest.mark.parametrize("cfg, expected", [
    ({"max_position_embeddings": 32768, "seq_length": 8192}, 8192),  # the smallest key wins (Qwen/Qwen-7B-Chat)
    ({"n_positions": 1024}, 1024),  # GPT-2
    ({"max_position_embeddings": 8192, "model_max_length": 131072}, 131072),  # model_max_length replaces it (Cohere)
    ({"max_seq_len": 2048, "max_target_positions": 448}, 448),
    ({"n_ctx": 2048}, 2048),  # none of vLLM's keys: as before
    ({}, 4096),
])
def test_the_context_is_derived_as_vllm_derives_it(cfg, expected):
    assert vllm_context_length(cfg) == expected
