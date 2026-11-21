"""A checkpoint's own MTP head as a speculation setting, under --speculative all only.

Engine releases are seen through tests/engine_trees.py; memory figures are written out by hand from the engines'
rules (cited in rigtune/backends/vllm.py and sglang.py), not computed with the code under test. Sizes are those
Qwen/Qwen3.5-9B's safetensors headers list. Nothing here needs a GPU or the network: the head is found through an
injected introspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from rigtune import introspect, speculative
from rigtune.backends import own_head
from rigtune.backends.sglang import SglangBackend
from rigtune.backends.vllm import AMPERE_FP8_KV, VllmBackend, vllm_hybrid_kv
from rigtune.hfconfig import arch_from_config
from rigtune.memory import estimate
from rigtune.models import Config, ModelSpec, PreparedModel
from tests.engine_trees import SGLANG_PYTHON, VERSIONS, installed
from tests.golden_cases import hw as golden_hw
from tests.test_hybrid_memory import STATE, qwen25, qwen35

HEAD_BYTES = 486_581_248  # Qwen/Qwen3.5-9B's 15 mtp.* tensors
SHARED_BYTES = 2 * 2_034_237_440  # its embed_tokens and lm_head, each 248,320 x 4,096 x 2 B
WEIGHTS = {"bf16": 19_306_201_056, "fp8": 9_653_100_528}
FIXTURES = Path(__file__).parent / "fixtures"


def with_head(backend="vllm", head_bytes=HEAD_BYTES, shared_bytes=SHARED_BYTES, revision=None, hf_id="Qwen/Qwen3.5-9B"):
    pm = qwen35(backend, weights=dict(WEIGHTS))
    pm.spec = ModelSpec(hf_id=hf_id, revision=revision)
    pm.spec_repos = {"mtp": hf_id}
    pm.spec_bytes = {k: v for k, v in (("mtp", head_bytes), ("mtp_shared", shared_bytes)) if v}
    return pm


def vcfg(**kw):
    return Config(**{"backend": "vllm", "quant": "fp8", "ctx": 4096, "batch": 16, "gpu_memory_utilization": 0.95,
                     **kw})


def scfg(**kw):
    return Config(**{"backend": "sglang", "quant": "fp8", "ctx": 4096, "batch": 16, "gpu_memory_utilization": 0.88,
                     **kw})


def specs(variants):
    return [v.spec_decode for v in variants]


# ---------------------------------------------------------------- the spec string


def test_the_spec_string():
    assert speculative.MTP_TOKENS == (1, 3)
    assert speculative.mtp(1) == "mtp:1"
    assert speculative.parse("mtp:1") == ("mtp", None, 1)
    assert speculative.parse("mtp:3") == ("mtp", None, 3)
    for bad in ("mtp", "mtp:0", "mtp:-1"):
        with pytest.raises(ValueError):
            speculative.parse(bad)
    assert speculative.own_head("mtp:1") and not speculative.own_head("ngram:4")
    assert not speculative.own_head(None) and not speculative.own_head("eagle3:a/mtp:3")


@pytest.mark.parametrize("k", [1, 3])
def test_rendered_for_each_engine(k):
    assert speculative.vllm_config(f"mtp:{k}") == {"method": "mtp", "num_speculative_tokens": k}
    assert speculative.vllm_config(f"mtp:{k}", revision="abc123") == {"method": "mtp", "num_speculative_tokens": k,
                                                                       "revision": "abc123"}
    assert speculative.sglang_config(f"mtp:{k}", model_path="Qwen/Qwen3.5-9B") == [
        "--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", "Qwen/Qwen3.5-9B",
        "--speculative-num-steps", str(k), "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", str(k + 1)]
    assert speculative.sglang_config(f"mtp:{k}", model_path="Qwen/Qwen3.5-9B", revision="abc123")[4:6] == [
        "--speculative-draft-model-revision", "abc123"]


def test_other_methods_ignore_the_targets_path_and_revision():
    assert "revision" not in speculative.vllm_config("ngram_gpu:4", revision="abc123")
    with pytest.raises(ValueError):
        speculative.sglang_config("mtp:3")  # the head is the target's own: its path is needed
    assert speculative.sglang_config("ngram:12", model_path="x", revision="y") == [
        "--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", "12"]


def test_what_sglangs_draft_holds_while_its_pools_are_sized():
    # The head, and the draft's own embedding and output head, which init_lm_head swaps out only afterwards.
    assert speculative.draft_bytes("mtp:3", with_head()) == HEAD_BYTES + SHARED_BYTES == 4_555_056_128
    assert speculative.draft_bytes("mtp:3", with_head(head_bytes=None)) is None
    assert speculative.draft_bytes("mtp:3", with_head(shared_bytes=None)) is None
    big = PreparedModel(spec=ModelSpec(hf_id="Qwen/Qwen3.8-27B"), backend="sglang",
                        arch=arch_from_config(json.loads((FIXTURES / "qwen3_8_27b_config.json").read_text("utf-8"))),
                        spec_repos={"mtp": "Qwen/Qwen3.8-27B"},
                        spec_bytes={"mtp": 849_398_784, "mtp_shared": 5_085_593_600})
    assert big.arch.mtp_layers == 1 and big.arch.architecture in speculative.MTP_ARCHS
    assert speculative.draft_bytes("mtp:1", big) == 849_398_784 + 5_085_593_600  # 248,320 x 5,120 x 2 B, twice
    assert sg.sglang_draft_tokens("mtp:1") == 2 and sg.sglang_draft_tokens("mtp:3") == 4
    assert sg.sglang_decode_alloc("mtp:1") == 4 and sg.sglang_decode_alloc("mtp:3") == 8  # 2 x max(steps, draft)


# ---------------------------------------------------------------- what each release offers, read from its source


@pytest.mark.parametrize("name", VERSIONS)
def test_every_checked_release_can_load_a_qwen3_5_head(name, tmp_path):
    """Read from source: on vLLM, what is checked when vLLM is not asked (tests/test_vllm_own_head.py)."""
    engine, version = name.split("-")
    with installed(engine, version, tmp_path):
        found = vl._mtp_archs() if engine == "vllm" else sg.sglang_mtp_archs()
    assert found == speculative.MTP_ARCHS


def test_vllm_reads_the_model_types_of_the_branch_that_rewrites():
    spec = ('        if hf_config.model_type in (\n            "qwen3_5",\n            "qwen3_5_text",\n        ):\n'
            '            is_moe = hf_config.model_type in ("qwen3_5_moe", "qwen3_5_moe_text")\n'
            '            hf_config.model_type = "qwen3_5_mtp"\n')
    model = "class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):\n"
    files = vl.MTP_SOURCES
    # The MoE types on the is_moe line are not the branch's own test, and there is no MoE draft class.
    assert vl.mtp_archs_in({files[0]: spec, files[1]: model}) == {"Qwen3_5ForConditionalGeneration",
                                                                 "Qwen3_5ForCausalLM"}
    assert vl.mtp_archs_in({files[0]: spec.replace("qwen3_5_mtp", "x"), files[1]: model}) == frozenset()
    assert vl.mtp_archs_in({files[0]: spec}) == frozenset()  # no draft class
    assert vl.mtp_archs_in({}) == frozenset()


def test_sglang_reads_the_architectures_of_the_branch_that_rewrites():
    args = "Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK"
    rewrite = ('        if is_draft_model and self.hf_config.architectures[0] in [\n'
               '            "Qwen3_5ForConditionalGeneration",\n        ]:\n'
               '            self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"\n')
    elsewhere = 'MAMBA_ARCHS = ["Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLM"]\n'  # named, but not rewritten
    cls = "class Qwen3_5ForCausalLMMTP(nn.Module):\n"
    assert sg.mtp_archs_in(args, elsewhere + rewrite, cls) == {"Qwen3_5ForConditionalGeneration"}
    assert sg.mtp_archs_in(args, elsewhere, cls) == frozenset()
    assert sg.mtp_archs_in('choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"]', rewrite, cls) == frozenset()
    assert sg.mtp_archs_in(args.replace("NEXTN, ", ""), rewrite, cls) == frozenset()
    assert sg.mtp_archs_in(args, rewrite, "") == frozenset()


# ---------------------------------------------------------------- exact launch flags, per engine and release


@pytest.mark.parametrize("version", ["0.29.0", "0.33.0"])
@pytest.mark.parametrize("k", [1, 3])
def test_vllm_launch(version, k, tmp_path):
    with installed("vllm", version, tmp_path):
        be, pm = VllmBackend(), with_head()
        plain = be.launch_spec(vcfg(), pm, 8000).args
        args = be.launch_spec(vcfg(spec_decode=f"mtp:{k}"), pm, 8000).args
        pinned = be.launch_spec(vcfg(spec_decode=f"mtp:{k}"), with_head(revision="abc123"), 8000).args
    i = args.index("--speculative-config")
    assert args[i:i + 2] == ["--speculative-config", f'{{"method": "mtp", "num_speculative_tokens": {k}}}']
    assert args[:i] + args[i + 2:] == plain  # nothing else moves
    j = pinned.index("--speculative-config")
    assert json.loads(pinned[j + 1]) == {"method": "mtp", "num_speculative_tokens": k, "revision": "abc123"}
    assert pinned[-2:] == ["--revision", "abc123"]  # the target's own pin, as before


@pytest.mark.parametrize("version", ["0.5.19", "0.5.23"])
@pytest.mark.parametrize("k", [1, 3])
def test_sglang_launch(version, k, tmp_path):
    with installed("sglang", version, tmp_path):
        be, pm = SglangBackend(), with_head("sglang")
        plain = be.launch_spec(scfg(), pm, 8000).args
        args = be.launch_spec(scfg(spec_decode=f"mtp:{k}"), pm, 8000).args
        pinned = be.launch_spec(scfg(spec_decode=f"mtp:{k}"), with_head("sglang", revision="abc123"), 8000).args
    assert args[0] == SGLANG_PYTHON and args[args.index("--model-path") + 1] == "Qwen/Qwen3.5-9B"
    flags = ["--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", "Qwen/Qwen3.5-9B",
             "--speculative-num-steps", str(k), "--speculative-eagle-topk", "1",
             "--speculative-num-draft-tokens", str(k + 1)]
    i = args.index("--speculative-algorithm")
    assert args[i:i + 10] == flags and args[:i] + args[i + 10:] == plain
    j = pinned.index("--speculative-algorithm")
    assert pinned[j:j + 12] == flags[:4] + ["--speculative-draft-model-revision", "abc123"] + flags[4:]
    assert pinned[-2:] == ["--revision", "abc123"]


# ---------------------------------------------------------------- when it is offered


def test_both_depths_offered_on_vllm_only_under_all_with_a_head_and_support(tmp_path, tmp_home, hw_rtx4090):
    with installed("vllm", "0.33.0", tmp_path / "v"):
        be = VllmBackend()
        on = specs(be.spec_variants(vcfg(), with_head(), hw_rtx4090))
        assert not any(s.startswith("mtp") for s in on)  # --speculative on: exactly as before
        assert specs(be.spec_variants(vcfg(), with_head(), hw_rtx4090, all_methods=True)) == on + ["mtp:1", "mtp:3"]
        assert specs(be.spec_variants(vcfg(spec_decode="mtp:1"), with_head(), hw_rtx4090, True))[-1] == "mtp:3"
        assert "mtp:1" not in specs(be.spec_variants(vcfg(), qwen35(weights=dict(WEIGHTS)), hw_rtx4090, True))
        assert "mtp:1" not in specs(be.spec_variants(vcfg(), with_head(), None, True))  # no machine, no fit check
        assert "mtp:1" not in specs(be.spec_variants(vcfg(quant="awq"), with_head(), hw_rtx4090, True))
    # Whether this vLLM loads the head is decided when it is recorded, by asking vLLM (tests/test_vllm_own_head.py).


def test_both_depths_offered_on_sglang_with_room_for_the_draft(tmp_path, tmp_home, hw_rtx4090, hw_a100):
    small = scfg(batch=4)
    with installed("sglang", "0.5.23", tmp_path / "s"):
        be = SglangBackend()
        assert be.spec_variants(small, with_head("sglang"), hw_rtx4090) == []  # --speculative on
        got = [(v.spec_decode, v.gpu_memory_utilization, v.gmu_before_spec)
               for v in be.spec_variants(small, with_head("sglang"), hw_rtx4090, True) if v.spec_decode[:3] == "mtp"]
        # 1 - (486,581,248 + 4,068,474,880 + 2 GiB workspace + 512 MiB) / 24 GiB = 1 - 7,239,410,688 /
        # 25,769,803,776 = 0.719 -> 0.71
        assert got == [("mtp:1", 0.71, 0.88), ("mtp:3", 0.71, 0.88)]
        wide = [(v.spec_decode, v.gpu_memory_utilization, v.gmu_before_spec)
                for v in be.spec_variants(scfg(batch=4, gpu_memory_utilization=0.93), with_head("sglang"), hw_a100,
                                          True) if v.spec_decode[:3] == "mtp"]
        assert wide == [("mtp:1", 0.91, 0.93), ("mtp:3", 0.91, 0.93)]  # 1 - 7,239,410,688 / 80 GiB = 0.916
        assert "mtp:3" not in specs(be.spec_variants(small, with_head("sglang"), None, True))
        assert "mtp:3" not in specs(be.spec_variants(small, qwen35("sglang", dict(WEIGHTS)), hw_rtx4090, True))
        # Sizes that could not be read leave no fraction known to fit the draft, as for a second model.
        assert "mtp:3" not in specs(be.spec_variants(small, with_head("sglang", head_bytes=None), hw_rtx4090, True))
        assert "mtp:3" not in specs(be.spec_variants(small, with_head("sglang", shared_bytes=None), hw_rtx4090,
                                                     True))
    # Whether this SGLang loads the head is decided when it is recorded (tests/test_sglang_own_head.py).
    assert SglangBackend().tier_host_need(scfg(spec_decode="mtp:3"), 1000) == 2000  # the draft's host pool


def test_vllm_does_not_pair_it_with_the_fp8_cache_through_flashinfer_and_says_so(tmp_path, tmp_home):
    be = VllmBackend()
    assert be.known_failure(vcfg(spec_decode="mtp:1", kv_dtype=AMPERE_FP8_KV)).startswith("MTP with the fp8 KV")
    assert be.known_failure(vcfg(spec_decode="mtp:3", kv_dtype="fp8")) is None
    assert be.known_failure(vcfg(spec_decode="ngram_gpu:4", kv_dtype=AMPERE_FP8_KV)) is None
    a40 = golden_hw("a40", "vllm", "0.33.0")
    leader = vcfg(quant="bf16", kv_dtype=AMPERE_FP8_KV, gpu_memory_utilization=0.90)
    with installed("vllm", "0.33.0", tmp_path):  # flashinfer present: the A40's search has fp8_e5m2
        assert AMPERE_FP8_KV in be.kv_dtypes(a40)
        assert not any(s.startswith("mtp") for s in specs(be.spec_variants(leader, with_head(), a40, True)))
        notes = be.spec_notes(leader, with_head(), a40, True)
        assert any(n.startswith("vllm: MTP is not paired with the fp8_e5m2 KV cache") and "unproven" in n
                   for n in notes)
        assert be.spec_notes(leader, with_head(), a40, False) == []  # nothing said without --speculative all
        assert not any("fp8_e5m2" in n for n in be.spec_notes(vcfg(), with_head(), golden_hw("rtx4090", "vllm",
                                                                                              "0.33.0"), True))


@pytest.mark.parametrize("version, refused", [("0.29.0", True), ("0.33.0", False)])
def test_vllm_0_29_offers_no_kv_tier_beside_mtp_on_a_linear_attention_model(version, refused, tmp_path,
                                                                            monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    tiered = vcfg(spec_decode="mtp:1", kv_tier="host:16")
    with installed("vllm", version, tmp_path):
        be = VllmBackend()
        reason = be.tier_refusal(tiered, with_head())
        assert (reason is not None and "counts every KV cache group as MTP's" in reason) is refused
        assert be.tier_refusal(tiered.model_copy(update={"spec_decode": None}), with_head()) is None
        assert be.tier_refusal(tiered.model_copy(update={"spec_decode": "ngram_gpu:4"}), with_head()) is None
        assert be.tier_refusal(tiered, qwen25()) is None  # no linear-attention layers


# ---------------------------------------------------------------- memory, with hand constants


def test_vllm_one_token_plans_the_heads_layer_and_one_more_state(hw_rtx4090, tmp_home):
    model, mm = with_head(), VllmBackend().memory_model(hw_rtx4090)
    c = vcfg(quant="bf16", gpu_memory_utilization=0.9, spec_decode="mtp:1")
    hk = vllm_hybrid_kv(c, model.arch)
    # A layer's state with one speculative token: 2,146,304 + 16,384 = 2,162,688 B = 33 x 65,536, so the block stays
    # 528 tokens and the state page is 528 x 4,096 = 2,162,688 B. The head's full-attention layer makes 9 full
    # layers against 24 linear: group size 9, the linear layers pad to 27 (state copy 27 x 2,162,688 = 58,392,576 B).
    assert (hk.full_block, hk.full_layers, hk.state_bytes, hk.state_slots) == (528, 9, 58_392_576, 32)
    # KV: 1,024 tokens a sequence round up to 1,056; 9 layers x 4,096 B = 36,864 B a token: 1,056 x 16 x 36,864.
    # State: 16 sequences x 2 states x 58,392,576 B.
    assert estimate(hw_rtx4090, model, c, mm).kv_cache == 1056 * 16 * 36_864 + 16 * 2 * 58_392_576 == 2_491_416_576
    # Start-up floor, one request at max_model_len with prefix caching (align mode): 9 x cdiv(4,096, 528) x 528 x
    # 4,096 of KV and 27 x (2 + 1) state pages, + a null block of 9 pages: 3 states against 2 without speculation.
    assert hk.min_bytes == 9 * 8 * 528 * 4096 + 27 * 3 * 2_162_688 + 9 * 2_162_688


def test_vllm_three_tokens_widen_the_block_and_hold_four_states(hw_rtx4090, tmp_home):
    model, mm = with_head(), VllmBackend().memory_model(hw_rtx4090)
    c = vcfg(quant="bf16", gpu_memory_utilization=0.9, spec_decode="mtp:3")
    hk = vllm_hybrid_kv(c, model.arch)
    # 2,146,304 + 3 x 16,384 = 2,195,456 B a layer; 16 x cdiv(2,195,456, 65,536) = 16 x 34 = 544-token blocks;
    # state page 544 x 4,096 = 2,228,224 B; 27 padded linear layers: 60,162,048 B a copy; 4 copies a sequence.
    assert (hk.full_block, hk.full_layers, hk.state_bytes, hk.state_slots) == (544, 9, 60_162_048, 64)
    # KV: 1,024 tokens round up to 1,088 (two 544-token blocks); 1,088 x 16 x 36,864 + 64 x 60,162,048.
    assert estimate(hw_rtx4090, model, c, mm).kv_cache == 1088 * 16 * 36_864 + 64 * 60_162_048 == 4_492_099_584
    # Start-up floor: 9 x cdiv(4,096, 544) = 8 blocks of 544 x 4,096, 27 x (2 + 3) state pages, a null block.
    assert hk.min_bytes == 9 * 8 * 544 * 4096 + 27 * 5 * 2_228_224 + 9 * 2_228_224 == 481_296_384


@pytest.mark.parametrize("k, draft", [(1, 2), (3, 4)])
def test_sglang_plans_the_heads_layer_and_its_draft_tokens(k, draft, hw_rtx4090, tmp_home):
    model, mm = with_head("sglang"), SglangBackend().memory_model(hw_rtx4090)
    # Radix cache off: 17 slots x (1 + draft tokens) states x 51,511,296 B; KV 16,384 tokens x 9 layers x 4,096 B
    # (the cell scaled by 1 + 1/8 for the head's layer).
    c = scfg(quant="bf16", gpu_memory_utilization=0.9, prefix_cache=False, spec_decode=f"mtp:{k}")
    expected = {1: 3_231_055_872, 3: 4_982_439_936}[k]
    assert estimate(hw_rtx4090, model, c, mm).kv_cache == 16_384 * 36_864 + 17 * (1 + draft) * STATE == expected
    # Radix cache on: 5 x 16 + 1 slots + draft x 17 intermediate states, inside the mamba share 0.9 / 1.9, which
    # outweighs the KV's 603,979,776 x 1.9: 115 x 51,511,296 x 19 / 9 = 12,505,797,973.3 (k = 1),
    # 149 x 51,511,296 x 19 / 9 = 16,203,164,330.7 (k = 3).
    c = c.model_copy(update={"prefix_cache": None})
    slots = 81 + draft * 17
    assert slots == {1: 115, 3: 149}[k]
    assert abs(estimate(hw_rtx4090, model, c, mm).kv_cache - slots * STATE * 19 / 9) <= 1


def test_vllm_counts_the_head_for_a_local_directory_only(tmp_path, hw_rtx4090, tmp_home):
    mm = VllmBackend().memory_model(hw_rtx4090)
    hub, local = with_head(), with_head(hf_id=str(tmp_path))
    c = vcfg(spec_decode="mtp:1")
    # From the Hub the parameter count is the safetensors total, which holds the head already; a local directory's
    # is estimated from its config, so the head's 486,581,248 B are added to the MTP setting alone.
    assert estimate(hw_rtx4090, hub, c, mm).weights == WEIGHTS["fp8"]
    assert estimate(hw_rtx4090, local, c, mm).weights == WEIGHTS["fp8"] + HEAD_BYTES
    assert estimate(hw_rtx4090, local, vcfg(), mm).weights == WEIGHTS["fp8"]
    assert estimate(hw_rtx4090, local, vcfg(spec_decode="ngram_gpu:4"), mm).weights == WEIGHTS["fp8"]


# ---------------------------------------------------------------- a setting that does not fit


def test_vllm_bf16_on_24_gb_tries_mtp_at_the_largest_batch_that_fits(tmp_path, tmp_home):
    card = golden_hw("rtx4090", "vllm", "0.33.0")
    pm, mm = with_head(), VllmBackend().memory_model(golden_hw("rtx4090", "vllm", "0.33.0"))
    leader = vcfg(quant="bf16", batch=16)  # vLLM's smallest grid batch
    at = {b: estimate(card, pm, leader.model_copy(update={"batch": b, "spec_decode": "mtp:1"}), mm).feasible
          for b in (8, 11, 12, 16)}
    assert estimate(card, pm, leader, mm).feasible and at == {8: True, 11: True, 12: False, 16: False}
    with installed("vllm", "0.33.0", tmp_path):
        be = VllmBackend()
        assert not any(s.startswith("mtp") for s in specs(be.spec_variants(leader, pm, card, True)))
        got = [(v.spec_decode, v.batch) for v in be.spec_variants(leader, pm, card, True, min_batch=8)
               if v.spec_decode.startswith("mtp")]
        assert got == [("mtp:1", 11)]  # mtp:3 misses at 8 as well
        notes = be.spec_notes(leader, pm, card, True, min_batch=8)
        assert notes[:2] == [
            "vllm mtp:1 does not fit at batch 16; tried at batch 11, the largest that fits and covers the workload's "
            "8 users",
            "vllm mtp:3 not tried: the planner does not fit it at batch 16 or at any batch down to the workload's 8 "
            "users"]
        assert specs(be.spec_variants(leader, pm, card, True, min_batch=32)) == specs(
            be.spec_variants(leader, pm, card, True))  # a busier workload: no batch covers it, nothing is lowered
        assert specs(be.spec_variants(leader, pm, card, False, min_batch=8)) == specs(
            be.spec_variants(leader, pm, card))  # without --speculative all nothing changes


def test_sglang_on_24_gb_lowers_the_batch_too(tmp_path, tmp_home, hw_rtx4090):
    with installed("sglang", "0.5.19", tmp_path):
        be, pm = SglangBackend(), with_head("sglang")
        assert not any(s.startswith("mtp") for s in specs(be.spec_variants(scfg(), pm, hw_rtx4090, True)))
        lowered = {v.spec_decode: v.batch for v in be.spec_variants(scfg(), pm, hw_rtx4090, True, min_batch=1)
                   if v.spec_decode.startswith("mtp")}
        assert set(lowered) == {"mtp:1", "mtp:3"} and all(1 <= b < 16 for b in lowered.values())
        for spec, b in lowered.items():
            v = scfg(spec_decode=spec, batch=b, gpu_memory_utilization=0.71, gmu_before_spec=0.88)
            mm = be.memory_model(hw_rtx4090)
            assert estimate(hw_rtx4090, pm, v, mm).feasible
            assert not estimate(hw_rtx4090, pm, v.model_copy(update={"batch": b + 1}), mm).feasible


def test_the_speculation_stage_passes_the_workload_and_collects_notes(tmp_path, tmp_home):
    from rigtune.backends import registry
    from rigtune.calibrate.workload import get_workload
    from rigtune.pipeline import PlanResult, SearchOptions, spec_stage_fn

    card, pm = golden_hw("rtx4090", "vllm", "0.33.0"), with_head()
    plan = PlanResult(hw=card, spec=pm.spec, candidates=["vllm"], prepared={"vllm": pm})
    notes: list = []
    with installed("vllm", "0.33.0", tmp_path):
        fn = spec_stage_fn(card, registry(), plan, SearchOptions(speculative_all=True), get_workload("chat"), notes)
        leader = vcfg(quant="bf16", batch=16)
        assert ("mtp:1", 11) in [(v.spec_decode, v.batch) for v in fn(leader)]  # chat's busiest level is 8 users
        fn(leader)  # asked again (another engine's turn, a budgeted rerun): said once
        assert len(notes) == 2 and notes[0].startswith("vllm mtp:1 does not fit at batch 16; tried at batch 11")
        quiet: list = []
        fn = spec_stage_fn(card, registry(), plan, SearchOptions(), get_workload("chat"), quiet)
        assert not any(v.spec_decode.startswith("mtp") for v in fn(leader)) and quiet == []


# ---------------------------------------------------------------- finding the head


def _facts(head=None, notes=()):
    return introspect.ModelFacts(mtp=head, notes=list(notes))


HEAD = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=HEAD_BYTES,
                          shared_bytes=SHARED_BYTES)


def test_the_head_is_recorded_from_the_introspector():
    pm = qwen35(weights=dict(WEIGHTS))
    assert speculative.record_mtp_head(pm, "vllm", inspect=lambda spec: _facts(HEAD)) == []
    assert (pm.spec_repos, pm.spec_bytes) == ({"mtp": "Qwen/Qwen3.5-9B"},
                                              {"mtp": HEAD_BYTES, "mtp_shared": SHARED_BYTES})
    pm = qwen35(weights=dict(WEIGHTS))
    sizeless = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=None)
    assert speculative.record_mtp_head(pm, "vllm", inspect=lambda spec: _facts(sizeless)) == []
    assert (pm.spec_repos, pm.spec_bytes) == ({"mtp": "Qwen/Qwen3.5-9B"}, {})
    pm = qwen35(weights=dict(WEIGHTS))
    notes = speculative.record_mtp_head(pm, "sglang", inspect=lambda spec: _facts(notes=["a", "b"]))
    assert notes == ["sglang: no MTP head tried: a; b"] and pm.spec_repos == {}
    assert speculative.record_mtp_head(pm, "vllm", inspect=lambda spec: _facts()) == []  # nothing declared

    def broken(spec):
        raise RuntimeError("boom")

    assert speculative.record_mtp_head(pm, "vllm", inspect=broken) == ["vllm: no MTP head tried (boom)"]


def test_the_head_is_looked_up_only_under_all_and_only_for_an_architecture_that_has_one(
        monkeypatch, tmp_path, tmp_home, hw_rtx4090):
    from rigtune.backends import registry
    from rigtune.pipeline import prepare_and_plan

    asked = []

    def inspect(spec, features=introspect.FEATURES, **kw):
        asked.append((spec.hf_id, tuple(features)))
        return _facts(HEAD)

    arch = qwen35().arch
    monkeypatch.setattr(introspect, "inspect_model", inspect)
    monkeypatch.setattr(vl, "load_arch", lambda spec, token=None: arch)
    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: None)
    monkeypatch.setattr(vl, "hf_weight_options", lambda *a, **k: (dict(WEIGHTS), {}))
    monkeypatch.setattr(speculative, "find_eagle3_head", lambda *a, **k: pytest.fail("no EAGLE3 head is listed"))
    spec = ModelSpec(hf_id="Qwen/Qwen3.5-9B")
    with installed("vllm", "0.33.0", tmp_path):
        assert prepare_and_plan(hw_rtx4090, spec, ["vllm"], registry()).prepared["vllm"].spec_repos == {}
        assert asked == []  # --speculative on: no lookup
        plan = prepare_and_plan(hw_rtx4090, spec, ["vllm"], registry(), speculative_all=True)
        assert plan.prepared["vllm"].spec_repos == {"mtp": "Qwen/Qwen3.5-9B"} and plan.notes == []
        assert plan.prepared["vllm"].spec_bytes == {"mtp": HEAD_BYTES, "mtp_shared": SHARED_BYTES}
        assert asked == [("Qwen/Qwen3.5-9B", ("mtp",))]  # the head alone: config and safetensors headers
        dense = qwen25().arch
        monkeypatch.setattr(vl, "load_arch", lambda spec, token=None: dense)
        monkeypatch.setattr(vl, "_own_head", lambda model, revision: own_head.Verdict(False, "no MTP draft"))
        plan = prepare_and_plan(hw_rtx4090, ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), ["vllm"], registry(),
                                speculative_all=True)
        # The checkpoint is read (this stub says it holds a head); vLLM says it has no draft for it: none is kept.
        assert plan.prepared["vllm"].spec_repos == {} and asked[-1] == ("Qwen/Qwen2.5-7B-Instruct", ("mtp",))


def test_sglang_notes_a_head_whose_size_is_unknown(monkeypatch, tmp_path, tmp_home):
    sizeless = introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=None)
    monkeypatch.setattr(introspect, "inspect_model", lambda spec, **kw: _facts(sizeless))
    pm = qwen35("sglang", dict(WEIGHTS))  # no EAGLE3 head or draft model is listed for it: nothing else is asked
    with installed("sglang", "0.5.23", tmp_path):
        notes = SglangBackend().find_second_models(pm)
    assert notes == ["sglang: MTP head of Qwen/Qwen3.5-9B not tried, its size could not be read"]
    monkeypatch.setattr(introspect, "inspect_model", lambda spec, **kw: _facts(notes=["Qwen/Qwen3.5-9B: offline"]))
    pm = qwen35("sglang", dict(WEIGHTS))
    with installed("sglang", "0.5.23", tmp_path):
        assert SglangBackend().find_second_models(pm) == ["sglang: no MTP head tried: Qwen/Qwen3.5-9B: offline"]


def test_the_config_declares_the_layers_the_planner_counts():
    assert qwen35().arch.mtp_layers == 1
    assert qwen25().arch.mtp_layers == 0
