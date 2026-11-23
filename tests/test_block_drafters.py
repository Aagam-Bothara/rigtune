"""DFlash and DSpark drafters on vLLM: what each drafts, how it is launched, planned, offered and refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rigtune.backends.vllm as vl
from rigtune import headsearch, speculative
from rigtune.backends.vllm import AMPERE_FP8_KV, INT8_KV, VllmBackend
from rigtune.memory import estimate
from rigtune.models import ArchInfo, Config, ModelSpec, PreparedModel
from tests.test_hybrid_memory import qwen35

FIXTURES = Path(__file__).parent / "fixtures" / "heads"
DFLASH, DSPARK = "z-lab/Qwen3-8B-DFlash-b16", "deepseek-ai/dspark_qwen3_8b_block7"


def _config(repo: str) -> dict:
    return json.loads((FIXTURES / (repo.replace("/", "__") + ".json")).read_text(encoding="utf-8"))["config"]


def _qwen3_8b() -> PreparedModel:
    arch = ArchInfo(architecture="Qwen3ForCausalLM", num_layers=36, hidden_size=4096, num_attention_heads=32,
                    num_kv_heads=8, head_dim=128, vocab_size=151936, max_position_embeddings=40960,
                    num_params=8_190_735_360)
    return PreparedModel(spec=ModelSpec(hf_id="Qwen/Qwen3-8B"), backend="vllm", arch=arch, hf_path="Qwen/Qwen3-8B",
                         weights_bytes={"bf16": 16_381_470_720, "fp8": 8_190_735_360})


@pytest.mark.parametrize("repo, kind, tokens, elements", [
    (DFLASH, "dflash", (15, 16), 10240),  # block 16: an anchor and 15 masks; 5 layers x 2 x 8 x 128
    ("z-lab/Qwen3.5-9B-DFlash", "dflash", (15, 16), 12288),  # block_size under dflash_config
    ("RedHatAI/Qwen3-30B-A3B-speculator.dflash", "dflash", (7, 8), 5120),  # vLLM's recipes: 7, "block_size 8 minus one"
    (DSPARK, "dspark", (7, 7), 10240),  # samples from its anchor: the whole block, as vLLM's own test runs it
    ("RadixArk/Qwen3.8-27B-DSpark", "dspark", (7, 7), 10240),
    # Gemma 4's full layers hold one 512-wide KV head (global_head_dim, attention_k_eq_v), not 8 x 256.
    ("deepseek-ai/dspark_gemma4_12b_block7", "dspark", (7, 7), 5 * 2 * 1 * 512),
    ("z-lab/gemma4-12B-it-DFlash", "dflash", (15, 16), 10240),  # a Qwen3-shaped drafter for Gemma 4
])
def test_what_each_drafter_drafts_and_holds_from_its_config(repo, kind, tokens, elements):
    cfg = _config(repo)
    assert speculative.drafter_tokens(cfg, kind) == tokens
    assert speculative.drafter_kv_elements(cfg) == elements


def test_a_drafter_without_a_block_size_drafts_an_unknown_number():
    assert speculative.drafter_tokens({"architectures": ["DFlashDraftModel"]}, "dflash") is None
    assert speculative.drafter_tokens(_config(DFLASH), "eagle3") is None


def test_the_launch_names_the_method_on_both_engines_at_one_k():
    spec = speculative.block_drafter("dflash", DFLASH, 15)
    assert speculative.parse(spec) == ("dflash", DFLASH, 15)
    assert speculative.vllm_config(spec) == {"method": "dflash", "model": DFLASH, "num_speculative_tokens": 15}
    assert speculative.vllm_config(f"dspark:{DSPARK}:7")["method"] == "dspark"
    # SGLang counts the block: the last accepted token and the 15 drafts (tests/test_sglang_block_drafters.py).
    assert speculative.sglang_config(spec)[:6] == ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path",
                                                   DFLASH, "--speculative-num-draft-tokens", "16"]
    assert speculative.vllm_supports_block_drafter("dflash", "0.33.0")
    assert not speculative.vllm_supports_block_drafter("dflash", "0.29.0")  # not checked before 0.33


@pytest.fixture
def found_on_the_hub(monkeypatch):
    """A kept search for Qwen3-8B holding z-lab's DFlash drafter and DeepSeek's DSpark drafter, read and sized."""
    def kept(target_id, revision, seeds=(), **kw):
        cands = [headsearch.Candidate(repo=r, downloads=1, config=_config(r), files={"model.safetensors": size},
                                      size_bytes=size, sha=sha)
                 for r, size, sha in ((DFLASH, 2_100_000_000, "d" * 40), (DSPARK, 2_200_000_000, "e" * 40))]
        return headsearch.Search(target=target_id, revision=revision, day="test", aliases=[target_id],
                                 candidates=cands), []

    monkeypatch.setattr(headsearch, "kept_or_search", kept)
    monkeypatch.setattr(speculative, "vllm_supports_block_drafter", lambda kind, version=None: True)
    monkeypatch.setattr(speculative, "vllm_supports_eagle3", lambda version=None: False)
    monkeypatch.setattr(VllmBackend, "_own_head_notes", lambda self, prepared: [])


def test_vllm_records_the_drafters_it_loads_with_what_plans_them(found_on_the_hub):
    pm = _qwen3_8b()
    notes = VllmBackend().find_second_models(pm)
    assert pm.spec_repos == {"dflash": DFLASH, "dspark": DSPARK}
    assert pm.spec_facts[DFLASH] == {"k": 15, "lookahead": 16, "kv_elements": 10240, "needs_v2": False}
    assert pm.spec_revisions[DFLASH] == "d" * 40  # the commit whose config was checked
    assert pm.spec_bytes[DSPARK] == 2_200_000_000
    assert any("drafting 15 tokens a step" in n for n in notes)
    assert "spec_facts" not in pm.model_dump()  # not saved in profiles


def test_a_drafter_this_release_refused_is_not_recorded_again(found_on_the_hub, hw_a100):
    """C8: vLLM refused z-lab's drafter at its checked commit in an earlier calibration; this one, on the same
    release, records DSpark alone and says why, and a calibration that does not know the release is unchanged."""
    release = f"vllm {hw_a100.backend_version('vllm')}"
    headsearch.refuse(DFLASH, "d" * 40, release, "target_layer_ids past the target's layers", "Qwen/Qwen3-8B",
                      today="2026-11-22")
    pm = _qwen3_8b()
    notes = VllmBackend().find_second_models(pm, hw_a100)
    assert pm.spec_repos == {"dspark": DSPARK}
    assert f"vllm: dflash drafter {DFLASH} not offered: {release} refused to start with it on 2026-11-22 " \
           "(target_layer_ids past the target's layers)" in notes
    unknown = _qwen3_8b()
    VllmBackend().find_second_models(unknown)
    assert unknown.spec_repos == {"dflash": DFLASH, "dspark": DSPARK}


def test_the_planner_counts_a_drafter_s_weights_and_layers(found_on_the_hub, hw_a100):
    pm = _qwen3_8b()
    VllmBackend().find_second_models(pm)
    mm = VllmBackend().memory_model(hw_a100)
    base = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    plain, drafting = estimate(hw_a100, pm, base, mm), estimate(
        hw_a100, pm, base.model_copy(update={"spec_decode": f"dflash:{DFLASH}:15"}), mm)
    assert drafting.weights - plain.weights == 2_100_000_000
    # Five drafter layers of the target's shape beside its 36: 41/36 of the KV.
    assert drafting.kv_cache == pytest.approx(plain.kv_cache * 41 / 36, rel=1e-6)


def test_on_a_hybrid_model_a_drafter_holds_a_state_per_token_it_drafts():
    pm = qwen35()
    pm.spec_repos["dflash"] = "z-lab/Qwen3.5-9B-DFlash"
    pm.spec_bytes["z-lab/Qwen3.5-9B-DFlash"] = 2_583_816_465
    pm.spec_facts["z-lab/Qwen3.5-9B-DFlash"] = {"k": 15, "lookahead": 16, "kv_elements": 12288}
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9,
                 spec_decode="dflash:z-lab/Qwen3.5-9B-DFlash:15")
    assert vl.vllm_drafter_layers(cfg, pm) == 6  # 12,288 elements over the target's 2 x 4 x 256 per full layer
    plan = vl.vllm_hybrid_kv(cfg, pm.arch)
    assert plan.state_slots == 16 * 16  # 1 + k states per running request (mamba/abstract.py)
    from rigtune.backends.vllm import VllmBackend as Backend
    from tests.conftest import make_hw

    hw = make_hw("a100")
    mm = Backend().memory_model(hw)
    plain = estimate(hw, pm, cfg.model_copy(update={"spec_decode": None}), mm)
    drafting = estimate(hw, pm, cfg, mm)
    assert drafting.weights - plain.weights == 2_583_816_465
    assert drafting.kv_cache > plain.kv_cache  # its layers, and fifteen more recurrent states a request


def test_a_drafter_keeps_its_batch_with_the_token_budget_vllm_needs_for_it(found_on_the_hub, hw_a100):
    """vLLM runs no more than max_num_batched_tokens // lookahead requests with a DFlash or DSpark drafter: an A100's
    default 2,048 would hold DFlash (16 tokens a request a step) to 128 of 256, so the budget is raised to 4,096."""
    pm = _qwen3_8b()
    be = VllmBackend()
    be.find_second_models(pm)
    leader = Config(backend="vllm", quant="bf16", ctx=4096, batch=256, gpu_memory_utilization=0.9)
    variants = {v.spec_decode: v for v in be.spec_variants(leader, pm, hw_a100, all_methods=True, min_batch=8)}
    dflash, dspark = variants[f"dflash:{DFLASH}:15"], variants[f"dspark:{DSPARK}:7"]
    assert (dflash.batch, dflash.prefill_budget) == (256, 4096)
    assert (dspark.batch, dspark.prefill_budget) == (256, 2048)  # 2,048 // 7 = 292 holds 256; stated all the same
    # ... so a later stage that raises the batch past it is refused, as one that lowers the budget is.
    assert "only 292 of batch 512" in be.config_refusal(dspark.model_copy(update={"batch": 512}), pm)
    notes = be.spec_notes(leader, pm, hw_a100, all_methods=True, min_batch=8)
    assert any("max_num_batched_tokens 4096" in n for n in notes)
    assert not any((v.spec_decode or "").startswith(("dflash", "dspark"))  # --speculative on: never offered
                   for v in be.spec_variants(leader, pm, hw_a100, all_methods=False))
    # A later stage may not lower the budget below what the batch needs.
    assert "only 64 of batch 256" in be.config_refusal(dflash.model_copy(update={"prefill_budget": 1024}), pm)
    args = be.launch_spec(dflash, pm, 8000).args
    payload = json.loads(args[args.index("--speculative-config") + 1])
    assert payload == {"method": "dflash", "model": DFLASH, "num_speculative_tokens": 15, "revision": "d" * 40}
    assert args[args.index("--max-num-batched-tokens") + 1] == "4096"


@pytest.mark.parametrize("batch", [64, 256])  # 2,048, vLLM's default on an A100, stated; and 4,096 for DFlash
def test_a_drafter_s_budget_is_not_carried_into_the_other_settings_once_it_leads(found_on_the_hub, hw_a100, batch):
    """The A100 run of 22 Nov 2026: once DSpark led, the speculation stage offered n-gram, the draft model and the
    EAGLE3 head again at DSpark's stated budget, three trials that measured what the first pass had. Offered from a
    leader drafting with either drafter, every setting is one the first pass offered."""
    pm = _qwen3_8b()
    be = VllmBackend()
    be.find_second_models(pm)
    plain = Config(backend="vllm", quant="bf16", ctx=4096, batch=batch, gpu_memory_utilization=0.9)
    first = {v.key() for v in be.spec_variants(plain, pm, hw_a100, all_methods=True, min_batch=8)}
    drafters = [v for v in be.spec_variants(plain, pm, hw_a100, all_methods=True, min_batch=8)
                if v.spec_decode.startswith(("dflash", "dspark"))]
    assert len(drafters) == 2 and all(d.prefill_budget for d in drafters)
    for leader in drafters:
        again = {v.key() for v in be.spec_variants(leader, pm, hw_a100, all_methods=True, min_batch=8)}
        assert leader.key() not in again and again <= first, again - first
    # A budget the prefill stage chose is the leader's own: the other settings are offered at it.
    chosen = drafters[0].model_copy(update={"prefill_budget": 16384})
    assert {v.prefill_budget for v in be.spec_variants(chosen, pm, hw_a100, all_methods=True, min_batch=8)} == {16384}


def test_what_would_never_start_is_not_offered(found_on_the_hub, hw_a100, monkeypatch):
    pm = _qwen3_8b()
    be = VllmBackend()
    be.find_second_models(pm)
    spec = f"dflash:{DFLASH}:15"
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, spec_decode=spec)
    assert "FlashInfer" in be.known_failure(cfg.model_copy(update={"kv_dtype": AMPERE_FP8_KV}))
    assert "int8" in be.known_failure(cfg.model_copy(update={"kv_dtype": INT8_KV}))
    # The drafter picks its own attention backend, not the target's (dflash/utils.py): Triton on the target is fine.
    assert be.config_refusal(cfg.model_copy(update={"extra": {"attention_backend": "TRITON_ATTN"}}), pm) is None
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert "Model Runner V2" in be.config_refusal(cfg.model_copy(update={"spec_decode": f"dspark:{DSPARK}:7"}), pm)
    assert be.config_refusal(cfg, pm) is None  # z-lab's Qwen3-8B drafter is all full attention: V1 runs it
    pm.spec_facts[DFLASH]["needs_v2"] = True  # as z-lab's Qwen3.5-9B drafter: five sliding layers and one full
    assert "Model Runner V2" in be.config_refusal(cfg, pm)


@pytest.mark.parametrize("repo, needs_v2", [
    (DFLASH, False),  # all full attention
    ("z-lab/Qwen3.5-9B-DFlash", True),  # five sliding layers and one full
    ("RedHatAI/Qwen3-30B-A3B-speculator.dflash", False),
])
def test_which_dflash_drafters_vllm_runs_on_its_v2_runner_only(repo, needs_v2):
    assert speculative.drafter_needs_v2(_config(repo)) is needs_v2
    assert speculative.drafter_needs_v2({"architectures": ["DFlash2DraftModel"]})
    mixed = {"model_type": "speculators", "speculators_config": {},
             "transformer_layer_config": {"layer_types": ["sliding_attention", "full_attention"]}}
    assert speculative.drafter_needs_v2(mixed)  # read where vLLM reads it after converting the format


def test_a_drafter_that_cannot_be_planned_is_not_offered(found_on_the_hub, hw_a100):
    pm = _qwen3_8b()
    be = VllmBackend()
    be.find_second_models(pm)
    pm.spec_facts[DFLASH]["kv_elements"] = None
    leader = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    assert f"dflash:{DFLASH}:15" not in [v.spec_decode for v in be.spec_variants(leader, pm, hw_a100, True, 8)]
    assert any("cannot be planned" in n for n in be.spec_notes(leader, pm, hw_a100, True, 8))
