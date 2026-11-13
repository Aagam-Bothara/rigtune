"""The planner's budget for a KV-cache tier: host RAM, disk, and the working set in the GPU pool's units."""

from __future__ import annotations

from types import SimpleNamespace

import rigtune.memory as memmod
from rigtune import kvtier
from rigtune.backends import get_backend
from rigtune.memory import (
    HOST_TIER_RESERVE,
    disk_tier_budget,
    host_tier_budget,
    safety_margin,
    working_set_bytes,
)
from rigtune.models import ArchInfo, Config, GiB, ModelSpec, PreparedModel
from tests.conftest import make_hw


def test_host_budget_is_free_ram_less_margin_and_reserve():
    hw = make_hw("a100")  # 48 GiB free
    free = hw.cpu.ram_free_bytes
    assert host_tier_budget(hw) == int(0.95 * free) - safety_margin(free) - HOST_TIER_RESERVE
    assert host_tier_budget(hw, reserve=10 * GiB) == host_tier_budget(hw) - 2 * GiB
    small = hw.model_copy(update={"cpu": hw.cpu.model_copy(update={"ram_free_bytes": 6 * GiB})})
    assert host_tier_budget(small) == 0  # never negative


def test_disk_budget_reads_the_filesystem_the_tier_would_be_created_on(tmp_path, monkeypatch):
    seen = []

    def usage(p):
        seen.append(p)
        return SimpleNamespace(total=1000 * GiB, used=600 * GiB, free=400 * GiB)

    monkeypatch.setattr(memmod.shutil, "disk_usage", usage)
    budget = disk_tier_budget(tmp_path / "not" / "yet")
    assert budget == int(0.95 * 400 * GiB) - safety_margin(400 * GiB)
    assert seen == [tmp_path]  # the nearest directory that exists

    def broken(p):
        raise OSError("gone")

    monkeypatch.setattr(memmod.shutil, "disk_usage", broken)
    assert disk_tier_budget(tmp_path) == 0


def _dense() -> PreparedModel:
    arch = ArchInfo(architecture="LlamaForCausalLM", num_layers=32, hidden_size=4096, num_attention_heads=32,
                    num_kv_heads=8, head_dim=128, max_position_embeddings=131072)
    return PreparedModel(spec=ModelSpec(hf_id="meta-llama/Llama-3.1-8B-Instruct"), backend="vllm", arch=arch,
                         weights_bytes={"bf16": 16 * GiB})


def _gemma_like() -> PreparedModel:
    arch = ArchInfo(architecture="Gemma3ForCausalLM", num_layers=48, hidden_size=3840, num_attention_heads=16,
                    num_kv_heads=8, head_dim=256, max_position_embeddings=131072, full_attention_layers=8,
                    sliding_layers=40, sliding_window=1024)
    return PreparedModel(spec=ModelSpec(hf_id="google/gemma-3-12b-it"), backend="vllm", arch=arch,
                         weights_bytes={"bf16": 24 * GiB})


def test_working_set_of_a_dense_model_is_bytes_per_token():
    model, mm = _dense(), get_backend("vllm").memory_model(make_hw("rtx4090"))
    cfg = Config(backend="vllm", quant="bf16", ctx=16384, batch=64)
    per_token = model.arch.kv_bytes_per_token(2)  # 128 KiB for Llama 3.1 8B in bf16
    assert working_set_bytes(model, cfg, 380_000, 64, mm) == per_token * 380_000
    fp8 = cfg.model_copy(update={"kv_dtype": "fp8"})
    assert working_set_bytes(model, fp8, 380_000, 64, mm) == per_token // 2 * 380_000


def test_working_set_of_a_hybrid_model_follows_the_hybrid_planner():
    """Written out by hand from vLLM 0.29's rules (backends/vllm.vllm_hybrid_kv), not with the code under
    test: 64 conversations of 6000 tokens on 8 full and 40 sliding layers (window 1024), 8 KV heads of 256."""
    model, mm = _gemma_like(), get_backend("vllm").memory_model(make_hw("rtx4090"))
    cfg = Config(backend="vllm", quant="bf16", ctx=16384, batch=16)
    full_per_token = 2 * 8 * 8 * 256 * 2  # K and V x 8 full layers x 8 heads x 256 x 2 B = 65,536 B
    sliding_per_token = 2 * 40 * 8 * 256 * 2  # 327,680 B
    full_tokens = 64 * 6000  # 6000 is a whole number of 16-token blocks
    # A window that starts mid-block spans one block more: 1024 + 16 per conversation, plus the tokens in
    # flight once (2 x the 2048-token default prefill budget on a 24 GB card).
    sliding_tokens = 64 * (1024 + 16) + 2 * 2048
    want = full_per_token * full_tokens + sliding_per_token * sliding_tokens
    assert want == 25_165_824_000 + 23_152_558_080  # 45.0 GiB; every layer full: 140.6 GiB
    assert working_set_bytes(model, cfg, 64 * 6000, 64, mm) == want
    # The start-up floor (one request at max_model_len) does not inflate a small working set.
    hk = mm.hybrid_for(cfg.model_copy(update={"batch": 1}), model.arch)
    assert working_set_bytes(model, cfg, 100, 1, mm) < hk.min_bytes


def _qwen35_like() -> PreparedModel:
    arch = ArchInfo(architecture="Qwen3_5ForCausalLM", num_layers=32, hidden_size=4096,
                    num_attention_heads=16, num_kv_heads=4, head_dim=256, max_position_embeddings=262144,
                    full_attention_layers=8, linear_layers=24, linear_state_bytes=51_511_296,
                    linear_conv_step_bytes=24 * 8192)
    return PreparedModel(spec=ModelSpec(hf_id="Qwen/Qwen3.5-9B"), backend="vllm", arch=arch,
                         weights_bytes={"bf16": 18 * GiB})


def test_speculation_does_not_count_states_a_cached_conversation_does_not_hold():
    model, mm = _qwen35_like(), get_backend("vllm").memory_model(make_hw("rtx4090"))
    plain = Config(backend="vllm", quant="bf16", ctx=16384, batch=16)
    drafted = plain.model_copy(update={"spec_decode": "suffix:24"})
    want = working_set_bytes(model, plain, 64 * 6000, 64, mm)
    assert working_set_bytes(model, drafted, 64 * 6000, 64, mm) == want
    # While a request runs, its speculative tokens hold states of their own: that is the GPU plan, not this.
    assert mm.hybrid_for(drafted, model.arch).state_slots == 16 * 25


def test_sglangs_pool_split_does_not_size_what_is_cached():
    """SGLang sizes its mamba pool for running requests (five slots each for Qwen3.5) inside a share of its
    budget; a cached conversation holds one state. By hand: 8 full layers x 4 heads x 256 x 2 (K, V) x 2 B =
    32,768 B a token over 384,000 tokens, plus 64 states of 51,511,296 B."""
    model = _qwen35_like().model_copy(update={"backend": "sglang"})
    mm = get_backend("sglang").memory_model(make_hw("rtx4090"))
    cfg = Config(backend="sglang", quant="bf16", ctx=16384, batch=16)
    assert working_set_bytes(model, cfg, 64 * 6000, 64, mm) == 32_768 * 384_000 + 64 * 51_511_296


MOUNTS ="""overlay / overlay rw,relatime 0 0
mfs#eu-ro-1.runpod.net:9421 /workspace fuse.mfs rw,nosuid,nodev 0 0
server:/export /mnt/shared nfs4 rw,vers=4.2 0 0
//filer/share /mnt/smb cifs rw 0 0
/dev/nvme0n1p1 /root ext4 rw,relatime 0 0
/dev/nvme1n1 /mnt/scratch\\040disk xfs rw 0 0
"""


def test_a_disk_tier_is_never_written_to_a_network_filesystem(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(MOUNTS)
    m = str(mounts)
    assert "fuse.mfs filesystem" in kvtier.refusal("/workspace/.rigtune/kv-tier", resolve=False, mounts=m)
    assert "nfs4 filesystem" in kvtier.refusal("/mnt/shared/home/.rigtune/kv-tier", resolve=False, mounts=m)
    assert "cifs" in kvtier.refusal("/mnt/smb/x", resolve=False, mounts=m)
    assert kvtier.refusal("/root/.rigtune/kv-tier", resolve=False, mounts=m) is None  # local ext4
    assert kvtier.refusal("/mnt/scratch disk/kv-tier", resolve=False, mounts=m) is None  # \040 is a space
    assert kvtier.refusal("/srv/kv-tier", resolve=False, mounts=m) is None  # the overlay root


def test_without_a_mount_table_the_known_volumes_are_refused(monkeypatch, tmp_path):
    missing = str(tmp_path / "no-mounts")
    assert kvtier.refusal(tmp_path, mounts=missing) is None
    assert "network volume /workspace" in kvtier.refusal("/workspace/.rigtune/kv-tier", resolve=False,
                                                         mounts=missing)
    assert kvtier.refusal("/workspace", resolve=False, mounts=missing)
    assert kvtier.refusal("/workspaces/x", resolve=False, mounts=missing) is None
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    assert kvtier.tier_root() == tmp_path / "home" / "kv-tier"
