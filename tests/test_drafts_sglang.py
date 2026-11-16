import pytest

from rigtune import pairing as p, speculative as sp
from rigtune.backends import sglang as sg
from rigtune.models import Config
from tests.test_pairing import pair, target


@pytest.fixture
def setup_pair(monkeypatch, prepared_vllm):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(sg, "feature", lambda name: {"draft", "ngram"} if name == "spec_methods" else None)
    pm = prepared_vllm.model_copy(deep=True)
    pm.backend = "sglang"
    pm.spec.hf_id = "org/target"
    pm.drafts_mode = "verified"
    files = {"org/target": target(), "org/draft": pair(size_bytes=2_500_000_000)}
    monkeypatch.setattr(p, "read_files", lambda repo, *a, **kw: files[repo].model_copy(deep=True))
    return sg.SglangBackend(), pm, files


def variants(be, pm, hw, quant="bf16"):
    cfg = Config(backend="sglang", quant=quant, ctx=4096, batch=16, gpu_memory_utilization=.9)
    return [c for c in be.spec_variants(cfg, pm, hw, True) if c.spec_decode.startswith("draft:")]


def test_flags_carry_draft_revision(setup_pair, hw_a100):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    cfg = variants(be, pm, hw_a100)[0]
    args = be.launch_spec(cfg, pm, 8000).args
    assert args[args.index("--speculative-draft-model-revision") + 1] == "b" * 40
    assert args[args.index("--speculative-algorithm") + 1] == "STANDALONE"


def test_quantized_target_gets_explicit_draft_quantization(setup_pair, hw_a100):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    args = be.launch_spec(variants(be, pm, hw_a100, "fp8")[0], pm, 8000).args
    assert args[args.index("--speculative-draft-model-quantization") + 1] == "unquant"


def test_all_plus_table_mode_flags_identical():
    assert sp.sglang_config("draft:org/draft:3") == [
        "--speculative-algorithm", "STANDALONE", "--speculative-draft-model-path", "org/draft",
        "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]


def test_ids_differ_pair_never_reaches_a_launch(setup_pair, hw_a100):
    be, pm, files = setup_pair
    files["org/target"].token_ids = {"a": 0}
    files["org/draft"].token_ids = {"b": 0}
    files["org/draft"].tokenizer_sha256 = "c" * 64
    notes = be.find_pairs(pm, draft="org/draft")
    assert any("ids_differ" in n for n in notes)
    assert not variants(be, pm, hw_a100)


def test_fraction_leaves_the_draft_room(setup_pair, hw_rtx4090):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    cfg = variants(be, pm, hw_rtx4090)[0]
    assert cfg.gpu_memory_utilization < .85 and cfg.gmu_before_spec == .9


def test_probe_without_STANDALONE_offers_none(setup_pair, hw_a100, monkeypatch):
    be, pm, _ = setup_pair
    monkeypatch.setattr(sg, "feature", lambda name: set())
    be.find_pairs(pm, draft="org/draft")
    assert not variants(be, pm, hw_a100)
