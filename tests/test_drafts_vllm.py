import json

import pytest

from rigtune import pairing as p, speculative as sp
from rigtune.backends import vllm as vl
from rigtune.backends.facts import EngineFacts
from rigtune.models import Config, GiB
from tests.test_pairing import pair, target


@pytest.fixture
def setup_pair(monkeypatch, prepared_vllm):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(sp, "vllm_supports_draft", lambda: True)
    monkeypatch.setattr(sp, "vllm_supports_suffix", lambda: False)
    monkeypatch.setattr(sp, "vllm_supports_ngram_gpu", lambda: False)
    prepared_vllm.spec.hf_id = "org/target"
    prepared_vllm.drafts_mode = "verified"
    files = {"org/target": target(), "org/draft": pair()}
    monkeypatch.setattr(p, "read_files", lambda repo, *a, **kw: files[repo].model_copy(deep=True))
    return vl.VllmBackend(), prepared_vllm, files


def drafts(be, pm, hw, **kw):
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=.9, **kw)
    return [c for c in be.spec_variants(cfg, pm, hw) if (c.spec_decode or "").startswith("draft:")]


def test_verified_pair_is_offered_with_its_revision(setup_pair, hw_a100):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    cfg = drafts(be, pm, hw_a100)[0]
    args = be.launch_spec(cfg, pm, 8000).args
    payload = json.loads(args[args.index("--speculative-config") + 1])
    assert payload == {"method": "draft_model", "model": "org/draft", "num_speculative_tokens": 4,
                       "revision": "b" * 40}
    assert pm.spec.revision == "b" * 40


def test_refused_pair_is_not_offered_and_noted(setup_pair, hw_a100):
    be, pm, files = setup_pair
    files["org/draft"].config["vocab_size"] = 7
    notes = be.find_pairs(pm, draft="org/draft")
    assert not drafts(be, pm, hw_a100)
    assert any("vocab_differs" in n for n in notes)


def test_unknown_pair_is_not_offered_and_note_says_not_checked(setup_pair, hw_a100):
    be, pm, files = setup_pair
    files["org/draft"].tokenizer_sha256 = None
    notes = be.find_pairs(pm, draft="org/draft")
    assert not drafts(be, pm, hw_a100)
    assert any("not checked" in n for n in notes)


def test_table_mode_config_json_identical():
    assert sp.vllm_config("draft:org/draft:4") == {
        "method": "draft_model", "model": "org/draft", "num_speculative_tokens": 4}


def test_probe_without_draft_model_offers_none(setup_pair, hw_a100, monkeypatch):
    be, pm, _ = setup_pair
    monkeypatch.setattr(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", spec_methods=["ngram"]))
    be.find_pairs(pm, draft="org/draft")
    assert not drafts(be, pm, hw_a100)


def test_draft_weights_enter_the_plan(setup_pair, hw_rtx4090):
    be, pm, files = setup_pair
    files["org/target"].num_params = 40 * GiB
    files["org/draft"].num_params = 10 * GiB
    files["org/draft"].size_bytes = 20 * GiB
    be.find_pairs(pm, draft="org/draft")
    assert not drafts(be, pm, hw_rtx4090)
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=.9)
    assert any("fit" in n for n in be.spec_notes(cfg, pm, hw_rtx4090))


def test_at_most_two_candidates(setup_pair, hw_a100, monkeypatch):
    be, pm, files = setup_pair
    for i in range(4):
        files[f"org/d{i}"] = pair(repo=f"org/d{i}", num_params=1_000_000-i, size_bytes=2_000_000-i*2)
    monkeypatch.setattr(sp, "candidates_for", lambda *a: [("draft", f"org/d{i}") for i in range(4)])
    be.find_pairs(pm)
    assert [sp.parse(c.spec_decode)[1] for c in drafts(be, pm, hw_a100)] == ["org/d3", "org/d2"]


def test_no_fp8_e5m2_pairing_on_ampere_in_verified_mode(setup_pair, hw_a100):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    assert not drafts(be, pm, hw_a100, kv_dtype="fp8_e5m2")


def test_checked_context_does_not_admit_longer_trial(setup_pair, hw_a100):
    be, pm, _ = setup_pair
    be.find_pairs(pm, draft="org/draft")
    cfg = Config(backend="vllm", quant="bf16", ctx=32768, batch=16)
    assert not any(c.spec_decode.startswith("draft:") for c in be.spec_variants(cfg, pm, hw_a100))
