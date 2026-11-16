import json

import pytest

from rigtune import pairing as p
from rigtune.pipeline import SearchOptions
from tests.test_pairing import pair, target


def kept():
    a, b = target(), pair()
    return p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 4096))


@pytest.mark.parametrize("offline_env", [False, True])
def test_kept_record_is_used_and_note_says_not_rechecked(prepared_vllm, monkeypatch, offline_env):
    kept()
    pm = prepared_vllm
    pm.spec.hf_id = "org/target"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1" if offline_env else "0")
    def unavailable(repo, source="hub", **kw):
        assert source == "cache"
        raise p.HubLookupError("offline")
    monkeypatch.setattr(p, "read_files", unavailable)
    notes = p.discover(pm, "vllm", draft="org/draft", hub="on" if offline_env else "off")
    assert pm.spec_revisions["org/draft"] == "b" * 40
    assert any("not rechecked" in n and "recalibrate" in n for n in notes)


def test_hub_off_makes_no_request(prepared_vllm, monkeypatch):
    import huggingface_hub
    def forbidden(*a, **kw):
        raise AssertionError("network")
    def cache_only(*a, **kw):
        assert kw["local_files_only"] is True
        raise FileNotFoundError()
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cache_only)
    notes = p.discover(prepared_vllm, "vllm", draft="org/draft", hub="off")
    assert any("not asked" in n for n in notes)
    assert not prepared_vllm.spec_pairs


def test_kept_refusal_rechecks_the_new_context(prepared_vllm, monkeypatch):
    a, b = target(), pair()
    p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 16384))
    prepared_vllm.spec.hf_id = "org/target"
    monkeypatch.setattr(p, "read_files", lambda *a, **kw: (_ for _ in ()).throw(p.HubLookupError("offline")))
    p.discover(prepared_vllm, "vllm", 4096, draft="org/draft", hub="off")
    assert prepared_vllm.spec_pairs  # record's files are evidence, old verdict isn't universal


def test_refuted_record_blocks_only_that_engine_release(prepared_vllm, monkeypatch):
    path = kept()
    prepared_vllm.spec_pairs = [{"repo": "org/draft", "record": path.name}]
    p.refute(prepared_vllm, "draft:org/draft:4", "vllm", "0.33.0")
    monkeypatch.setattr(p, "read_files", lambda *a, **kw: (_ for _ in ()).throw(p.HubLookupError("offline")))
    for engine, version, offered in [("vllm", "0.33.0", False), ("vllm", "0.33.1", True), ("sglang", "0.5.23", True)]:
        pm = prepared_vllm.model_copy(deep=True)
        pm.spec.hf_id = "org/target"
        pm.spec_pairs = []
        p.discover(pm, engine, draft="org/draft", hub="off", version=version)
        assert bool(pm.spec_pairs) == offered


def test_local_draft_checked_from_its_directory(tmp_path, prepared_vllm, monkeypatch):
    from tests.test_pairing_read import write_weights
    for name, count in [("target", 80), ("draft", 10)]:
        root = tmp_path / "private-person" / name
        root.mkdir(parents=True)
        config = pair().config
        (root / "config.json").write_text(json.dumps(config))
        write_weights(root / "model.safetensors", (count,))
        (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}))
    pm = prepared_vllm
    pm.spec.hf_id = str(tmp_path / "private-person" / "target")
    repo = str(tmp_path / "private-person" / "draft")
    notes = p.discover(pm, "vllm", draft=repo, hub="off")
    assert len(pm.spec_paths) == 1
    alias = next(iter(pm.spec_paths))
    assert alias.startswith("local/draft-")
    assert "private-person" not in json.dumps(notes + pm.spec_pairs)
    assert "private-person" not in json.dumps(SearchOptions(draft=repo).key())
    copy = pm.model_copy(deep=True)
    copy.spec_paths = {}
    assert p.restore_local_paths(copy, repo)
    assert copy.spec_paths[alias] == repo
    assert "private-person" not in p.redact_paths(json.dumps({"model": repo}), pm.spec_paths)


def test_verdict_and_notes_scrub_a_local_draft(prepared_vllm, hw_a100):
    from rigtune.backends.vllm import VllmBackend
    from rigtune.models import Config
    from rigtune.pipeline import _profile_for
    pm = prepared_vllm
    pm.spec_paths = {"local/draft-12345678": "C:/private-person/draft"}
    pm.spec_revisions = {"local/draft-12345678": "a" * 64}
    cfg = Config(backend="vllm", quant="bf16", spec_decode="draft:local/draft-12345678:4")
    profile = _profile_for(hw_a100, pm.spec, "balanced", VllmBackend(), cfg, pm)
    assert "private-person" not in profile.model_dump_json()


def test_quantized_or_unrecognised_draft_is_not_certified():
    b = pair()
    b.config["quantization_config"] = {"quant_method": "gptq"}
    assert p.check_pair(target(), b, "draft", 4096).status != "ok"
    b.config = {**b.config, "quantization_config": None, "architectures": ["MysteryMTP"], "model_type": "new_family"}
    assert p.check_pair(target(), b, "draft", 4096).status != "ok"
