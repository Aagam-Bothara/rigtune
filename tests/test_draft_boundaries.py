"""Verified evidence must survive real planning and reporting boundaries."""
import importlib
import json

import pytest
from typer.testing import CliRunner

from rigtune import pairing as p, pipeline as pl
from rigtune.backends.vllm import VllmBackend
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.test_pairing import pair, target


@pytest.mark.parametrize("kind", ["draft", "assist"])
def test_verified_ampere_pair_is_blocked_in_kv_combine_and_launch(prepared_vllm, hw_a100, monkeypatch, kind):
    pm, be = prepared_vllm, VllmBackend()
    pm.drafts_mode = "verified"
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=8, spec_decode=f"{kind}:org/draft:1")
    plan = pl.PlanResult(hw_a100, pm.spec, ["vllm"], prepared={"vllm": pm})
    monkeypatch.setattr(be, "kv_dtypes", lambda hw: ["fp8_e5m2"])
    bad = cfg.model_copy(update={"kv_dtype": "fp8_e5m2"})
    assert not pl.kv_variants_fn(hw_a100, {"vllm": be}, plan)(cfg)
    assert not pl.combination_fits(hw_a100, {"vllm": be}, plan)(bad)
    with pytest.raises(ValueError, match="fp8_e5m2"):
        be.launch_spec(bad, pm, 8000)
    pm.drafts_mode = "table"
    assert pl.combination_fits(hw_a100, {"vllm": be}, plan)(bad)


@pytest.mark.parametrize("changed", [False, True])
def test_partial_cache_reuses_only_matching_immutable_evidence(prepared_vllm, monkeypatch, tmp_path, changed):
    a, b = target(), pair()
    p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 4096))
    prepared_vllm.spec.hf_id = a.repo
    # An ordinary online check may leave only config.json cached, never the weights.
    configs = {}
    for files in (a, b):
        sha = "c" * 40 if changed and files.repo == b.repo else files.sha
        root = tmp_path / files.repo.replace("/", "--") / "snapshots" / sha
        root.mkdir(parents=True)
        configs[files.repo] = root / "config.json"
        configs[files.repo].write_text(json.dumps(files.config))
    read = p.read_files
    def cached(repo, source, **kw):
        assert source == "cache"
        return read(repo, source, **kw, file_fetcher=lambda *args, **kwargs: configs[repo])
    monkeypatch.setattr(p, "read_files", cached)
    notes = p.discover(prepared_vllm, "vllm", draft=b.repo, hub="off")
    assert bool(prepared_vllm.spec_pairs) is not changed
    if not changed:
        assert any("kept evidence" in note and "not rechecked" in note for note in notes)


def test_moving_target_is_pinned_once_before_all_backend_preparations(no_network, hw_a100, spec, monkeypatch):
    from rigtune.backends.sglang import SglangBackend
    a, b = target(), pair()
    a.repo = spec.hf_id
    reads, prepared_at = [], []
    def read(repo, *args, revision=None, **kw):
        if repo != a.repo:
            return b.model_copy(update={"repo": repo})
        reads.append(revision)
        # main moves after the first request. Only a pinned request keeps the original facts.
        return a.model_copy(update={"sha": revision or ("a" * 40 if len(reads) == 1 else "c" * 40)})
    monkeypatch.setattr(p, "read_files", read)
    reg = {"vllm": VllmBackend(), "sglang": SglangBackend()}
    for name, be in reg.items():
        original = be.prepare
        def prepare(model, hw, quants=None, name=name, original=original):
            prepared_at.append((name, model.revision))
            pm = original(model, hw, quants)
            pm.arch.vocab_size = 128 if model.revision == "a" * 40 else 256
            return pm
        monkeypatch.setattr(be, "prepare", prepare)
    plan = pl.prepare_and_plan(hw_a100, spec, list(reg), reg, drafts="verified", draft=b.repo)
    assert prepared_at == [("vllm", "a" * 40), ("sglang", "a" * 40)]
    assert not plan.errors
    assert all(pm.arch.vocab_size == 128 and pm.spec.revision == "a" * 40 for pm in plan.prepared.values())
    assert reads.count(None) == 1
    from tests.test_objectives_and_search import FakeRunner
    def wrong_revision(*args, **kw):
        raise AssertionError("must not resolve main again after a pinned calibration")
    monkeypatch.setattr(pl, "resolve_revision", wrong_revision)
    profile = pl.calibrate(hw_a100, spec, "balanced", plan, reg, runner=FakeRunner())
    assert profile.model_revision == "a" * 40


def test_profile_provenance_and_errors_use_checked_target_and_private_alias(prepared_vllm, hw_a100):
    pm = prepared_vllm
    original = pm.spec.model_copy(deep=True)
    pm.spec.revision = "a" * 40
    pm.spec_paths = {"local/draft-12345678": "C:/Users/private-person/draft"}
    cfg = Config(backend="vllm", quant="bf16")
    error = "failed to open C:/Users/private-person/draft/model.safetensors"
    trial = TrialResult(config=cfg, stage="spec", metrics=TrialMetrics(errors=[error]), error=error)
    profile = pl._profile_for(hw_a100, original, "balanced", VllmBackend(), cfg, pm,
                              table=[trial], notes=[error])
    assert profile.model_revision == "a" * 40
    assert "private-person" not in profile.model_dump_json()


def test_startup_error_is_scrubbed_before_progress_or_ledger(prepared_vllm, hw_a100):
    from tests.test_draft_failures import run
    prepared_vllm.spec_paths = {"local/draft-12345678": "C:/Users/private-person/draft"}
    result = run("failed C:/Users/private-person/draft/config.json", model=prepared_vllm, hw=hw_a100)
    assert "private-person" not in result.error
    assert "local/draft-12345678" in result.error


def test_conformance_reprepares_target_at_the_checked_pin(no_network, prepared_vllm, hw_a100, tmp_path, monkeypatch):
    from rigtune.conformance import Conformance
    be, seen = VllmBackend(), []
    pm = prepared_vllm
    monkeypatch.setattr(p, "pin_target", lambda spec, **kw: spec.model_copy(update={"revision": "a" * 40}))
    def prepare(spec, hw, quants=None):
        seen.append(spec.revision)
        return pm.model_copy(update={"spec": spec}, deep=True)
    monkeypatch.setattr(be, "prepare", prepare)
    run = Conformance("vllm", be, hw_a100, {"small": pm}, tmp_path, drafts="verified")
    try:
        assert seen == ["a" * 40]
        assert run.models["small"].spec.revision == "a" * 40
    finally:
        run.http.close()


@pytest.mark.usefixtures("no_network", "tmp_home")
@pytest.mark.parametrize("private", ["/home/private-person/draft", "C:/Users/private-person/draft"])
def test_actual_compare_verdict_scrubs_nested_launch_paths_and_errors(tmp_path, monkeypatch, hw_a100,
                                                                    prepared_vllm, private):
    from rigtune.cli import app
    from tests.test_extra_rows import _cli, MODEL
    from tests.test_compare import CompareRunner
    _cli(monkeypatch, hw_a100)
    pm = prepared_vllm
    pm.spec.hf_id = MODEL
    pm.spec.revision = "a" * 40
    alias = "local/draft-12345678"
    pm.spec_paths = {alias: private}
    pm.spec_revisions = {alias: "b" * 64}
    cfg = Config(backend="vllm", quant="bf16", spec_decode=f"draft:{alias}:4")
    profile = pl._profile_for(hw_a100, pm.spec, "balanced", VllmBackend(), cfg, pm)
    monkeypatch.setattr(pl, "resolve_profile", lambda *args, **kw: profile)
    cmp = importlib.import_module("rigtune.bench.compare")
    class Runner(CompareRunner):
        def run(self, cfg, stage):
            tr = super().run(cfg, stage)
            if cfg.spec_decode:
                tr.error = f"cannot read {private}/config.json"
            return tr
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: Runner())
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--verdict", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "private-person" not in result.output
    assert len(list(out.iterdir())) == 4
    for path in out.iterdir():
        assert "private-person" not in path.read_text(encoding="utf-8"), path.name
    data = json.loads(next(f for f in out.glob("*.json") if ".verdict." not in f.name).read_text())
    assert alias in json.dumps(data)


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("cached", ["none", "older", "newest"])
def test_offline_pin_takes_only_a_commit_whose_files_are_cached(spec, hw_a100, monkeypatch, cached):
    """Kept evidence may name a commit that was never downloaded: an engine cannot be launched at it."""
    newest, older = "a" * 40, "c" * 40
    for sha, day in ((older, 1), (newest, 2)):
        a, b = target(repo=spec.hf_id, sha=sha), pair()
        path = p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 4096))
        record = p.load_record(path)
        record.checked_at = f"2026-11-0{day}T00:00:00+00:00"
        p._write_record(path, record)
    present = {"none": None, "older": older, "newest": newest}[cached]

    def read(repo, source="hub", *, revision=None, **kw):
        assert source == "cache"
        if repo == spec.hf_id and revision is not None and revision == present:
            return target(repo=repo, sha=revision)
        raise p.HubLookupError("offline")

    monkeypatch.setattr(p, "read_files", read)
    notes = []
    pinned = p.pin_target(spec, draft="org/draft", hub="off", notes=notes)
    assert pinned.revision == present
    assert bool(notes) is (present is None) and all(newest[:12] in n and "unpinned" in n for n in notes)
    # Through planning: unpinned, no pair is offered, and the profile's notes say why.
    reg = {"vllm": VllmBackend()}
    plan = pl.prepare_and_plan(hw_a100, spec, ["vllm"], reg, drafts="verified", draft="org/draft", hub="off")
    assert plan.prepared["vllm"].spec.revision == present
    if present is None:
        assert not plan.prepared["vllm"].spec_pairs
        assert any("not in the local cache" in n for n in plan.notes)
        assert any("could not be pinned" in n for n in plan.notes)
