"""A profile calibrated with verified drafts is found again where it was saved, on every route that saves one."""
import functools

import pytest

from rigtune import cache, disagg as D, layout as L, pairing as p, pipeline as pl
from rigtune.cache import profile_path
from rigtune.models import ModelSpec
from rigtune.pipeline import SearchOptions
from tests.test_objectives_and_search import FakeRunner
from tests.test_pairing import pair, target
from tests.test_phases import FakeDisaggRunner, two_gpus
from tests.test_strategies import FakeReplicaRunner

# bench, compare and serve all reach the cache through resolve_profile, by one of these three routes.
ROUTES = {"single": {}, "replicas": {"layout": "replicas"}, "disaggregated": {"phases": "disaggregated"}}
PIN = "a" * 40


@pytest.fixture
def calibrations(no_network, tmp_home, monkeypatch):
    """Real planning, cache and multi-GPU decisions; only the engines are faked. Returns the calibrations run."""
    ran = []
    real = pl.calibrate

    def calibrate(hw, spec, objective, plan, reg, **kw):
        ran.append(spec.revision)
        return real(hw, spec, objective, plan, reg, **{**kw, "runner": FakeRunner(), "progress": None})

    monkeypatch.setattr(pl, "calibrate", calibrate)
    monkeypatch.setattr(D, "connector_available", lambda pkg: True)
    monkeypatch.setattr(L, "calibrate_layout",
                        functools.partial(L.calibrate_layout, replica_runner=FakeReplicaRunner(5000.0)))
    monkeypatch.setattr(D, "calibrate_disaggregated",
                        functools.partial(D.calibrate_disaggregated, runner=FakeDisaggRunner(5000.0)))
    return ran


def resolve_twice(spec, opts, route, ran):
    hw, stages = two_gpus(), []
    first = pl.resolve_profile(spec, "throughput", force_backend="vllm", hw=hw, options=opts, **ROUTES[route])
    count = len(ran)
    again = pl.resolve_profile(spec, "throughput", force_backend="vllm", hw=hw, options=opts,
                               on_stage=stages.append, **ROUTES[route])
    assert "cache hit" in stages and "calibrate" not in stages and len(ran) == count
    assert again.config.key() == first.config.key()
    return first, again


def test_lookup_and_save_name_one_file_when_the_target_was_pinned():
    """The review's demo: the lookup names no commit, the calibrated model carries the pin."""
    opts = {"drafts": "verified"}
    spec = ModelSpec(hf_id="unsloth/Llama-3.1-8B-Instruct")
    lookup = profile_path("hw", spec, "balanced", options=opts)
    pinned = profile_path("hw", spec.model_copy(update={"revision": PIN}), "balanced", options=opts)
    assert lookup != pinned  # so a save may never be filed under the pin


@pytest.mark.parametrize("route", list(ROUTES))
def test_verified_profile_is_found_again(calibrations, spec, monkeypatch, route):
    files = {spec.hf_id: target(repo=spec.hf_id, sha=PIN), "org/draft": pair()}

    def read(repo, *args, **kw):
        if repo not in files:  # the table's own candidates for this target: not published here
            raise p.HubLookupError("not_found")
        return files[repo].model_copy(deep=True)

    monkeypatch.setattr(p, "read_files", read)
    # A stale local `main` must not unseat a profile that launches at its own pin.
    monkeypatch.setattr(cache, "snapshot_revision", lambda model: "c" * 40)
    first, again = resolve_twice(spec, SearchOptions(drafts="verified", draft="org/draft"), route, calibrations)
    assert first.prepared.spec.revision == PIN and again.prepared.spec.revision == PIN
    assert again.prepared.spec_revisions == {"org/draft": "b" * 40}


def test_table_mode_profile_keeps_its_name_and_its_revision_check(calibrations, spec, monkeypatch, tmp_home):
    first, _ = resolve_twice(spec, SearchOptions(), "single", calibrations)
    saved = [f.name for f in tmp_home.rglob("throughput*.json")]
    assert saved == [profile_path("hw", spec, "throughput", options=first.options).name]
    assert "-rev-" not in saved[0] and "drafts" not in saved[0]
    # Unpinned, the local snapshot still decides: a new commit recalibrates, as before.
    first.model_revision = "d" * 40
    monkeypatch.setattr(cache, "snapshot_revision", lambda model: "e" * 40)
    assert "is now commit" in cache.revision_reason(first, spec)
