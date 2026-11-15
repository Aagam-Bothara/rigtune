"""A cached profile is served only where it answers today's request: the same constraints, model commit and engine
libraries it was calibrated with."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import pytest

from rigtune import cache, hfconfig, pipeline
from rigtune.calibrate.objectives import Constraints
from rigtune.calibrate.workload import get_workload
from rigtune.models import ModelSpec
from rigtune.pipeline import SearchOptions, default_constraints
from tests.test_cache_and_pipeline import _profile

COMMIT_A, COMMIT_B = "a" * 40, "b" * 40


def _refuse_calibration(monkeypatch, message):
    def select(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "select", select)


# ---- constraints


@pytest.mark.parametrize("changes", [
    {"ttft_ceiling_ms": 50}, {"tpot_ceiling_ms": 10}, {"tok_s_floor_abs": 999},
    {"tok_s_floor_frac": 0.9}, {"noise_tolerance": 0.1},
])
def test_new_constraints_do_not_reuse_an_old_pick(changes, tmp_home, hw_a100, spec, prepared_vllm, monkeypatch):
    p = _profile(hw_a100, spec, prepared_vllm)
    p.options = SearchOptions().key("balanced")
    cache.save(p)
    assert pipeline.resolve_profile(spec, hw=hw_a100) is not None  # the defaults are served from it
    _refuse_calibration(monkeypatch, "fresh calibration required")
    wanted = Constraints(**{**asdict(default_constraints(None, SearchOptions())), **changes})
    with pytest.raises(RuntimeError, match="fresh calibration required"):
        pipeline.resolve_profile(spec, hw=hw_a100, constraints=wanted)


def test_a_profile_scored_with_other_constraints_is_not_served_at_the_same_path(tmp_home, hw_a100, spec,
                                                                             prepared_vllm, monkeypatch, caplog):
    """The fingerprint only covers non-default constraints; if a workload's default ceiling moves between releases,
    the record itself still tells an old pick from a new request."""
    p = _profile(hw_a100, spec, prepared_vllm)
    p.options = SearchOptions().key("balanced")
    p.calibration_constraints["ttft_ceiling_ms"] = 900.0
    cache.save(p)
    _refuse_calibration(monkeypatch, "fresh calibration required")
    with caplog.at_level(logging.INFO, logger="rigtune.pipeline"), \
            pytest.raises(RuntimeError, match="fresh calibration required"):
        pipeline.resolve_profile(spec, hw=hw_a100)
    assert "calibrated for other constraints (ttft_ceiling_ms 900.0 -> 500.0); recalibrating" in caplog.text


def test_legacy_profiles_without_recorded_constraints_recalibrate_once_and_say_why(tmp_home, hw_a100, spec,
                                                                                 prepared_vllm, monkeypatch, caplog):
    p = _profile(hw_a100, spec, prepared_vllm)
    p.calibration_constraints = {}
    p.options = SearchOptions().key("balanced")
    cache.save(p)
    _refuse_calibration(monkeypatch, "legacy profile needs calibration")
    with caplog.at_level(logging.INFO, logger="rigtune.pipeline"), \
            pytest.raises(RuntimeError, match="legacy profile needs calibration"):
        pipeline.resolve_profile(spec, hw=hw_a100)
    assert "records no calibration constraints" in caplog.text and "recalibrating once" in caplog.text


def test_a_cached_disaggregated_balanced_profile_is_found_again(tmp_home, hw_a100, spec, prepared_vllm,
                                                                monkeypatch):
    """It is saved under its base's options, which carry the objective's percentile key; the lookup used to drop
    the objective, so it looked under another name and recalibrated every time."""
    p = _profile(hw_a100, spec, prepared_vllm)
    p.phases = "auto"
    p.options = SearchOptions().key("balanced")
    assert p.options == {"ttft": "p95"}
    cache.save(p)
    _refuse_calibration(monkeypatch, "recalibrated instead of reusing")
    assert pipeline.resolve_profile(spec, "balanced", hw=hw_a100, phases="auto").phases == "auto"


def test_default_constraints_add_no_key_so_cache_paths_stay_as_they_were():
    from rigtune.cli import _constraints

    for name in ("default", "chat", "long-context", "high-concurrency"):
        wl = get_workload(name)
        assert SearchOptions().key("balanced", _constraints(None, None, wl), wl) == {"ttft": "p95"}
        assert SearchOptions().key("throughput", None, wl) == {}
    wl = get_workload("default")
    assert SearchOptions(ttft_percentile=50).key("balanced", _constraints(None, None, wl, ttft_percentile=50),
                                                 wl) == {}


def test_the_fingerprint_does_not_depend_on_how_a_number_was_written():
    wl = get_workload("default")
    whole = SearchOptions().key("balanced", Constraints(ttft_ceiling_ms=50, tpot_ceiling_ms=100), wl)
    decimal = SearchOptions().key("balanced", Constraints(ttft_ceiling_ms=50.0, tpot_ceiling_ms=100.0), wl)
    assert whole == decimal and "limits" in whole
    other = SearchOptions().key("balanced", Constraints(ttft_ceiling_ms=60, tpot_ceiling_ms=100), wl)
    assert other["limits"] != whole["limits"]


def test_a_constraint_added_later_with_a_default_does_not_re_key_existing_profiles():
    @dataclass
    class Wider(Constraints):
        new_knob: float = 1.0

    wl = get_workload("default")
    for fields in ({}, {"ttft_ceiling_ms": 50.0}):
        before = SearchOptions().key("balanced", Constraints(**fields), wl)
        assert SearchOptions().key("balanced", Wider(**fields), wl) == before
    assert SearchOptions().key("balanced", Wider(new_knob=2.0), wl) != SearchOptions().key(
        "balanced", Constraints(), wl)
    # ...and a record written before the field existed reads it as its default.
    record = asdict(default_constraints(wl, SearchOptions()))
    assert pipeline.constraint_changes(record, Wider(**record)) == {}


@pytest.mark.usefixtures("no_network")
def test_calibration_records_its_constraints_ceilings_and_model_commit(hw_a100, spec, monkeypatch):
    from tests.test_objectives_and_search import FakeRunner

    monkeypatch.setattr(pipeline, "resolve_revision", lambda s: COMMIT_A)
    _, reg = pipeline.select(hw_a100, spec, force="vllm")
    wl = get_workload("default")
    plan = pipeline.prepare_and_plan(hw_a100, spec, ["vllm"], reg, workload=wl)
    cons = Constraints(ttft_ceiling_ms=800.0, tpot_ceiling_ms=60.0)
    profile = pipeline.calibrate(hw_a100, spec, "throughput", plan, reg, workload=wl, constraints=cons,
                                 runner=FakeRunner())
    assert profile.model_revision == COMMIT_A
    assert profile.calibration_constraints == asdict(cons)
    assert profile.workload_spec["ttft_ceiling_ms"] == 800.0 and profile.workload_spec["tpot_ceiling_ms"] == 60.0
    assert "limits" in profile.options


# ---- the model's commit


def test_explicit_model_revisions_have_separate_cache_paths(tmp_home, hw_a100, spec, prepared_vllm):
    one = ModelSpec(hf_id=spec.hf_id, revision="commit-one")
    two = ModelSpec(hf_id=spec.hf_id, revision="commit-two")
    p = _profile(hw_a100, one, prepared_vllm.model_copy(deep=True))
    p.prepared.spec = one
    cache.save(p)
    assert cache.load(hw_a100, one, "balanced") is not None
    assert cache.load(hw_a100, two, "balanced") is None
    assert cache.load(hw_a100, spec, "balanced") is None


@pytest.mark.parametrize("recorded, current, reused, said", [
    (COMMIT_A, COMMIT_A, True, None),
    (COMMIT_A, COMMIT_B, False, "is now commit bbbbbbbbbbbb, calibrated on aaaaaaaaaaaa); recalibrating"),
    (COMMIT_A, None, True, "cannot be resolved offline; calibrated on aaaaaaaaaaaa"),
    (None, COMMIT_B, True, None),  # not recorded (an older profile, a local directory): nothing to compare
])
def test_a_reused_profile_is_checked_against_the_local_snapshot_when_both_commits_are_known(
        recorded, current, reused, said, tmp_home, hw_a100, spec, prepared_vllm, monkeypatch, caplog):
    p = _profile(hw_a100, spec, prepared_vllm)
    p.model_revision = recorded
    cache.save(p)
    monkeypatch.setattr(cache, "snapshot_revision", lambda m: current)
    with caplog.at_level(logging.INFO, logger="rigtune.cache"):
        assert (cache.load(hw_a100, spec, "balanced") is not None) == reused
    if said:
        assert said in caplog.text


def _snapshot(root, repo, commit, ref="main"):
    base = root / ("models--" + repo.replace("/", "--"))
    (base / "refs").mkdir(parents=True)
    (base / "refs" / ref).write_text(commit)
    (base / "snapshots" / commit).mkdir(parents=True)
    (base / "snapshots" / commit / "config.json").write_text("{}")


def test_the_commit_is_read_from_the_cached_snapshot_path_without_asking_the_hub(tmp_path):
    _snapshot(tmp_path, "org/model", COMMIT_A)
    asked = []
    found = hfconfig.resolve_revision(ModelSpec(hf_id="org/model"), cache_dir=tmp_path,
                                      fetch=lambda s: asked.append(s) or COMMIT_B)
    assert found == COMMIT_A and asked == []
    assert hfconfig.snapshot_revision(ModelSpec(hf_id="org/model"), cache_dir=tmp_path) == COMMIT_A
    assert hfconfig.snapshot_revision(ModelSpec(hf_id="org/other"), cache_dir=tmp_path) is None


def test_an_uncached_model_asks_the_hub_once_and_a_failure_records_nothing(tmp_path):
    spec = ModelSpec(hf_id="org/model")
    assert hfconfig.resolve_revision(spec, cache_dir=tmp_path, fetch=lambda s: COMMIT_B) == COMMIT_B

    def offline(s):
        raise OSError("no network")

    assert hfconfig.resolve_revision(spec, cache_dir=tmp_path, fetch=offline) is None
    assert hfconfig.resolve_revision(ModelSpec(hf_id=str(tmp_path)), fetch=lambda s: COMMIT_B) is None  # a directory


# ---- the engine's runtime


@pytest.mark.parametrize("change", ["libraries", "driver"])
def test_runtime_changes_invalidate_the_pick(change, hw_a100, spec, prepared_vllm):
    hw_a100.backends["vllm"].libs = {"torch": "2.13.0"}
    p = _profile(hw_a100.model_copy(deep=True), spec, prepared_vllm)
    if change == "libraries":
        hw_a100.backends["vllm"].libs = {"torch": "2.14.0"}
    else:
        hw_a100.gpus[0].driver_version = "different"
    assert cache.invalid_reason(p, hw_a100) is not None


def test_simultaneous_saves_do_not_share_a_temporary_file(tmp_home, hw_a100, spec, prepared_vllm, monkeypatch):
    p = _profile(hw_a100, spec, prepared_vllm)
    barrier = threading.Barrier(2)
    replace = cache.os.replace
    staged = []
    replace_lock = threading.Lock()

    def together(src, dst):
        staged.append(str(src))
        barrier.wait(timeout=10)
        with replace_lock:
            return replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", together)
    with ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(cache.save, [p, p]))
    assert paths[0] == paths[1]
    assert len(set(staged)) == 2  # one writer cannot replace the other's staging file
    assert cache.load(hw_a100, spec, "balanced").config == p.config
    assert sorted(x.name for x in paths[0].parent.iterdir()) == [paths[0].name]  # no staging file left behind
