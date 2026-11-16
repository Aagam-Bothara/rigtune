"""A local draft directory stays out of what multi-GPU and split prefill/decode profiles show and save."""
import io
import json

import pytest
from rich.console import Console

from rigtune import cache, cli, disagg as D
from rigtune.layout import calibrate_layout
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_objectives_and_search import FakeRunner
from tests.test_phases import FakeDisaggRunner, two_gpus
from tests.test_strategies import FakeReplicaRunner

ALIAS = "local/draft-12345678"
PRIVATE = {"windows": "C:\\Users\\private-person\\models\\draft", "posix": "/home/private-person/models/draft"}


def leaks(text, private):
    """The directory as given, JSON-escaped once (a launch argument) and twice (that argument in a saved file)."""
    forms = {private, json.dumps(private)[1:-1], json.dumps(json.dumps(private))[2:-2], "private-person"}
    return sorted(form for form in forms if form in text)


def local_draft_pick(spec, engine, private):
    """A single-GPU profile whose pick drafts from a directory under the user's home."""
    hw = two_gpus()
    candidates, reg = select(hw, spec, force=engine)
    base = calibrate(hw, spec, "throughput", prepare_and_plan(hw, spec, candidates, reg), reg, runner=FakeRunner())
    picked = base.config.key()
    base.config = base.config.model_copy(update={"spec_decode": f"draft:{ALIAS}:4"})
    for trial in base.calibration_table:
        if trial.config.key() == picked:
            trial.config = base.config
    for model in [base.prepared, *base.prepared_all.values()]:
        model.spec_paths = {ALIAS: private}
        model.spec_revisions = {ALIAS: "b" * 64}
    # The engine itself is launched with the directory: only what is shown and saved names the alias.
    assert leaks(" ".join(reg[engine].launch_spec(base.config, base.prepared, 0).args), private)
    return hw, reg, base


def shown_and_saved(profile, monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=100_000))
    cli._print_profile(profile)
    return out.getvalue() + cache.save(profile).read_text(encoding="utf-8")


@pytest.mark.usefixtures("no_network", "tmp_home")
@pytest.mark.parametrize("form", list(PRIVATE))
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_replicas_profile_names_the_alias_not_the_directory(spec, monkeypatch, engine, form):
    hw, reg, base = local_draft_pick(spec, engine, PRIVATE[form])
    profile = calibrate_layout(hw, "throughput", base, reg, "replicas", replica_runner=FakeReplicaRunner(9000.0))
    assert profile.replicas == 2
    text = shown_and_saved(profile, monkeypatch)
    assert ALIAS in " ".join(profile.launch_args) and ALIAS in text
    assert not leaks(text, PRIVATE[form])


@pytest.mark.usefixtures("no_network", "tmp_home")
@pytest.mark.parametrize("form", list(PRIVATE))
def test_disaggregated_profile_names_the_alias_not_the_directory(spec, monkeypatch, form):
    monkeypatch.setattr(D, "connector_available", lambda pkg: True)
    hw, reg, base = local_draft_pick(spec, "vllm", PRIVATE[form])
    profile = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="disaggregated",
                                        runner=FakeDisaggRunner(9000.0))
    assert profile.disagg is not None
    text = shown_and_saved(profile, monkeypatch)
    assert ALIAS in " ".join(profile.launch_args) and ALIAS in text
    assert not leaks(text, PRIVATE[form])


class _Refusing:
    """Every multi-GPU trial fails with an engine error that names the directory."""

    def __init__(self, private):
        self.error = f"failed to start (rc=1)\nOSError: cannot read {private}/config.json"

    def _failed(self, cfg, stage, **kw):
        from rigtune.models import TrialMetrics, TrialResult
        return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=self.error, **kw)

    def run_replicas(self, cfg, gpus, stage):
        return self._failed(cfg, stage, replicas=len(gpus))

    def run(self, cfg, stage):
        decode = getattr(cfg, "decode", None)  # a prefill/decode pair, or one tensor-parallel engine
        return self._failed(decode or cfg, stage, **({"disagg": cfg} if decode is not None else {}))


@pytest.mark.usefixtures("no_network", "tmp_home")
@pytest.mark.parametrize("form", list(PRIVATE))
@pytest.mark.parametrize("arrangement", ["layout", "phases"])
def test_failed_multi_gpu_trials_are_scrubbed_in_progress_notes_and_table(spec, monkeypatch, form, arrangement):
    monkeypatch.setattr(D, "connector_available", lambda pkg: True)
    hw, reg, base = local_draft_pick(spec, "vllm", PRIVATE[form])
    failing, seen = _Refusing(PRIVATE[form]), []

    def progress(stage, cfg, res):
        seen.append(res.error if res is not None else "")

    if arrangement == "layout":
        profile = calibrate_layout(hw, "throughput", base, reg, "auto", replica_runner=failing, tp_runner=failing,
                                   progress=progress)
    else:
        profile = D.calibrate_disaggregated(hw, spec, "throughput", base, reg, phases="auto", runner=failing,
                                            progress=progress)
    assert any(ALIAS in error for error in seen) and any("failed" in note for note in profile.notes)
    assert not leaks("\n".join(seen) + shown_and_saved(profile, monkeypatch), PRIVATE[form])
