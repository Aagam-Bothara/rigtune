"""The engine releases rigtune was checked with (rigtune/data/conformance), and the note any other one gets: a
log line and a profile note, never an error, and nothing at all for a listed release."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

import rigtune.pipeline as pipeline
from rigtune import engines
from rigtune.calibrate.workload import get_workload
from rigtune.models import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


def test_every_record_is_where_its_engine_and_version_say():
    names = sorted(p.name for p in engines.DATA.glob("*.json"))
    assert names == ["sglang-0.5.19.json", "sglang-0.5.23.json", "vllm-0.11.0.json", "vllm-0.29.0.json",
                     "vllm-0.33.0.json"]
    for path in engines.DATA.glob("*.json"):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            t = engines.Tested.model_validate(rec)
            assert path.name == f"{t.engine}-{t.version}.json"
            assert t.checks and set(t.checks.values()) <= {"pass", "fail", "skip"} and t.source


def test_the_checked_releases():
    assert engines.checked_versions("vllm") == ["0.11.0", "0.29.0", "0.33.0"]
    assert engines.checked_versions("sglang") == ["0.5.19", "0.5.23"]
    # 0.33.0 and 0.5.23 were added by the A40 conformance run of 17 November 2026, which certified both.
    for engine, version in (("vllm", "0.33.0"), ("sglang", "0.5.23")):
        (rec,) = [t for t in engines.tested()[engine] if t.version == version]
        assert rec.card == "NVIDIA A40" and rec.date == "2026-11-17" and rec.source.startswith("rigtune conformance")


@pytest.mark.parametrize("engine, version, note", [
    ("vllm", "0.29.0", None),
    ("vllm", "0.11.0", None),
    ("sglang", "0.5.19", None),
    ("vllm", "0.33.0", None),
    ("sglang", "0.5.23", None),
    ("vllm", "0.34.0", "vllm 0.34.0 has not been checked with rigtune (checked: 0.11.0, 0.29.0, 0.33.0); its "
                       "flags, defaults and memory rules may differ from what rigtune plans for"),
    ("vllm", "0.29.1", "vllm 0.29.1 has not been checked with rigtune (checked: 0.11.0, 0.29.0, 0.33.0); its "
                       "flags, defaults and memory rules may differ from what rigtune plans for"),  # a patch release too
    ("sglang", "0.5.24", "sglang 0.5.24 has not been checked with rigtune (checked: 0.5.19, 0.5.23); its flags, "
                         "defaults and memory rules may differ from what rigtune plans for"),
    ("sglang", None, None),  # not installed
    ("llamacpp-cuda", "b6000", None),  # no table kept for it
    ("vllm-cpu", "0.29.0", None),
])
def test_support_note(engine, version, note):
    assert engines.support_note(engine, version) == note


def _plan(hw, backend="vllm"):
    from rigtune.backends import registry
    from rigtune.memory import plan
    from rigtune.pipeline import PlanResult
    from tests.test_hybrid_memory import qwen25

    pm, be = qwen25(backend), registry()[backend]
    feasible = plan(hw, pm, be.candidate_configs(hw, pm), be.memory_model(hw))
    return PlanResult(hw=hw, spec=ModelSpec(hf_id=pm.spec.hf_id), candidates=[backend], prepared={backend: pm},
                      feasible={backend: feasible})


def _calibrate(hw, version):
    from rigtune.backends import registry
    from tests.test_objectives_and_search import FakeRunner

    hw = hw.model_copy(deep=True)
    hw.backends["vllm"] = hw.backends["vllm"].model_copy(update={"version": version})
    result = _plan(hw)
    return pipeline.calibrate(hw, result.spec, "balanced", result, registry(), workload=get_workload("chat"),
                              runner=FakeRunner())


def _without_timing(notes):
    return [n for n in notes if not n.startswith("calibration took")]


@pytest.mark.usefixtures("tmp_home")
def test_an_unchecked_release_is_logged_once_and_noted_and_still_calibrated(hw_rtx4090, caplog, monkeypatch):
    monkeypatch.setattr(engines, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="rigtune.engines"):
        first, second = _calibrate(hw_rtx4090, "0.34.0"), _calibrate(hw_rtx4090, "0.34.0")
    note = engines.support_note("vllm", "0.34.0")
    assert note in first.notes and note in second.notes
    assert caplog.text.count("vllm 0.34.0 has not been checked") == 1
    # The note is all it changes: the same pick, trials and every other note as on a checked release.
    checked = _calibrate(hw_rtx4090, "0.29.0")
    assert first.config.key() == checked.config.key()
    assert [r.config.key() for r in first.calibration_table] == [r.config.key() for r in checked.calibration_table]
    assert [n for n in _without_timing(first.notes) if n != note] == _without_timing(checked.notes)


@pytest.mark.usefixtures("tmp_home")
def test_a_checked_release_adds_nothing(hw_rtx4090, monkeypatch):
    checked = _calibrate(hw_rtx4090, "0.29.0")
    monkeypatch.setattr(engines, "support_note", lambda engine, version: None)  # as if the table did not exist
    assert _without_timing(_calibrate(hw_rtx4090, "0.29.0").notes) == _without_timing(checked.notes)
    assert not any("has not been checked" in n for n in checked.notes)


def test_the_probe_prints_the_table_and_the_note(hw_rtx4090, monkeypatch):
    import rigtune.hardware as hardware
    from rigtune.cli import app

    hw = hw_rtx4090.model_copy(deep=True)
    hw.backends["vllm"] = hw.backends["vllm"].model_copy(update={"version": "0.34.0"})
    hw.backends["sglang"] = hw.backends["sglang"].model_copy(update={"version": "0.5.19"})
    monkeypatch.setattr(hardware, "probe", lambda: hw)
    r = CliRunner().invoke(app, ["probe"], env={"COLUMNS": "250"})
    assert r.exit_code == 0, r.output
    assert "Engine releases rigtune was checked with" in r.output
    assert "vllm 0.29.0" in r.output and "sglang 0.5.19" in r.output and "5.23.0" in r.output
    assert "note: vllm 0.34.0 has not been checked with rigtune" in r.output
    assert "sglang 0.5.19 has not been checked" not in r.output


def test_the_wheel_carries_the_table():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    assert '"data/conformance/*.json"' in section
