"""rigtune was PolyServe. What the rename must not break: calibrations on disk, recorded results, old settings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rigtune import paths
from rigtune.bench.compare import ComparisonResult, self_label
from rigtune.models import Profile

REPO = Path(__file__).resolve().parent.parent


def test_a_profile_saved_under_the_old_name_still_loads():
    f = REPO / "benchmarks/strategies/results-14b/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    assert "polyserve_version" in data and "rigtune_version" not in data  # really written before the rename
    assert Profile.model_validate(data).rigtune_version == data["polyserve_version"]


def test_a_comparison_recorded_under_the_old_name_still_finds_its_own_row():
    recorded = [p for p in sorted((REPO / "benchmarks/results").glob("*.json"))
                if any(r.get("label") == "polyserve"
                       for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []))]
    assert recorded  # the evidence really is labelled with the old name
    result = ComparisonResult.model_validate_json(recorded[0].read_text(encoding="utf-8"))
    assert result.rigtune_row is not None and all(r.label != "polyserve" for r in result.rows)
    assert self_label("polyserve") == "rigtune" and self_label("vllm-default") == "vllm-default"


def test_the_old_environment_variables_still_work(monkeypatch, tmp_path):
    monkeypatch.delenv("RIGTUNE_HOME", raising=False)
    monkeypatch.setenv("POLYSERVE_HOME", str(tmp_path / "old"))
    assert paths.home() == tmp_path / "old"
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "new"))
    assert paths.home() == tmp_path / "new"  # the new name wins when both are set


def test_calibrations_in_the_old_directory_are_not_thrown_away(monkeypatch, tmp_path):
    """A calibration takes half an hour to an hour of GPU time; an upgrade must not start over in an empty folder."""
    monkeypatch.delenv("RIGTUNE_HOME", raising=False)
    monkeypatch.delenv("POLYSERVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert paths.home() == tmp_path / ".rigtune"  # a fresh machine gets the new name
    (tmp_path / ".polyserve").mkdir()
    assert paths.home() == tmp_path / ".polyserve"  # an upgraded one keeps its calibrations
    (tmp_path / ".rigtune").mkdir()
    assert paths.home() == tmp_path / ".rigtune"  # and once the new folder exists, it wins


def test_the_router_subprocess_names_a_module_that_exists(tmp_path):
    """Multi-GPU calibration starts the router by module path: a string, which no import would catch."""
    from rigtune.disagg import ProcessServer

    srv = ProcessServer("lb", ["http://127.0.0.1:1"], log_path=tmp_path / "router.log")
    srv._log.close()
    exe, flag, module = srv.argv[:3]
    assert flag == "-m"
    done = subprocess.run([exe, "-m", module, "--help"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
