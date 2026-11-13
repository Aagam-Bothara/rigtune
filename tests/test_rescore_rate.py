"""benchmarks/rescore_rate.py: recorded calibrations re-scored by J/request at a rate, under stated assumptions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tests.test_rate_model import level
from tests.test_rate_objective import _config

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "rescore_rate.py"


def _load():
    spec = importlib.util.spec_from_file_location("rescore_rate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _profile(tmp_path: Path) -> Path:
    # Recorded before idle was measured: no idle power on any trial, so the script's assumptions apply.
    a = _config(16, [level(1, 0.5, 150.0), level(4, 1.6, 300.0), level(8, 2.4, 400.0)], idle_w=None)
    b = _config(64, [level(1, 0.5, 200.0), level(4, 1.6, 220.0), level(8, 2.4, 280.0)], idle_w=None)
    early = _config(128, [level(8, 2.4, 250.0)], idle_w=None)  # stopped at its busiest level: says nothing quiet
    data = {"workload_spec": {"concurrencies": [1, 4, 8], "ttft_ceiling_ms": 500.0},
            "hardware": {"gpus": [{"name": "NVIDIA A100-SXM4-80GB"}]},
            "calibration_table": [t.model_dump(mode="json") for t in (a, b, early)]}
    path = tmp_path / "efficiency-chat.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_table_says_where_the_rate_moves_the_pick(tmp_path, capsys):
    mod = _load()
    path = _profile(tmp_path)
    got = mod.rescore(path, [0.05, 1.8], idle_w=60.0, hold_j=30.0, hold_s=1.0)
    assert got["trials"] == 2 and got["skipped"] == 1
    quiet, busy = got["rows"]
    # J/token scores each trial at its best level, 8 users, where b is cheaper; a lone request favours a.
    assert quiet["per_token"].endswith("/b64/gmu0.90") and quiet["per_request"].endswith("/b16/gmu0.90")
    assert quiet["changed"] and not busy["changed"]
    assert quiet["idle"] == 60.0 / 0.05  # the assumed idle draw, over a 20 s gap
    assert quiet["tokens"] == quiet["per_token_tokens"] == 500 and got["size"] == (500, 1, 2)
    assert quiet["judged_at"] == "1/4" and busy["judged_at"] == "8"  # 1 arrival in 10 meets another at 0.05/s
    mod.main([str(tmp_path), "--rates", "0.05", "1.8"])
    out = capsys.readouterr().out
    assert "Assumed where not recorded: idle 60 W, wind-down 30 J over 1 s." in out
    assert "| A100-SXM4-80GB | efficiency-chat.json | 0.05 |" in out and "| yes |" in out and "| no |" in out
    assert "(500) |" in out and "1/4 users |" in out
    assert "2 trials scored, 1 skipped for missing levels; counted in requests of 500 tokens" in out


def test_the_j_per_token_pick_is_flagged_when_the_rate_makes_it_infeasible(tmp_path, capsys):
    mod = _load()
    a = _config(16, [level(1, 0.5, 150.0, p95=200.0), level(4, 1.6, 300.0, p95=3000.0),
                     level(8, 2.4, 400.0, p95=3000.0)], idle_w=None)
    b = _config(64, [level(1, 0.5, 200.0, p95=200.0), level(4, 1.6, 220.0, p95=3000.0),
                     level(8, 2.4, 280.0, p95=3000.0)], idle_w=None)  # the J/token pick, slow under load
    data = {"workload_spec": {"concurrencies": [1, 4, 8], "ttft_ceiling_ms": 500.0},
            "calibration_table": [t.model_dump(mode="json") for t in (a, b)]}
    (tmp_path / "p.json").write_text(json.dumps(data), encoding="utf-8")
    row = mod.rescore(tmp_path / "p.json", [0.3], 60.0, 30.0, 1.0)["rows"][0]
    assert not row["per_token_feasible"]  # 0.6 in flight, judged at 4 users, where its p95 is 3 s
    mod.main([str(tmp_path / "p.json"), "--rates", "0.3"])
    assert "(infeasible at this rate)" in capsys.readouterr().out


def test_a_profile_without_a_calibration_table_is_skipped(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert _load().rescore(tmp_path / "results.json", [0.1], 60.0, 30.0, 1.0) is None
