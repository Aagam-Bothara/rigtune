"""rigtune/selfcheck.py: the report's arithmetic, its thresholds, the profile note, and how a calibration asks for
it. Reports are built from injected outcomes and engine records; the runs against the real fake engine assert
the report's shape only, never a wall-clock number."""

from __future__ import annotations

import json
import logging

import pytest
from typer.testing import CliRunner

import rigtune.pipeline as pipeline
from rigtune import selfcheck as sc
from rigtune.calibrate.measure import RequestOutcome
from rigtune.calibrate.workload import get_workload
from rigtune.models import ModelSpec


def _outcomes(ttfts_ms, tokens=10, failed=0):
    out = [RequestOutcome(ok=True, ttft_s=t / 1000, duration_s=1.0, tokens=tokens, token_source="usage")
           for t in ttfts_ms]
    return out + [RequestOutcome(ok=False, error="boom") for _ in range(failed)]


def _engine(ttfts_ms, tokens=10, span_s=1.0):
    """Records that start together and end `span_s` later."""
    return [{"start_s": 100.0, "ttft_s": t / 1000, "duration_s": span_s, "tokens": tokens} for t in ttfts_ms]


def test_report_arithmetic():
    client = [21, 22, 23, 24, 25, 26, 27, 28, 29, 40]
    engine = [20] * 9 + [30]
    r = sc.level_report(10, _outcomes(client), 1.0, _engine(engine), [0.001] * 98 + [0.004, 0.009])
    # Taken as a calibration takes them (calibrate/measure.py::_metrics_from): the median, and the element at
    # round(0.95 x (n - 1)) of the sorted list, the last of ten.
    assert (r.client_ttft_p50_ms, r.client_ttft_p95_ms) == (pytest.approx(25.5), pytest.approx(40))
    assert (r.engine_ttft_p50_ms, r.engine_ttft_p95_ms) == (pytest.approx(20), pytest.approx(30))
    assert r.ttft_error_p95_ms == pytest.approx(10) and r.ttft_allowance_ms == 5.0
    # 100 tokens in 1.0 s of wall against 100 sent over the engine's 1.0 s span.
    assert r.client_tok_s == pytest.approx(100) and r.engine_tok_s == pytest.approx(100)
    assert r.throughput_ratio == pytest.approx(1.0)
    assert r.loop_lag_p99_ms == pytest.approx(4.0)  # nearest rank: the 99th of 100 samples
    assert not r.passed and r.problems() == ["p95 time to first token 10.0 ms over the engine's (allowed 5.0)"]


def test_thresholds():
    ok = sc.level_report(4, _outcomes([24] * 4), 1.0, _engine([20] * 4), [])
    assert ok.passed and ok.problems() == []
    slow = sc.level_report(4, _outcomes([24] * 4), 1.04, _engine([20] * 4), [])  # 40 tokens in 1.04 s: 0.962
    assert not slow.passed and slow.problems() == ["throughput 0.962 of the engine's (allowed 0.97)"]
    # A slow engine earns a wider allowance: 5% of its 400 ms p95 is 20 ms, so 15 ms over is within it.
    wide = sc.level_report(4, _outcomes([415] * 4), 1.0, _engine([400] * 4), [])
    assert wide.ttft_allowance_ms == pytest.approx(20) and wide.passed
    failed = sc.level_report(4, _outcomes([20] * 3, failed=1), 0.75, _engine([20] * 3), [])
    assert not failed.passed and failed.problems() == ["1 of 4 requests failed"]


def test_the_note_says_which_levels_missed_and_what_would_fix_it():
    good = sc.level_report(32, _outcomes([21] * 4), 1.0, _engine([20] * 4), [])
    bad = sc.level_report(128, _outcomes([60] * 4), 1.0, _engine([20] * 4), [])
    assert sc.summary([good]) is None
    note = sc.summary([good, bad])
    assert note.startswith("client self-check: at 128 users this machine's load generator is part of what is "
                           "measured (p95 time to first token 40.0 ms over the engine's (allowed 5.0) at 128)")
    assert note.endswith("spread 128 users over 2 load-generator processes (one per 64 users); rigtune does so only "
                         "for multi-GPU layouts so far.")


def test_a_calibration_runs_it_once_per_cpu_and_level(tmp_path):
    from rigtune import __version__

    bad = sc.level_report(128, _outcomes([60] * 4), 1.0, _engine([20] * 4), [])
    calls = []

    def fake_run(users):
        calls.append(list(users))
        return [bad]

    path = tmp_path / "selfcheck.json"
    first = sc.cached_note(128, path, "Test CPU", run_fn=fake_run)
    assert first == sc.summary([bad]) and calls == [[128]]
    assert sc.cached_note(128, path, "Test CPU", run_fn=fake_run) == first and calls == [[128]]  # from the cache
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert list(cached) == [f"{__version__}|Test CPU|128"] and cached[f"{__version__}|Test CPU|128"]["passed"] is False
    sc.cached_note(64, path, "Test CPU", run_fn=fake_run)
    sc.cached_note(128, path, "Other CPU", run_fn=fake_run)
    assert calls == [[128], [64], [128]]


def test_a_check_that_cannot_run_adds_nothing(tmp_path, caplog):
    def broken(users):
        raise RuntimeError("the fake engine did not start")

    with caplog.at_level(logging.WARNING, logger="rigtune.selfcheck"):
        assert sc.cached_note(128, tmp_path / "selfcheck.json", "Test CPU", run_fn=broken) is None
    assert "client self-check did not run: the fake engine did not start" in caplog.text


def _plan(hw):
    from rigtune.backends import registry
    from rigtune.memory import plan
    from rigtune.pipeline import PlanResult
    from tests.test_hybrid_memory import qwen25

    pm, be = qwen25("vllm"), registry()["vllm"]
    feasible = plan(hw, pm, be.candidate_configs(hw, pm), be.memory_model(hw))
    return PlanResult(hw=hw, spec=ModelSpec(hf_id=pm.spec.hf_id), candidates=["vllm"], prepared={"vllm": pm},
                      feasible={"vllm": feasible})


@pytest.mark.usefixtures("tmp_home")
@pytest.mark.parametrize("workload, asked", [("high-concurrency", [(128, "Test CPU")]), ("chat", [])])
def test_only_a_calibration_with_a_level_above_32_asks(workload, asked, hw_rtx4090, monkeypatch):
    from rigtune.backends import registry
    from tests.test_objectives_and_search import FakeRunner

    calls = []
    monkeypatch.setattr(sc, "cached_note", lambda users, path, cpu: calls.append((users, cpu)) and None
                        or "client self-check: missed")
    monkeypatch.setattr(pipeline, "SubprocessTrialRunner", lambda **kw: FakeRunner())  # no engine is launched
    result = _plan(hw_rtx4090)
    profile = pipeline.calibrate(hw_rtx4090, result.spec, "balanced", result, registry(),
                                 workload=get_workload(workload))
    assert calls == asked
    assert ("client self-check: missed" in profile.notes) is bool(asked)
    # An injected runner (tests, replays) never asks.
    calls.clear()
    pipeline.calibrate(hw_rtx4090, result.spec, "balanced", result, registry(), workload=get_workload(workload),
                       runner=FakeRunner())
    assert calls == []


def test_against_the_real_fake_engine_the_report_has_every_level():
    reports = sc.run([1, 3], ttft_ms=0, itl_ms=0, tokens=4)
    assert [(r.users, r.requests, r.failed) for r in reports] == [(1, 8, 0), (3, 8, 0)]
    assert all(r.engine_tok_s > 0 and r.client_tok_s > 0 for r in reports)


def test_the_command_prints_a_level_per_user_count(monkeypatch):
    import rigtune.hardware as hardware
    from rigtune.cli import app
    from rigtune.models import CPUInfo

    monkeypatch.setattr(hardware, "probe_cpu", lambda: CPUInfo(
        model_name="Test CPU", physical_cores=1, logical_cores=1, avx2=False, avx512=False, ram_total_bytes=1,
        ram_free_bytes=1, arch="x86_64"))

    r = CliRunner().invoke(app, ["selfcheck", "--users", "2", "--ttft-ms", "0", "--itl-ms", "0", "--tokens", "4",
                                 "--json"])
    assert r.exit_code in (0, 1), r.output  # 1 when the machine missed a threshold, which is not under test
    out = json.loads(r.output)
    assert [lv["users"] for lv in out["levels"]] == [2] and out["levels"][0]["failed"] == 0
    bad = CliRunner().invoke(app, ["selfcheck", "--users", "two"])
    assert bad.exit_code == 2
