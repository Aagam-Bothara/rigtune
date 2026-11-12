from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from rigtune import __version__
from rigtune.cli import app

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_probe_json():
    r = runner.invoke(app, ["probe", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "hardware_hash" in data and "backends" in data


def test_profiles_empty(tmp_home):
    r = runner.invoke(app, ["profiles"])
    assert r.exit_code == 0 and "no profiles" in r.output


def test_bad_objective():
    r = runner.invoke(app, ["bench", "x/y", "--objective", "fast"])
    assert r.exit_code != 0


@pytest.mark.usefixtures("no_network")
def test_plan_command(hw_a100, monkeypatch):
    import rigtune.hardware as hwmod

    monkeypatch.setattr(hwmod, "probe", lambda: hw_a100)
    r = runner.invoke(app, ["plan", "meta-llama/Llama-3.2-3B-Instruct", "--backend", "vllm", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["vllm"] and data["_errors"] == {}


def test_serve_listens_on_this_machine_only_by_default():
    """0.0.0.0 would put an unauthenticated inference endpoint on every network the host can reach."""
    import typer.main

    serve = typer.main.get_command(app).commands["serve"]
    host = next(p for p in serve.params if p.name == "host")
    assert host.default == "127.0.0.1"
    assert host.envvar == ["RIGTUNE_HOST", "POLYSERVE_HOST"]  # the Docker image sets the first


def test_loopback_detection():
    from rigtune.cli import _is_loopback

    assert _is_loopback("127.0.0.1") and _is_loopback("localhost") and _is_loopback("::1")
    assert _is_loopback("127.0.0.2")  # the whole 127/8 block is loopback
    assert not _is_loopback("0.0.0.0") and not _is_loopback("10.0.0.5") and not _is_loopback("my-gpu-box")


def test_the_docker_image_opts_back_in_to_every_interface():
    """Inside a container, -p 8000:8000 only reaches a server listening on 0.0.0.0. Without this line the
    documented `docker run` would start cleanly and be unreachable, and no other test would notice."""
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV RIGTUNE_HOST=0.0.0.0" in dockerfile


def test_the_accuracy_gate_options_reach_the_search_options():
    from rigtune.cli import _opts

    opts = _opts("auto", "on", "on", "on", eval_set="gsm8k", max_accuracy_loss=5.0, eval_items=0)
    assert (opts.eval_set, opts.max_accuracy_loss, opts.eval_items) == ("gsm8k", 5.0, 0)
    assert _opts("auto", "on", "on", "on").eval_set is None


def test_a_bad_eval_set_fails_before_anything_runs(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "fine", "answer": "1", "grader": "number"}\n{"prompt": "q", "answer": "1"}\n',
                   encoding="utf-8")
    r = runner.invoke(app, ["recalibrate", "x/y", "--eval-set", str(bad)])
    assert r.exit_code == 2 and "line 2: no grader" in r.output
    r = runner.invoke(app, ["recalibrate", "x/y", "--eval-set", "gsm8k", "--max-accuracy-loss", "0"])
    assert r.exit_code == 2 and "above 0 and below 100" in r.output


@pytest.mark.parametrize("command", ["bench", "recalibrate", "compare", "serve"])
def test_every_calibrating_command_takes_the_accuracy_gate_options(command):
    import typer.main

    params = {p.name for p in typer.main.get_command(app).commands[command].params}
    assert {"eval_set", "max_accuracy_loss", "eval_items", "quality_probe"} <= params


def test_an_eval_set_error_found_during_calibration_is_a_clean_message(monkeypatch):
    """The context check needs the plan, so it can only fail inside calibrate(): that must still read as a user
    error, not a traceback under the live view."""
    import rigtune.pipeline as pl
    from rigtune.evalset import EvalSetError

    def too_long(*args, **kwargs):
        raise EvalSetError("set.jsonl: item long needs about 3512 tokens with its answer, more than the largest "
                           "feasible context (2048)")

    monkeypatch.setattr(pl, "resolve_profile", too_long)
    r = runner.invoke(app, ["recalibrate", "x/y"])
    assert r.exit_code == 2 and "--eval-set: set.jsonl: item long needs about 3512 tokens" in r.output
    assert not isinstance(r.exception, EvalSetError) and "Traceback" not in r.output
