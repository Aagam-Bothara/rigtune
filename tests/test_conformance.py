"""The conformance run against a fake engine (tests/conformance_engine.py) launched with rigtune's real launch
arguments: the record's shape, the pass and fail logic, and that a failed check never stops the rest. Nothing here
needs a GPU; the run itself has never been on one."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from rigtune import conformance as cf
from rigtune import engines, introspect, speculative
from rigtune.backends import registry
from rigtune.backends.base import LaunchSpec, Process
from tests import golden_cases
from tests.engine_trees import installed

ENGINE = str(Path(__file__).with_name("conformance_engine.py"))


class _Level:
    def __init__(self, users, passed=True):
        self.users, self.passed = users, passed

    def problems(self):
        return [] if self.passed else ["p95 time to first token 9.0 ms over the engine's (allowed 5.0)"]

    def to_dict(self):
        return {"users": self.users, "passed": self.passed}


def _fake_launch(engine):
    """The backend's launch, with the engine swapped for the fake one: the flags are rigtune's own."""

    def launch(self, cfg, model, port, log_path=None):
        spec = self.launch_spec(cfg, model, port)
        first = next(i for i, a in enumerate(spec.args) if a.startswith("--"))
        args = [sys.executable, ENGINE, "--fake-engine", engine] + spec.args[first:]
        return Process(LaunchSpec(args=args, env=spec.env), port, f"http://127.0.0.1:{port}/health",
                       log_path=log_path).start()

    return launch


@pytest.fixture
def run_for(tmp_path, monkeypatch):
    """A Conformance run of `engine` `version` as rigtune sees that release (tests/engine_trees.py), its probe
    answering with the release's facts fixture, launching the fake engine."""
    import contextlib

    stack = contextlib.ExitStack()

    def make(engine, version, models=("small",), selfcheck_passes=True, only=None, card="rtx4090"):
        stack.enter_context(installed(engine, version, tmp_path / "tree"))
        mod = vl if engine == "vllm" else sg
        facts = golden_cases.facts_fixture(engine, version)
        monkeypatch.setattr(mod, "_engine_facts", functools.lru_cache(maxsize=1)(lambda: facts))
        monkeypatch.setattr(type(registry()[engine]), "launch", _fake_launch(engine))
        monkeypatch.setattr(speculative, "find_eagle3_head", lambda spec, arch, **kw: (golden_cases.HEAD, 850_000_000))
        monkeypatch.setattr(speculative, "repo_bytes", lambda repo, token=None: 2_471_645_608)
        monkeypatch.setattr(introspect, "inspect_model", lambda spec, **kw: introspect.ModelFacts())
        monkeypatch.setattr(vl, "nvlink_between", lambda gpus: False)
        names = {"small": "llama-3.1-8b", "hybrid": "qwen3.5-9b-mtp", "multimodal": "gemma-4"}
        prepared = {role: golden_cases.prepared(names[role], engine) for role in models}
        hw = golden_cases.hw(card, engine, version)
        return cf.Conformance(engine, registry()[engine], hw, prepared, tmp_path / "out", startup_timeout=60,
                              selfcheck=lambda users: [_Level(u, selfcheck_passes) for u in users],
                              reporting=registry(report_reuse=True)[engine], only=only, poll=0.05), facts, hw

    yield make
    stack.close()


def _status(results):
    return {r.id: r.status for r in results}


def test_a_vllm_release_that_does_what_rigtune_plans_is_certified(run_for):
    run, facts, hw = run_for("vllm", "0.33.0", models=("small", "hybrid", "multimodal"))
    results = run.run()
    status = _status(results)
    assert status == {"C01": "pass", "C02": "pass", "C05": "pass", "C03": "pass", "C04": "pass",
                      "C06-ngram_gpu": "pass", "C06-suffix": "pass", "C06-draft": "pass", "C06-eagle3": "pass",
                      "C06-mtp-hybrid": "pass", "C07": "pass",
                      "C08": "pass", "C09": "pass", "C10": "pass", "C11": "pass", "C12": "pass"}, \
        {r.id: r.detail for r in results if r.status != "pass"}
    assert cf.certified(results)
    by = {r.id: r for r in results}
    assert by["C02"].data["measured"]["weights_mb"] == pytest.approx(0.93 * 1024)
    assert by["C04"].data["cached_tokens"] == [0, 384]
    assert sorted(by["C03"].data["launches"]) == ["kv-fp8", "kv-int8_per_token_head", "prefill-2048",
                                                  "prefix-off", "prefix-on"]
    assert by["C07"].data["error"].startswith("out of memory at start-up")
    assert by["C06-ngram_gpu"].data["accepted_tokens"] > 0
    assert by["C10"].data["planned_hybrid"]["state_slots"] > 0 and by["C10"].data["measured"]["kv_mb"]


def test_an_sglang_release_skips_what_rigtune_does_not_use_there(run_for):
    # An A40: on a 24 GB card the planner leaves Llama 3.1 8B no room for SGLang's second models at batch 256.
    run, facts, hw = run_for("sglang", "0.5.23", models=("small", "multimodal"), card="a40")
    results = run.run()
    status = _status(results)
    assert status["C05"] == "skip"  # rigtune reads no SGLang metrics
    assert status["C08"] == "skip"  # rigtune has no text-only launch for SGLang
    assert status["C10"] == "skip"  # no --hybrid-model
    assert {k: v for k, v in status.items() if k.startswith("C06")} == {
        "C06-ngram": "pass", "C06-eagle3": "pass", "C06-draft": "pass"}
    assert all(v == "pass" for k, v in status.items() if k not in ("C05", "C08", "C10"))
    assert cf.certified(results)  # C05 does not apply to SGLang, which must not keep every release uncertified


def test_a_failed_check_never_stops_the_rest(run_for, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "int8_per_token_head,draft_model")
    run, facts, hw = run_for("vllm", "0.33.0")
    results = run.run()
    status = _status(results)
    assert status["C03"] == "fail" and "kv-int8_per_token_head" in next(r for r in results if r.id == "C03").detail
    assert status["C06-draft"] == "fail" and status["C06-ngram_gpu"] == "pass"
    assert status["C11"] == "pass" and status["C12"] == "pass"  # later checks still ran
    assert not cf.certified(results)


def test_optional_head_field_stays_unanswered_without_revoking_baseline_conformance(run_for):
    run, facts, _ = run_for("vllm", "0.33.0", only=["C01"])
    assert facts.spec_fields is None and not facts.answered("spec_fields")
    result, = run.run()
    assert result.status == "pass"
    assert "spec_fields" in result.data["unanswered"]


def test_a_check_that_raises_is_a_failed_check(run_for, monkeypatch):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C04", "C11"])

    def broken():
        raise RuntimeError("the GPU fell off the bus")

    monkeypatch.setattr(run, "c04_reuse", broken)
    results = run.run()
    assert [(r.id, r.status) for r in results] == [("C04", "fail"), ("C11", "pass")]
    assert "the GPU fell off the bus" in results[0].detail


def test_an_engine_that_reports_no_reuse_or_a_probe_that_disagrees_is_not_certified(run_for, monkeypatch):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C01", "C04"])
    run.reporting = run.backend  # launched without --enable-prompt-tokens-details: no cached tokens in usage
    wrong = facts.model_copy(update={"flags": {"--attention-backend": {"choices": None, "default": None}}})
    monkeypatch.setattr(vl, "_engine_facts", functools.lru_cache(maxsize=1)(lambda: wrong))
    by = {r.id: r for r in run.run()}
    assert by["C04"].status == "fail" and by["C04"].data["cached_tokens"] == [None, None]
    assert by["C01"].status == "fail"
    assert by["C01"].data["disagreements"] == ["language_model_only", "prompt_tokens_details"]


def test_without_the_probe_c01_fails_and_says_why(run_for, monkeypatch):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C01"])
    monkeypatch.setattr(vl, "_engine_facts", functools.lru_cache(maxsize=1)(lambda: None))
    (result,) = run.run()
    assert result.status == "fail" and "not run" in result.detail


def test_an_oversized_configuration_that_starts_is_skipped_not_passed(run_for, monkeypatch):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C07"])
    monkeypatch.setattr(cf, "OVERSIZED", {"gpu_memory_utilization": 0.85, "batch": 1024})
    (result,) = run.run()
    assert result.status == "skip"


@pytest.mark.parametrize("engine, version, card", [("vllm", "0.29.0", "rtx4090"), ("sglang", "0.5.19", "a40")])
def test_c07_asks_for_no_more_of_the_card_than_vllm_grants_at_start_up(run_for, engine, version, card):
    """vLLM refuses a fraction above the card's free share before loading anything (v1/worker/utils.py, line 761 at
    v0.33.0 and 525 at v0.29.0), with a ValueError that is not an out-of-memory error: at 99% that is every card, and
    C07 failed on each. The fake engine refuses the same way."""
    assert cf.OVERSIZED["gpu_memory_utilization"] <= 0.9
    run, facts, hw = run_for(engine, version, only=["C07"], card=card)
    (result,) = run.run()
    assert result.status == "pass" and result.data["error"].startswith("out of memory at start-up"), result


def test_a_refusal_by_vllms_free_memory_check_is_skipped_not_failed(run_for, monkeypatch):
    """A card another process holds part of: the refusal says nothing about the oversized configuration."""
    run, facts, hw = run_for("vllm", "0.33.0", only=["C07"])
    monkeypatch.setattr(cf, "OVERSIZED", dict(cf.OVERSIZED, gpu_memory_utilization=0.99))
    (result,) = run.run()
    assert result.status == "skip" and "free-memory check" in result.detail
    assert not result.data["error"].startswith("out of memory")


def test_one_launch_that_raises_does_not_stop_the_others_of_its_check(run_for, monkeypatch):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C03", "C06"])
    served, spec_check = run._served, run._spec_check

    def served_but_one(cfg, model, label):
        if label == "kv-fp8":
            raise OSError("no space left on device")
        return served(cfg, model, label)

    def spec_but_one(cid, variant, model, notes):
        if cid == "C06-suffix":
            raise RuntimeError("the launch refused")
        return spec_check(cid, variant, model, notes)

    monkeypatch.setattr(run, "_served", served_but_one)
    monkeypatch.setattr(run, "_spec_check", spec_but_one)
    by = {r.id: r for r in run.run()}
    launches = by["C03"].data["launches"]
    assert by["C03"].status == "fail" and by["C03"].detail == "failed: kv-fp8"
    assert "no space left" in launches["kv-fp8"]["error"]
    assert sorted(k for k, s in launches.items() if s["ready"]) == ["kv-int8_per_token_head", "prefill-2048",
                                                                    "prefix-off", "prefix-on"]
    assert by["C06-suffix"].status == "fail" and "the launch refused" in by["C06-suffix"].detail
    assert {k: r.status for k, r in by.items() if k.startswith("C06") and k != "C06-suffix"} == {
        "C06-ngram_gpu": "pass", "C06-draft": "pass", "C06-eagle3": "pass"}


def test_a_start_up_failure_that_is_not_memory_fails_c07(run_for, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "--max-num-seqs")
    run, facts, hw = run_for("vllm", "0.33.0", only=["C07"])
    monkeypatch.setattr(cf, "OVERSIZED", {"gpu_memory_utilization": 0.85, "batch": 1024})
    (result,) = run.run()
    assert result.status == "fail" and result.detail.startswith("classified as: failed to start")


def test_a_client_that_misses_a_threshold_fails_c12_only(run_for):
    run, facts, hw = run_for("vllm", "0.33.0", selfcheck_passes=False, only=["C12"])
    (result,) = run.run()
    assert result.status == "fail" and result.detail.startswith("64 users: p95")


def test_the_record_is_one_entry_of_the_checked_releases_table(run_for, tmp_path):
    run, facts, hw = run_for("vllm", "0.33.0", only=["C01", "C11"])
    results = run.run()
    rec = cf.record("vllm", "0.33.0", hw, results, facts, {"model": "Qwen/Qwen2.5-0.5B-Instruct"})
    path = cf.write(tmp_path / "certs", rec)
    again = cf.write(tmp_path / "certs", rec)
    assert path.name == "vllm-0.33.0.json" and again.name == "vllm-0.33.0-2.json"  # never written over
    (loaded,) = json.loads(path.read_text(encoding="utf-8"))
    tested = engines.Tested.model_validate(loaded)
    assert tested.engine == "vllm" and tested.version == "0.33.0" and tested.card == "NVIDIA GeForce RTX 4090"
    assert tested.checks == {"C01 the engine probe answers every question and agrees with the source checks": "pass",
                             "C11 fp8 weights start and serve on this card": "pass"}
    assert loaded["certified"] is False  # C02 to C05 were not run
    assert loaded["details"]["C11"]["data"]["ready"] is True and loaded["probe"]["provenance"].startswith("source")
    assert "python" not in json.dumps(loaded["probe"]).lower().split("provenance")[0]  # no interpreter path


def _path_forms(*paths):
    """Each path as it could appear in a file: native, POSIX, and JSON-escaped (doubled backslashes)."""
    out = set()
    for p in paths:
        for form in (str(p), Path(p).as_posix()):
            out |= {form, json.dumps(form)[1:-1]}
    return out


def test_the_record_names_no_folder_of_the_user(run_for, tmp_path):
    """A start-up error quotes the engine's log, a launch that raises names a file, the probe's errors name modules'
    files and a model can be a local folder: in the written record each is cut to its last part, whichever system's
    form it has."""
    import site

    run, facts, hw = run_for("vllm", "0.33.0", only=["C11"])
    home, env = Path.home(), Path(sys.executable).parent
    packages = Path(site.getsitepackages()[-1])
    windows = "C:\\Users\\alice\\venv\\Lib\\site-packages\\vllm\\config\\cache.py"
    posix = "/home/alice/.venv/lib/python3.12/site-packages/vllm/config/cache.py"
    results = list(run.run()) + [
        cf.CheckResult("C02", "a stock launch", "fail", f"did not start: FileNotFoundError: {env / 'vllm'}",
                       {"error": f"ImportError: cannot import name 'X' from 'vllm' ({packages / 'vllm' / 'x.py'})",
                        "launches": {"a": {"ready": False, "error": f'File "{windows}", line 3'},
                                     "b": {"ready": False, "errors": [f"{posix}: no such file", str(home / "m")]}}}),
    ]
    probed = facts.model_copy(update={"errors": {"kv_dtypes": f"ImportError: ({posix})", "archs": f"at {windows}"}})
    rec = cf.record("vllm", "0.33.0", hw, results, probed, {"model": str(tmp_path / "models" / "my-model"),
                                                            "hybrid": posix.rsplit("/", 2)[0]})
    text = cf.write(tmp_path / "certs", rec).read_text(encoding="utf-8")
    forms = _path_forms(tmp_path, home, env, packages, windows, posix) | {"alice", "site-packages"}
    assert sorted(f for f in forms if f in text) == []
    assert "model my-model" in rec["source"] and "cache.py" in text  # the last part stays


def test_certified_needs_c01_to_c05_passed_and_c07_not_failed():
    def r(i, s):
        return cf.CheckResult(i, "", s)

    base = [r(c, "pass") for c in cf.CERTIFYING]
    assert cf.certified(base + [r("C07", "pass")]) and cf.certified(base + [r("C07", "skip")])
    assert not cf.certified(base + [r("C07", "fail")]) and not cf.certified(base)
    rest = [x for x in base if x.id != "C05"] + [r("C07", "pass")]
    assert not cf.certified(rest + [r("C05", "skip")])  # skipped for another reason than not applying
    assert cf.certified(rest + [cf.CheckResult("C05", "", "skip", "", dict(cf.NOT_APPLICABLE))])
    assert not cf.certified(rest + [cf.CheckResult("C05", "", "fail", "", dict(cf.NOT_APPLICABLE))])


def test_the_command_refuses_without_an_nvidia_gpu(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from rigtune import cli
    from tests.conftest import make_hw

    monkeypatch.setattr("rigtune.hardware.probe", lambda: make_hw("cpu"))
    out = CliRunner().invoke(cli.app, ["conformance", "vllm", "--out", str(tmp_path)])
    assert out.exit_code == 2 and "needs an NVIDIA GPU" in out.output
    out = CliRunner().invoke(cli.app, ["conformance", "llamacpp-cuda", "--out", str(tmp_path)])
    assert out.exit_code == 2


def test_the_command_runs_the_checks_and_writes_the_record(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from rigtune import cli
    from tests.conftest import make_hw

    hw = make_hw("rtx4090")
    seen = {}

    class FakeRun:
        def __init__(self, engine, backend, hw, models, out_dir, **kw):
            seen.update(engine=engine, models=sorted(models), only=kw.get("only"))

        def run(self, progress=None):
            results = [cf.CheckResult(c, f"check {c}", "pass") for c in cf.CERTIFYING + ("C07",)]
            for r in results:
                progress(r)
            return results

    monkeypatch.setattr("rigtune.hardware.probe", lambda: hw)
    monkeypatch.setattr(cf, "Conformance", FakeRun)
    monkeypatch.setattr(cli, "_conformance_models",
                        lambda backend, hw, roles: {role: object() for role, hf_id in roles.items() if hf_id})
    out = CliRunner().invoke(cli.app, ["conformance", "vllm", "--out", str(tmp_path), "--hybrid-model",
                                       "Qwen/Qwen3.5-9B", "--only", "C01,C02"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert seen == {"engine": "vllm", "models": ["hybrid", "small"], "only": ["C01", "C02"]}
    assert "certified" in out.output
    (rec,) = json.loads((tmp_path / "vllm-1.0.json").read_text(encoding="utf-8"))
    assert rec["certified"] is True and rec["source"].startswith("rigtune conformance")
