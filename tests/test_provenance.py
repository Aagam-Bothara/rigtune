"""What a scored comparison ran on, recorded in its results and its verdict."""

from __future__ import annotations

import getpass
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rigtune.bench import compare
from rigtune.bench.compare import provenance, rigtune_commit
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.conftest import make_hw
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


def test_provenance_names_versions_and_hashes_but_no_paths(tmp_path):
    hw = make_hw("a100")
    hw.gpus[0] = hw.gpus[0].model_copy(update={"driver_version": "595.91.07", "cuda_version": "13.0",
                                               "uuid": "GPU-1234"})
    hw.backends["vllm"] = hw.backends["vllm"].model_copy(update={"libs": {"transformers": "5.23.0",
                                                                          "torch": "2.13.0"}})
    prompts = tmp_path / "private" / "dolly.jsonl"
    prompts.parent.mkdir()
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    p = provenance(hw, {"calibration_prompts": prompts, "measured_prompts": None})
    assert p["gpus"] == [{"name": "NVIDIA A100-SXM4-80GB", "driver_version": "595.91.07",
                          "cuda_version": "13.0", "vram_total_bytes": 80 * 2**30}]  # no uuid
    assert p["engines"]["vllm"] == {"version": "1.0", "libs": {"torch": "2.13.0", "transformers": "5.23.0"}}
    assert set(p["engines"]) == {"vllm", "sglang", "llamacpp-cuda", "llamacpp-cpu"}  # the available ones
    assert p["files"] == {"calibration_prompts": {
        "name": "dolly.jsonl", "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest()}}
    assert "private" not in json.dumps(p) and str(tmp_path) not in json.dumps(p)
    assert (p["os"], p["python"]) == ("Linux test", "3.12")


def _fake_git(monkeypatch, top, head="0123abc", status=""):
    def run(cmd, **kw):
        args = cmd[3:]
        answers = {("rev-parse", "--show-toplevel"): str(top), ("rev-parse", "HEAD"): head}
        out = answers.get(tuple(args), status)
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)


def test_the_commit_is_recorded_only_from_a_checkout_of_rigtune_itself(monkeypatch, tmp_path):
    package = Path(compare.__code__.co_filename).resolve().parent.parent
    _fake_git(monkeypatch, package.parent)
    assert rigtune_commit() == "0123abc"
    _fake_git(monkeypatch, package.parent, status=" M rigtune/cli.py")
    assert rigtune_commit() == "0123abc+dirty"
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-dotfiles"\n', encoding="utf-8")
    _fake_git(monkeypatch, tmp_path)  # installed inside some other repository, a home directory say
    assert rigtune_commit() is None

    def no_git(cmd, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert rigtune_commit() is None


@pytest.mark.usefixtures("no_network")
def test_compare_records_the_provenance_and_each_rows_launch_environment(hw_a100, spec, monkeypatch):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    plain = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner())
    dumped = json.loads(plain.model_dump_json())
    assert "provenance" not in dumped and all("launch_env" not in r for r in dumped["rows"])
    fp8 = profile.config.model_copy(update={"kv_dtype": "fp8_e5m2"})  # it pins an attention backend
    origin = provenance(hw_a100, {})
    res = compare(hw_a100, spec, profile.model_copy(update={"config": fp8}), planned.prepared, reg,
                  runner=CompareRunner(), origin=origin)
    assert res.provenance == origin
    assert res.rigtune_row.launch_env == {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}


HOME, USER = Path.home(), getpass.getuser()
LEAK = ("/home/alice/.venv/lib/python3.12/site-packages/vllm/entrypoints/serve.py",
        f"{Path(sys.executable).parent}{os.sep}site-packages{os.sep}x.py",
        f"{HOME}{os.sep}models{os.sep}model.gguf")


def assert_no_local_paths(folder: Path, *also: Path) -> None:
    """No file a scored comparison wrote names this machine's folders: not the home folder or the user name,
    not the interpreter's folder, not site-packages, not the test's own folder, in Windows or POSIX form,
    raw or as JSON escapes a backslash."""
    places = [HOME, Path(sys.executable).parent, *also]
    needles = {"site-packages", f"/{USER}/", f"\\{USER}\\", f"\\\\{USER}\\\\"}
    for place in places:
        needles |= {str(place), place.as_posix(), str(place).replace("\\", "\\\\")}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            found = [n for n in needles if n in text]
            assert not found, f"{path.name} names {found}"


class LeakyRunner(CompareRunner):
    """A runner whose engine leaks paths everywhere it can: a failed row's traceback, and a failed request's
    error passed to on_level, as a real engine's would."""

    def __init__(self, on_level=None):
        super().__init__()
        self.on_level = on_level

    def run(self, cfg, stage):
        from rigtune.calibrate.measure import RequestOutcome
        from rigtune.models import TrialMetrics, TrialResult

        if self.on_level is not None:
            self.on_level(8, [RequestOutcome(ok=False, error=f"connection reset by {LEAK[1]}")])
        if stage == "compare:vllm-fp8-default":
            error = f'Traceback: File "{LEAK[0]}", line 3\nloading {LEAK[2]} failed'
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(errors=[error]), error=error)
        return super().run(cfg, stage)


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_a_scored_comparison_carries_provenance_and_its_verdict_the_results_hash(tmp_path, monkeypatch,
                                                                                hw_a100):
    from rigtune.cli import app
    from tests.test_extra_rows import MODEL, _cli

    _cli(monkeypatch, hw_a100)
    cmp = importlib.import_module("rigtune.bench.compare")
    monkeypatch.setattr(cmp, "rigtune_commit", lambda: "0123abc")
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: LeakyRunner(kw.get("on_level")))
    prompts = tmp_path / "held-out.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": f"Question {i}?"}) + "\n" for i in range(24)),
                       encoding="utf-8")
    out = tmp_path / "out"
    args = ["compare", MODEL, "--backend", "vllm", "--repeats", "1", "--verdict", "--eval-workload-file",
            str(prompts), "--out", str(out)]
    monkeypatch.setattr("sys.argv", [str(tmp_path / "bin" / "rigtune"), *args])  # what the shell passed
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    (results,) = [p for p in out.iterdir() if p.name.endswith("balanced.json")]
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["provenance"]["rigtune_commit"] == "0123abc"
    assert data["provenance"]["files"] == {"measured_prompts": {
        "name": "held-out.jsonl", "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest()}}
    stem = results.name.removesuffix(".json")
    assert data["provenance"]["inputs"] == {"measured_prompts": f"{stem}.inputs/held-out.jsonl"}
    assert (out / f"{stem}.inputs" / "held-out.jsonl").read_bytes() == prompts.read_bytes()
    assert data["provenance"]["options"]["repeats"] == 1 and data["provenance"]["model"]["hf_id"] == MODEL
    assert data["provenance"]["workload"]["temperature"] == 0.0
    assert data["provenance"]["command"] == ["rigtune", *args[:-3], "held-out.jsonl", "--out", "out"]
    assert data["workload_spec"]["source"] == "file:held-out.jsonl"  # named, not located
    verdict = json.loads(next(out.glob("*.verdict.json")).read_text(encoding="utf-8"))
    fp8 = next(r for r in data["rows"] if r["label"] == "vllm-fp8-default")
    # The engine's error, still readable, with its folders cut out.
    assert fp8["error"].startswith('Traceback: File "<path>/serve.py", line 3')
    assert "<path>/model.gguf" in fp8["error"]
    assert (out / f"{stem}.requests").is_dir()
    assert_no_local_paths(out, tmp_path)
    assert verdict["results_sha256"] == hashlib.sha256(results.read_bytes()).hexdigest()
    assert verdict["provenance"] == data["provenance"]
    md = next(out.glob("*.verdict.md")).read_text(encoding="utf-8")
    assert "rigtune" in md and "(0123abc)" in md and f"sha256 {verdict['results_sha256']}" in md


def test_untracked_files_make_the_checkout_dirty_and_the_changes_are_hashed(monkeypatch, tmp_path):
    package = Path(compare.__code__.co_filename).resolve().parent.parent
    top = package.parent
    _fake_git(monkeypatch, top, status="?? rigtune/new_module.py")
    assert rigtune_commit() == "0123abc+dirty"  # a file git does not track yet still changes what ran

    def run(cmd, **kw):
        args = tuple(cmd[3:])
        out = {("rev-parse", "--show-toplevel"): str(top), ("ls-files", "--others", "--exclude-standard"): "",
               ("diff", "HEAD", "--stat"): " rigtune/cli.py | 1 +"}.get(args, "")
        if args == ("diff", "HEAD", "--binary"):
            out = b"diff --git a/rigtune/cli.py b/rigtune/cli.py\n+x\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=b"")
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert compare_module().rigtune_diff_sha256() == hashlib.sha256(
        b"diff --git a/rigtune/cli.py b/rigtune/cli.py\n+x\n").hexdigest()


def compare_module():
    return importlib.import_module("rigtune.bench.compare")


@pytest.mark.parametrize("arg, cut", [
    ("/home/me/rows.json", "rows.json"),  # on Windows too, and whether or not it exists
    ("--rows=/home/me/rows.json", "--rows=rows.json"),
    (r"C:\Users\me\x.json", "x.json"),  # on POSIX too: split on either separator
    ("C:/Users/me/x.json", "x.json"),
    (r"--rows=C:\Users\me\x.json", "--rows=x.json"),
    (r"\\server\share\me\x.json", "x.json"),
    ("-o/home/me/out", "-oout"),  # a short flag with its value attached
    (r"-oC:\Users\me\out", "-oout"),
    ("~/prompts/dolly.jsonl", "dolly.jsonl"),
    ("/home/alice/models/qwen/", "qwen"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),  # a Hub id
    ("at/rows.json", "at/rows.json"),  # relative: it names no folder above the one it ran in
    ("--repeats", "--repeats"), ("3", "3"), ("-p", "-p"), ("no:warnings", "no:warnings"),
])
def test_a_command_line_keeps_file_names_and_hub_ids_but_no_paths(arg, cut):
    from rigtune.bench.compare import name_only

    assert name_only(arg) == cut


def test_engine_errors_keep_their_message_without_the_folders():
    from rigtune.bench.compare import scrub_text

    text = ('Traceback: File "/home/alice/.venv/lib/python3.12/site-packages/vllm/x.py", line 3; '
            r'C:\Users\bob\site-packages\y.py failed (see /tmp/run/x.log); POST /v1/chat/completions '
            'to http://127.0.0.1:8000/v1/models; file:/home/a/p.jsonl; ~/m/q')
    assert scrub_text(text) == ('Traceback: File "<path>/x.py", line 3; <path>/y.py failed '
                                '(see <path>/x.log); POST /v1/chat/completions to '
                                'http://127.0.0.1:8000/v1/models; file:<path>/p.jsonl; <path>/q')


def test_provenance_under_verdict_records_weights_options_workload_and_added_rows(tmp_path, monkeypatch):
    import huggingface_hub

    from rigtune.calibrate.workload import Workload
    from rigtune.models import ModelSpec

    cached = tmp_path / "hub" / "models--Qwen--Q" / "snapshots" / ("c0ffee" * 6 + "abcd") / "config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, name, revision=None: str(cached))
    rows = tmp_path / "rows.json"
    rows.write_text('{"rows": [{"label": "autotune-vllm", "skipped": "no pick"}]}', encoding="utf-8")
    prompts = tmp_path / "dolly.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    wl = Workload(name="file-dolly", source=f"file:{prompts}", seed=7, natural_stop=True)
    model = ModelSpec(hf_id="Qwen/Q", revision="v1")
    p = provenance(make_hw("a100"), {"measured_prompts": prompts}, model=model, workload=wl,
                   options={"repeats": 3, "include": []},
                   command=["/home/someone/.venv/bin/rigtune", "compare", "Qwen/Q", "--rows", str(rows)],
                   rows=[rows], expert=True, inputs={"measured_prompts": "r.inputs/dolly.jsonl"},
                   logs="Qwen__Q/c")
    assert p["model"] == {"hf_id": "Qwen/Q", "revision": "v1", "snapshot": "c0ffee" * 6 + "abcd"}
    assert p["workload"] == {"name": "file-dolly", "source": "file:dolly.jsonl", "seed": 7,
                             "temperature": 0.0, "natural_stop": True,
                             "prompts": "a copy of the file is beside the results"}
    assert p["command"] == ["rigtune", "compare", "Qwen/Q", "--rows", "rows.json"]
    assert p["options"] == {"repeats": 3, "include": []}
    assert p["rows_files"] == [{"name": "rows.json", "sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
                                "content": {"rows": [{"label": "autotune-vllm", "skipped": "no pick"}]}}]
    shipped = Path(compare.__code__.co_filename).resolve().parent.parent / "data" / "expert_rows.json"
    assert p["expert_rows"]["sha256"] == hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert p["expert_rows"]["content"]["rows"][0]["label"] == "vllm-expert-batched-tokens"
    assert p["inputs"] == {"measured_prompts": "r.inputs/dolly.jsonl"}
    assert str(tmp_path) not in json.dumps(p) and "someone" not in json.dumps(p)
    synthetic = provenance(make_hw("a100"), {}, workload=Workload(name="chat", seed=3))
    assert synthetic["workload"]["prompts"].startswith("rigtune.calibrate.datasets rebuilds them from "
                                                       "`synthetic` with seed 3")


def test_a_sampled_workloads_answers_are_unchecked_not_unchanged():
    from rigtune.bench.verdict import score
    from tests.test_answer_stats import _res, _row as answered

    res = _res(answered("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0)),
               answered("vllm-default", "vllm", 100))
    assert score(res).rigtune["workload_answers"] == "same"
    res.provenance = {"workload": {"temperature": 0.7}}
    assert score(res).rigtune["workload_answers"] == ("unknown: the workload was sampled at temperature 0.7, "
                                                      "not greedily")


def test_inputs_are_copied_beside_the_results_and_names_never_reused(tmp_path):
    from rigtune.bench.compare import copy_inputs, free_results_path

    target = tmp_path / "r.json"
    a, b = tmp_path / "a" / "p.jsonl", tmp_path / "b" / "p.jsonl"
    for f, text in ((a, "1\n"), (b, "2\n")):
        f.parent.mkdir()
        f.write_text(text, encoding="utf-8")
    copied = copy_inputs(target, {"calibration_prompts": a, "measured_prompts": b, "other": None})
    assert copied == {"calibration_prompts": "r.inputs/p.jsonl",
                      "measured_prompts": "r.inputs/measured_prompts-p.jsonl"}
    assert (tmp_path / "r.inputs" / "measured_prompts-p.jsonl").read_text(encoding="utf-8") == "2\n"
    assert free_results_path(target) == tmp_path / "r-2.json"  # the copies alone keep the name taken


def test_gpu_state_reads_power_limit_and_temperature_and_never_fails(monkeypatch):
    from rigtune.bench.compare import gpu_state

    def smi(cmd, **kw):
        assert "uuid" not in " ".join(cmd)
        out = "0, 300.00, 41, 1740\n1, [N/A], 38, 210\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", smi)
    assert gpu_state() == [
        {"index": 0, "power_limit_w": 300.0, "temperature_c": 41.0, "sm_clock_mhz": 1740.0},
        {"index": 1, "power_limit_w": None, "temperature_c": 38.0, "sm_clock_mhz": 210.0}]

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert gpu_state() == []


@pytest.mark.usefixtures("no_network")
def test_each_run_records_the_gpus_at_its_start_its_engine_logs_and_its_requests(hw_a100, spec, monkeypatch,
                                                                                tmp_path):
    from rigtune.calibrate.measure import RequestOutcome

    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    logs = tmp_path / "logs"
    logs.mkdir()

    class Runner(CompareRunner):  # writes a log per run and hands each level's requests to on_level
        def __init__(self, on_level=None, **kw):
            super().__init__()
            self.on_level, self.n = on_level, 0

        def run(self, cfg, stage):
            self.n += 1
            (logs / f"{self.n}_{stage.split(':')[1]}.log").write_text("x", encoding="utf-8")
            self.on_level(8, [RequestOutcome(ok=True, ttft_s=0.05, duration_s=1.0, tokens=12,
                                             prompt_tokens=30, finish_reason="stop"),
                              RequestOutcome(ok=False, error="boom\nmore")])
            return super().run(cfg, stage)

    cmp = compare_module()
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: Runner(on_level=kw["on_level"]))
    temps = iter(range(40, 60))
    res = compare(hw_a100, spec, profile, planned.prepared, reg, repeats=2, record_runs=True, log_dir=logs,
                  gpu_probe=lambda: [{"index": 0, "temperature_c": float(next(temps))}],
                  requests_dir=tmp_path / "r.requests")
    rig = res.rigtune_row.run_records
    assert [r.gpu_state[0]["temperature_c"] for r in rig] == [40.0, 43.0]  # 3 rows a round
    assert [r.engine_logs for r in rig] == [["1_rigtune.log"], ["4_rigtune.log"]]
    lines = (tmp_path / "r.requests" / "rigtune.r2.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [
        {"level": 8, "request": 0, "ok": True, "ttft_ms": 50.0, "e2e_ms": 1000.0, "prompt_tokens": 30,
         "tokens": 12, "finish_reason": "stop", "error": None},
        {"level": 8, "request": 1, "ok": False, "ttft_ms": 0.0, "e2e_ms": 0.0, "prompt_tokens": None,
         "tokens": 0, "finish_reason": None, "error": "boom"}]
    plain = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner())
    assert "gpu_state" not in json.dumps(json.loads(plain.model_dump_json()))  # nothing new without --verdict


@pytest.mark.usefixtures("no_network")
def test_a_local_model_folder_is_named_not_located_in_a_scored_comparison(hw_a100, spec, tmp_path):
    """--model /home/alice/models/qwen: under --verdict the results, the provenance and every launch argument
    name the folder `qwen`, and the results file is named after it; without --verdict nothing changes."""
    from rigtune.bench.compare import name_only, results_path
    from rigtune.models import ModelSpec

    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    folder = "/home/alice/models/qwen"
    local = ModelSpec(hf_id=folder)
    moved = {"spec": local, "hf_path": folder, "hf_paths": {}}  # the same model, from a local folder
    prepared = {k: m.model_copy(update=moved) for k, m in planned.prepared.items()}
    profile = profile.model_copy(update={"prepared": profile.prepared.model_copy(update=moved)})
    res = compare(hw_a100, local, profile, prepared, reg, runner=CompareRunner(),
                  origin=provenance(hw_a100, {}, model=local))
    assert res.model_id == "qwen" and res.provenance["model"]["hf_id"] == "qwen"
    stock = next(r for r in res.rows if r.label == "vllm-default")
    assert "alice" not in res.model_dump_json() and "qwen" in stock.launch_args
    named = results_path("h", ModelSpec(hf_id=name_only(folder)), "chat", "balanced", tmp_path)
    assert "alice" not in named.name and "qwen" in named.name
    plain = compare(hw_a100, local, profile, prepared, reg, runner=CompareRunner())
    stock = next(r for r in plain.rows if r.label == "vllm-default")
    assert plain.model_id == folder and folder in stock.launch_args  # default output as before
