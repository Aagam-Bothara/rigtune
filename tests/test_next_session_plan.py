"""benchmarks/next_session_plan.py: the next GPU session's phases, as benchmarks/gpu_session.py runs them.

No phase here reaches a GPU or the network. Most checks are on what each phase would run: that the comparison
reuses the calibration (same search options), that every tuner gets the same model, limits and lengths, that
every rigtune command parses with rigtune's own CLI. Where a bash runs, the phases whose job is to refuse a bad
input are run as the runner runs them, with that input.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"
M, MODEL = "qwen3.5-9b", "Qwen/Qwen3.5-9B"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gs = _load("gpu_session")
nsp = _load("next_session_plan")


def _plan(tmp: Path, *args: str) -> dict:
    out = tmp / "plan.json"
    assert nsp.main(["--out", str(out), *args]) == 0
    assert b"\r\n" not in out.read_bytes()
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return _plan(tmp_path_factory.mktemp("plan"), "--commit", "0123abc")


@pytest.fixture(scope="module")
def local_plan(tmp_path_factory):
    """The plan with this checkout as the source tree, so a phase's helper scripts can really run."""
    return _plan(tmp_path_factory.mktemp("local"), "--commit", "0123abc", "--repo", REPO.as_posix(),
                 "--models", MODEL)


def by_name(plan):
    return {p["name"]: p for p in plan["phases"]}


def _bash_works() -> bool:
    """A bash that runs (on Windows, `bash` can be WSL's launcher with no distribution behind it)."""
    if shutil.which("bash") is None:
        return False
    try:
        return subprocess.run(["bash", "-c", "true"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bash = pytest.mark.skipif(
    not _bash_works() or not gs.SHELL_SAFE.match(Path(sys.executable).as_posix()),
    reason="needs a working bash (CI's Linux has one) and an interpreter path a shell does not split")


def run_as_the_runner(plan: dict, name: str, root: Path, key: str = "cmd", env=None):
    """One phase's command (or its requirement), substituted and started exactly as gpu_session.py does."""
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    subs = {"out": out.as_posix(), "root": root.as_posix(), "python": Path(sys.executable).as_posix()}
    got = subprocess.run(gs._argv(by_name(plan)[name][key], subs), capture_output=True, text=True, cwd=out,
                         env={**os.environ, **(env or {})}, timeout=300)
    return got, out


def test_the_plan_is_one_the_runner_accepts_in_the_intended_order(plan):
    gs.read_plan(plan)
    names = [p["name"] for p in plan["phases"]]
    assert names[:5] == ["env", "hub-access", "prompts", "selfcheck", "smoke"]
    for m in ("qwen3.5-9b", "gemma-4-12b-it"):
        mine = [n for n in names if n.endswith(m)]
        assert mine == [f"drafts-vllm-{m}", f"drafts-sglang-{m}", f"cal-{m}", f"ledger-{m}", f"lengths-{m}", f"autotune-{m}", f"llmopt-vllm-{m}",
                        f"llmopt-sglang-{m}", f"aic-vllm-{m}", f"aic-sglang-{m}", f"cmp-{m}"]
    assert plan["home"] == "rigtune-home" and plan["env"] == {"HF_HOME": "/workspace/hf"}
    # The prompt files are written into the prompts phase's own directory, so they are in its copy, and every
    # later phase reads them from there.
    p = by_name(plan)
    assert "--out {root}/prompts/dolly.jsonl" in p["prompts"]["cmd"] and "cp " not in p["prompts"]["cmd"]
    assert "--workload-file {root}/prompts/dolly.jsonl" in p["cal-qwen3.5-9b"]["cmd"]
    assert "--eval-workload-file {root}/prompts/dolly-heldout.jsonl" in p["cmp-qwen3.5-9b"]["cmd"]


def test_the_comparison_passes_the_calibrations_search_options(plan):
    """Otherwise compare would miss the cached profile and calibrate again for an hour."""
    p = by_name(plan)
    for model, m in (("Qwen/Qwen3.5-9B", "qwen3.5-9b"), ("google/gemma-4-12B-it", "gemma-4-12b-it")):
        cal = p[f"cal-{m}"]["cmd"].split(f"recalibrate {model} ", 1)[1]
        cmp_ = p[f"cmp-{m}"]["cmd"].split(f"compare {model} ", 1)[1]
        assert cmp_.startswith(cal + " --eval-workload-file ")
        assert "--speculative all" in cal and "--budget 60m" in cal and "--ttft-ceiling 500" in cal
        for flag in ("--repeats 3", "--expert", "--bare-stock", "--verdict", "--check-answers gsm8k"):
            assert flag in cmp_
        # A competitor that fails never blocks the comparison.
        assert p[f"cmp-{m}"]["needs"] == ["env", f"cal-{m}"]


def test_every_tuner_gets_the_same_model_limits_and_lengths(plan):
    p = by_name(plan)
    lengths = "{root}/lengths-qwen3.5-9b/lengths.env"
    for name in (f"autotune-{M}", f"llmopt-vllm-{M}", f"llmopt-sglang-{M}", f"aic-vllm-{M}", f"aic-sglang-{M}"):
        cmd = p[name]["cmd"]
        assert f"--model {MODEL}" in cmd and f". {lengths}" in cmd and p[name]["needs"] == ["env", f"lengths-{M}"]
        assert "--input-len $INPUT_LEN --output-len $OUTPUT_LEN" in cmd
        assert ("--limit-ms 500" in cmd) if name.startswith("autotune") else ("--ttft-ms 500 --tpot-ms 50" in cmd)
    for name in (f"llmopt-vllm-{M}", f"llmopt-sglang-{M}"):
        assert "--levels 1,4,8" in p[name]["cmd"] and "--max-model-len 4096" in p[name]["cmd"]
        assert "timeout 3600 " in p[name]["cmd"]  # the calibration's budget, and no more
    assert "$SGLANG_PYTHON -m sglang.launch_server" in p[f"llmopt-sglang-{M}"]["cmd"]


def test_every_competitor_phase_leaves_a_row_and_the_comparison_reads_them_all(plan):
    p = by_name(plan)
    m = "gemma-4-12b-it"
    for name in (f"autotune-{m}", f"llmopt-vllm-{m}", f"llmopt-sglang-{m}", f"aic-vllm-{m}", f"aic-sglang-{m}"):
        cmd = p[name]["cmd"]
        assert "competitor_rows.py not-applicable --out {out}/rows.json" in cmd
    cmp_ = p[f"cmp-{m}"]["cmd"]
    for name in ("autotune", "llmopt-vllm", "llmopt-sglang", "aic-vllm", "aic-sglang"):
        assert f"{{root}}/{name}-{m}/rows.json" in cmp_
    assert "ROW MISSING" in cmp_ and "{out}/rows-missing.txt" in cmp_  # a dropped row is named, not silent


def test_every_phase_waits_for_env_which_checks_the_installed_rigtune(plan):
    for p in plan["phases"]:
        assert (p["name"] == "env") != ("env" in p.get("needs", [])), p["name"]
    env = by_name(plan)["env"]["cmd"]
    assert "WANT=0123abc;" in env and "rigtune.__file__" in env and "exit 1" in env and not env.endswith("true")


def test_a_commit_that_is_not_a_hash_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        nsp.main(["--out", str(tmp_path / "p.json"), "--commit", "main; rm -rf x"])


@needs_bash
def test_env_fails_unless_the_installed_rigtune_is_the_plans_commit(local_plan, tmp_path):
    """The reviewed plan never installed rigtune on the pod and env ended in `true`."""
    src = tmp_path / "src"
    (src / "rigtune").mkdir(parents=True)
    (src / "rigtune" / "__init__.py").write_text("", encoding="utf-8")
    env = {"PYTHONPATH": str(src)}  # the rigtune the phase's interpreter imports
    (src / "COMMIT").write_text("fffffff\n", encoding="utf-8")
    got, out = run_as_the_runner(local_plan, "env", tmp_path / "s1", env=env)
    assert got.returncode == 1 and "is not the source at the plan's commit (0123abc; it reports 'fffffff')" in got.stdout
    (src / "COMMIT").write_text("0123abc\n", encoding="utf-8")
    got, out = run_as_the_runner(local_plan, "env", tmp_path / "s2", env=env)
    assert got.returncode == 0, got.stdout + got.stderr
    assert (out / "rigtune-commit.txt").read_text().strip() == "plan: 0123abc installed: 0123abc"
    (src / "COMMIT").unlink()  # a rigtune from somewhere else: no commit beside it
    assert run_as_the_runner(local_plan, "env", tmp_path / "s3", env=env)[0].returncode == 1


def test_a_plan_without_a_commit_makes_env_fail(tmp_path):
    env = by_name(_plan(tmp_path, "--repo", str(tmp_path / "nowhere")))["env"]["cmd"]
    assert "WANT=unknown;" in env and '[ "$WANT" = unknown ] ||' in env


def test_every_rigtune_command_in_the_plan_parses_with_rigtunes_own_cli(plan):
    """A phase must not name a command or pass an option this rigtune does not have (the reviewed plan's hooks
    called `probe-engines`, which never existed)."""
    import typer

    from rigtune import cli as rcli

    group = typer.main.get_command(rcli.app)
    subs = {"{out}": "/r/x", "{root}": "/r", "{python}": "py", "$INPUT_LEN": "100", "$OUTPUT_LEN": "100",
            "$G4_JPT": "0.5", "$RIGTUNE_HOME": "/r/h", "$ROWS": "--rows /r/a/rows.json", '"$P"': "/r/p.json",
            '"$ID"': "abcdef"}
    parsed = []
    for p in plan["phases"]:
        for key in ("cmd", "requires"):
            text = p.get(key) or ""
            for k, v in subs.items():
                text = text.replace(k, v)
            for found in re.finditer(r"py -m rigtune\.cli (.+?)(?=; | && | > | \| | \|\| |$)", text):
                argv = shlex.split(found.group(1))
                cmd, used = group, []
                # By what the objects offer, not by class: typer may bring its own copy of click.
                while hasattr(cmd, "get_command"):
                    sub = cmd.get_command(None, argv[0])
                    assert sub is not None, f"{p['name']}: rigtune has no command {' '.join(used + argv[:1])}"
                    cmd, used, argv = sub, used + argv[:1], argv[1:]
                try:
                    cmd.make_context(used[-1], argv, resilient_parsing=False)
                except Exception as exc:  # noqa: BLE001 - told apart below
                    if type(exc).__name__ != "Exit":  # Exit: --help printed
                        said = exc.format_message() if hasattr(exc, "format_message") else repr(exc)
                        pytest.fail(f"{p['name']}: rigtune {' '.join(used)}: {said}")
                parsed.append(" ".join(used))
    for name in ("probe", "conformance", "ledger verify", "ledger rebuild", "serve", "recalibrate", "compare",
                 "tradeoffs", "power status"):
        assert name in parsed, name


def test_probe_and_conformance_are_skipped_only_where_their_options_are_missing(plan):
    p = by_name(plan)
    assert "probe --engines --reprobe --json > {out}/probe-engines.json" in p["engine-probe"]["cmd"]
    assert p["engine-probe"]["requires"].endswith("probe --help | grep -e --engines")
    for engine in ("vllm", "sglang"):
        ph = p[f"conformance-{engine}"]
        assert ph["cmd"].endswith(f"conformance {engine} --out {{out}}/conformance")
        assert ph["requires"].endswith("conformance --help | grep -e --out")
    assert "RIGTUNE_SESSION_POWER" in p["power"]["requires"]
    # `grep -q` closes the pipe at its first match, which pipefail can turn into a failure.
    assert not any("grep -q" in (ph.get(k) or "") for ph in plan["phases"] for k in ("cmd", "requires"))


@needs_bash
def test_the_requirements_pass_on_this_rigtune(local_plan, tmp_path):
    for name in ("engine-probe", "conformance-vllm"):
        got, _ = run_as_the_runner(local_plan, name, tmp_path, key="requires")
        assert got.returncode == 0, (name, got.stdout, got.stderr)


def test_g2_calibrates_and_serves_under_the_same_cache_key(plan):
    """The budget is part of the key: the reviewed plan gave it to the calibration only, so serve could never
    find that profile, calibrated, and G2 read as refuted."""
    from rigtune.pipeline import SearchOptions

    p = by_name(plan)
    cal = p["g2-efficiency-cal"]["cmd"].split("recalibrate Qwen/Qwen2.5-7B-Instruct ", 1)[1]
    srv = p["g2-goal-is-preset"]["cmd"].split("serve Qwen/Qwen2.5-7B-Instruct ", 1)[1].split(" > ", 1)[0]
    options = "--workload chat --budget 20m"
    assert cal == options + " --objective efficiency"
    assert srv == options + " --optimise joules_per_token --limit 'tok_s>=50%best'"
    assert SearchOptions(budget_s=20 * 60).key("efficiency") != SearchOptions().key("efficiency")  # why it matters


@needs_bash
@pytest.mark.parametrize("log, code, says", [
    ("goal: joules_per_token, the efficiency\npreset\n-> probe\n-> cache hit\n", 0, "G2 confirmed"),
    ("goal: the efficiency preset\n-> probe\n-> select\n-> calibrate\n", 1, "serve calibrated"),
    ("goal: a custom goal, cached under its own name\n-> probe\n-> cache hit\n", 1, "did not name"),
    ("goal: the efficiency preset\nTraceback\n", 1, "did not report a cache hit"),
])
def test_g2_is_decided_from_serves_output_and_fails_when_it_calibrated(tmp_path, log, code, says):
    (tmp_path / "serve.log").write_text(log, encoding="utf-8")
    got = subprocess.run(gs._argv(nsp.g2_check("serve.log", "serve.flat"), {}), capture_output=True, text=True,
                         cwd=tmp_path)
    assert got.returncode == code and says in got.stdout


def test_g4_needs_no_second_runner(plan):
    """The reviewed G4 waited for `export G4_JPT`, which only a second runner on the same root could supply."""
    p = by_name(plan)
    assert "--pick-jpt" in p["g3-balanced"]["cmd"] and "{out}/g4_jpt.txt" in p["g3-balanced"]["cmd"]
    for name in ("g4-custom", "g4-custom-all-levels"):
        ph = p[name]
        assert "${G4_JPT" not in ph["requires"] and "export" not in ph["requires"]
        assert "{root}/G4_JPT {root}/g3-balanced/g4_jpt.txt" in ph["requires"]
        assert "{root}/G4_JPT {root}/g3-balanced/g4_jpt.txt" in ph["cmd"] and ph["needs"] == ["env", "g3-balanced"]
    assert "--all-levels" in p["g4-custom-all-levels"]["cmd"]


@needs_bash
def test_g4_is_skipped_until_a_limit_is_in_a_file_under_the_root(local_plan, tmp_path):
    got, _ = run_as_the_runner(local_plan, "g4-custom", tmp_path, key="requires")
    assert got.returncode == 1 and "no J/token limit for G4" in got.stdout
    (tmp_path / "g3-balanced").mkdir()
    (tmp_path / "g3-balanced" / "g4_jpt.txt").write_text("0.42\n", encoding="utf-8")
    assert run_as_the_runner(local_plan, "g4-custom", tmp_path, key="requires")[0].returncode == 0
    (tmp_path / "g3-balanced" / "g4_jpt.txt").unlink()
    (tmp_path / "G4_JPT").write_text("0.5\n", encoding="utf-8")  # the user's own value
    assert run_as_the_runner(local_plan, "g4-custom", tmp_path, key="requires")[0].returncode == 0


def _profile(jpt=0.31, prompt_tokens=63, output_tokens=3950) -> dict:  # tokens: a mean, and a total
    pick = {"backend": "vllm", "batch": 64}
    level = {"requests": 10, "failed": 0, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
    return {"config": pick, "calibration_id": "abc123", "calibration_table": [
        {"config": pick, "metrics": {"joules_per_token": 0.9, "by_concurrency": {"1": level}}},
        {"config": {"backend": "vllm", "batch": 8}, "metrics": {"joules_per_token": 0.1}},
        {"config": pick, "metrics": {"joules_per_token": jpt}}]}


def test_the_picks_joules_per_token_is_read_from_its_own_last_trial(tmp_path, capsys):
    assert nsp.pick_jpt(_profile()) == 0.31
    assert nsp.pick_jpt(_profile(jpt=None)) == 0.9 and nsp.pick_jpt({"config": {}, "calibration_table": []}) is None
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    assert nsp.main(["--pick-jpt", str(path)]) == 0 and capsys.readouterr().out.strip() == "0.31"
    path.write_text(json.dumps({"config": {}}), encoding="utf-8")
    assert nsp.main(["--pick-jpt", str(path)]) == 1


def _calibrated(root: Path, profile: dict) -> None:
    d = root / f"cal-{M}" / "rigtune-home" / "profiles" / "hw" / MODEL.replace("/", "__")
    d.mkdir(parents=True)
    (d / "balanced-chat.json").write_text(json.dumps(profile), encoding="utf-8")


@needs_bash
def test_lengths_fails_without_request_lengths_and_writes_them_otherwise(local_plan, tmp_path):
    """The reviewed phase piped into tee and exited 0 with an empty lengths.env."""
    got, out = run_as_the_runner(local_plan, f"lengths-{M}", tmp_path / "none")  # no calibration profile
    assert got.returncode != 0
    assert not (out / "lengths.env").exists() or not (out / "lengths.env").read_text().strip()
    _calibrated(tmp_path / "bare", {"config": {}, "calibration_table": []})  # a profile with no token counts
    assert run_as_the_runner(local_plan, f"lengths-{M}", tmp_path / "bare")[0].returncode != 0
    _calibrated(tmp_path / "good", _profile())
    got, out = run_as_the_runner(local_plan, f"lengths-{M}", tmp_path / "good")
    assert got.returncode == 0, got.stdout + got.stderr
    assert (out / "lengths.env").read_text().split() == ["INPUT_LEN=63", "OUTPUT_LEN=395"]


@needs_bash
def test_a_tuner_phase_given_empty_lengths_fails_and_still_leaves_a_row(local_plan, tmp_path):
    """The reviewed llm-optimizer phase ran an empty command, which exits 0, and was recorded done."""
    (tmp_path / f"lengths-{M}").mkdir()
    (tmp_path / f"lengths-{M}" / "lengths.env").write_text("", encoding="utf-8")
    got, out = run_as_the_runner(local_plan, f"llmopt-vllm-{M}", tmp_path)
    assert got.returncode != 0 and not (out / "command.txt").exists()
    row = json.loads((out / "rows.json").read_text(encoding="utf-8"))["rows"][0]
    assert row["label"] == "llm-optimizer-vllm" and "could not be written" in json.dumps(row)


@needs_bash
def test_an_auto_tune_run_that_writes_no_row_fails_with_a_row_saying_so(local_plan, tmp_path):
    (tmp_path / f"lengths-{M}").mkdir()
    (tmp_path / f"lengths-{M}" / "lengths.env").write_text("INPUT_LEN=63\nOUTPUT_LEN=395\n", encoding="utf-8")
    got, out = run_as_the_runner(local_plan, f"autotune-{M}", tmp_path)  # no vLLM checkout here: it cannot run
    assert got.returncode != 0
    row = json.loads((out / "rows.json").read_text(encoding="utf-8"))["rows"][0]
    assert row["label"] == "autotune-vllm" and "wrote no row" in json.dumps(row)


def test_the_ledger_is_checked_on_each_real_calibration(plan):
    cmd = by_name(plan)[f"ledger-{M}"]["cmd"]
    assert 'ledger verify "$P" > {out}/verify.txt' in cmd
    assert 'ledger rebuild "$ID" --out {out}/rebuilt.json && cmp "$P" {out}/rebuilt.json' in cmd
    assert cmd.endswith("[ $v = 0 ] && [ $r = 0 ]") and by_name(plan)[f"ledger-{M}"]["needs"] == ["env", f"cal-{M}"]


def test_the_power_phase_writes_into_its_own_directory_with_the_runners_python(plan):
    """The reviewed phase ran measure_power.sh as it was: results under the source tree, `rigtune` from PATH."""
    cmd = by_name(plan)["power"]["cmd"]
    assert 'OUT={out}/results RIGTUNE="{python} -m rigtune.cli" bash benchmarks/measure_power.sh' in cmd
    script = (BENCH / "measure_power.sh").read_text(encoding="utf-8")
    assert "OUT=${OUT:-benchmarks/strategies/results-power}" in script and "RIGTUNE=${RIGTUNE:-rigtune}" in script
    assert not re.search(r"^\s*(trap ')?rigtune ", script, flags=re.M)


@needs_bash
def test_measure_power_sends_every_command_and_result_where_it_is_told(tmp_path):
    got = subprocess.run(["bash", (BENCH / "measure_power.sh").as_posix(), "some/model", "rag"],
                         capture_output=True, text=True, env={**os.environ, "OUT": "there", "RIGTUNE": "echo ran"})
    assert got.returncode == 0, got.stderr
    lines = got.stdout.splitlines()
    assert lines[0] == "ran power status" and lines[-1] == "ran power reset"
    assert "ran compare some/model --workload rag --objective efficiency --power both --out there" in lines


@needs_bash
def test_every_shell_command_parses(plan):
    for p in plan["phases"]:
        for key in ("cmd", "requires"):
            if isinstance(p.get(key), str):
                got = subprocess.run(["bash", "-n", "-c", p[key]], capture_output=True, text=True)
                assert got.returncode == 0, (p["name"], key, got.stderr)


def test_a_dry_run_of_the_plan_lists_every_phase_pending(plan, tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert gs.main(["run", "--plan", str(path), "--root", str(tmp_path / "session"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("[pending]") == len(plan["phases"]) == 41 and not (tmp_path / "session").exists()


def test_a_root_that_shell_commands_would_split_is_refused(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"phases": [{"name": "a", "cmd": "echo {out}"}]}), encoding="utf-8")
    assert gs.main(["run", "--plan", str(path), "--root", str(tmp_path / "my session")]) == 2
    assert "unquoted" in capsys.readouterr().err and not (tmp_path / "my session").exists()
