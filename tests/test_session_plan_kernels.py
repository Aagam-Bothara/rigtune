"""The GPU session's kernel phases, and the checks that read their results.

A check answers confirmed, refuted or undecided, and only confirmed exits 0: an empty ledger, a log nothing was read
from, or a setting that was never tried is undecided, never a pass. Nothing here has run on a pod."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

from rigtune import ledger as lg

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
spec = importlib.util.spec_from_file_location("next_session_plan_k", BENCH / "next_session_plan.py")
nsp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nsp
spec.loader.exec_module(nsp)

K = ("k0-startup", "k1-rewrites", "k2-sglang-cap", "k3-vllm-triton", "k4-compile-cache")


@pytest.fixture(scope="module")
def phases(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("plan") / "plan.json"
    assert nsp.main(["--out", str(out), "--commit", "0123abc"]) == 0
    return {p["name"]: p for p in json.loads(out.read_text(encoding="utf-8"))["phases"]}


def test_plan_lists_k0_to_k4(phases):
    assert [name for name in phases if name in K] == list(K)


def test_kernel_phases_present_and_need_env(phases):
    for name in K:
        assert "env" in phases[name]["needs"] and phases[name]["why"]
    assert set(phases["k0-startup"]["needs"]) >= {"cal-qwen3.5-9b", "cal-gemma-4-12b-it"}
    for name in K[1:]:
        assert "k0-startup" in phases[name]["needs"]  # nothing is tried until the records are shown to be read
    for name in K[2:]:
        assert "recalibrate --help | grep -e --kernel" in phases[name]["requires"]  # skipped, not failed, without it
    assert "SGLANG_PYTHON" in phases["k2-sglang-cap"]["requires"]


def test_commands_parse_with_real_cli(phases):
    """Each phase's rigtune command is one this rigtune's own parser accepts, option by option."""
    from typer.main import get_command

    from rigtune.cli import app

    recalibrate = get_command(app).commands["recalibrate"]
    known = {opt for param in recalibrate.params for opt in param.opts}
    seen = 0
    for name in K[2:]:
        (call,) = [part for part in phases[name]["cmd"].split(";") if " recalibrate " in part and "--kernel" in part]
        words = shlex.split(call.split(" > ")[0])
        args = words[words.index("recalibrate") + 1:]
        options = [w for w in args if w.startswith("--")]
        assert options and set(options) <= known, set(options) - known
        assert args[options and args.index("--kernel") + 1] in ("cgmax=64", "attn=TRITON_ATTN", "O=1")
        seen += 1
    assert seen == 3


def test_no_phase_passes_a_local_path(phases):
    for name in K:
        text = phases[name]["cmd"] + (phases[name].get("requires") or "")
        assert "C:" not in text and "\\Users" not in text and "/home/" not in text and "/Users/" not in text


def test_no_grep_q_under_pipefail(phases):
    assert not any("grep -q" in (phases[name].get(k) or "") for name in K for k in ("cmd", "requires"))


# ---- check-startup

def _trial(startup=None, launched=True, engine="vllm", key="vllm/bf16/ctx4096/b64", spec_decode=None, version="0.33.0"):
    rec = {"schema": 1, "type": "trial", "calibration_id": "c1", "engine": {"name": engine, "version": version},
           "config_key": key, "outcome": {"kind": "ok", "launched": launched},
           "trial": {"config": {"backend": engine, "spec_decode": spec_decode}}}
    if startup is not None:
        rec["startup"] = startup
    return rec


def _ledger(tmp_path: Path, records: list) -> Path:
    path = tmp_path / "ledger" / "hw" / "model.jsonl"
    for i, rec in enumerate(records):
        rec = {**rec, "seq": i}
        rec["id"] = lg.record_id(rec)
        lg.append(path, rec)
    return tmp_path / "ledger"


READ = {"seconds": 90.0, "capture_s": 20.0, "cudagraph_gib": 0.7, "capture_lines": 1, "compile_s": 30.0,
        "compile_cached": False, "rewrites": [], "parsed": ["capture", "compile"]}
UNREAD = {"seconds": 90.0, "capture_s": None, "cudagraph_gib": None, "capture_lines": 0, "compile_s": None,
          "compile_cached": None, "rewrites": [], "parsed": []}


def test_check_startup_fails_on_empty_ledger(tmp_path, capsys):
    (tmp_path / "ledger").mkdir()
    assert nsp.check_startup([tmp_path / "ledger"]) == 1
    assert capsys.readouterr().out.startswith("undecided: no launched trial")
    skipped = _ledger(tmp_path, [_trial(launched=False)])
    assert nsp.check_startup([skipped]) == 1


def test_check_startup_fails_when_no_line_was_parsed(tmp_path, capsys):
    assert nsp.check_startup([_ledger(tmp_path, [_trial(READ), _trial(UNREAD)])]) == 2
    assert "1 of 2 ready vLLM trials had no graph-capture line read" in capsys.readouterr().out


def test_check_startup_confirms_only_what_was_read(tmp_path, capsys):
    never_ready = {**UNREAD, "seconds": None}
    assert nsp.check_startup([_ledger(tmp_path, [_trial(READ), _trial(READ), _trial(never_ready)])]) == 0
    assert capsys.readouterr().out.startswith("confirmed: 2 start-ups timed, median 90 s; 2 vLLM graph captures")


def test_check_startup_without_a_record_or_without_vllm(tmp_path, capsys):
    assert nsp.check_startup([_ledger(tmp_path / "a", [_trial(READ), _trial(None)])]) == 2
    assert "have no start-up record" in capsys.readouterr().out
    only_sglang = _ledger(tmp_path / "b", [_trial({**UNREAD}, engine="sglang")])
    assert nsp.check_startup([only_sglang]) == 1  # timed, and no log line was ever looked for
    assert capsys.readouterr().out.startswith("undecided")


def test_rewrites_table_and_an_unread_log(tmp_path, capsys):
    changed = {**READ, "rewrites": ["cudagraph_mode=PIECEWISE"], "parsed": ["capture", "rewrites"]}
    ledger = _ledger(tmp_path / "a", [_trial(READ), _trial(changed, spec_decode="ngram_gpu:4", version="0.29.0")])
    assert nsp.check_startup([ledger], "rewrites") == 0
    out = capsys.readouterr().out
    assert "vLLM 0.29.0 speculating: 1 trials, changed: cudagraph_mode=PIECEWISE" in out
    assert "vLLM 0.33.0 plain: 1 trials, changed: none" in out
    assert nsp.check_startup([_ledger(tmp_path / "b", [_trial(UNREAD)])], "rewrites") == 1  # "none" proves nothing


def test_compile_cache_check(tmp_path, capsys):
    hit = {**READ, "compile_cached": True, "compile_s": None, "parsed": ["capture", "compile_cache"]}
    assert nsp.check_startup([_ledger(tmp_path / "a", [_trial(READ), _trial(hit)])], "cache") == 0
    assert nsp.check_startup([_ledger(tmp_path / "b", [_trial(READ), _trial(READ)])], "cache") == 2
    assert nsp.check_startup([_ledger(tmp_path / "c", [_trial(READ)])], "cache") == 1  # nothing launched twice
    twice = [_trial(READ, key="vllm/bf16/sd=ngram_gpu:4"), _trial(READ, key="vllm/bf16/sd=ngram_gpu:4")]
    assert nsp.check_startup([_ledger(tmp_path / "d", twice)], "cache") == 1  # the cache is off there by design
    capsys.readouterr()


# ---- check-kernel

def _row(kernels=None, stage="batch", tok_s=100.0, at64=None, error=None, batch=64):
    config = {"backend": "sglang", "quant": "bf16", "ctx": 4096, "batch": batch}
    if kernels:
        config["kernels"] = kernels
    by = {"64": {"tok_s": at64}} if at64 is not None else {}
    return {"stage": stage, "config": config, "launched": error is None, "error": error,
            "metrics": {"tok_s": tok_s, "by_concurrency": by}}


def _profile(*rows, notes=(), pick=None) -> dict:
    return {"calibration_table": list(rows), "notes": list(notes), "config": pick or rows[0]["config"]}


def test_check_kernel_confirms_and_refutes_a_gain(capsys):
    leader, faster = _row(at64=1000.0), _row("cgmax=64", "kernels", at64=1120.0)
    assert nsp.check_kernel(_profile(leader, faster), "cgmax=64", level=64, min_gain=0.03) == 0
    assert capsys.readouterr().out.startswith("confirmed: cgmax=64: +12.0% tok/s at 64 users, not adopted")
    flat = _row("cgmax=64", "kernels", at64=1010.0)
    assert nsp.check_kernel(_profile(leader, flat), "cgmax=64", level=64, min_gain=0.03) == 2
    assert capsys.readouterr().out.startswith("refuted")


def test_check_kernel_within_a_band(capsys):
    leader = _row()
    assert nsp.check_kernel(_profile(leader, _row("attn=TRITON_ATTN", "kernels", tok_s=103.0)),
                            "attn=TRITON_ATTN", max_abs=0.05) == 0
    assert nsp.check_kernel(_profile(leader, _row("attn=TRITON_ATTN", "kernels", tok_s=88.0)),
                            "attn=TRITON_ATTN", max_abs=0.05) == 2
    capsys.readouterr()


def test_check_kernel_undecided_is_not_confirmed(capsys):
    leader = _row(at64=1000.0)
    note = "kernel setting cgmax=64 not tried on sglang: this SGLang has no --cuda-graph-max-bs-decode"
    cases = [
        _profile(leader, notes=[note]),  # never tried
        _profile(leader, _row("cgmax=64", "kernels", error="out of memory at start-up; failed to start")),
        _profile(leader, _row("cgmax=64", "kernels", at64=None)),  # the level was not measured
        _profile(_row("cgmax=64", "kernels", at64=1100.0)),  # no trial of the leader to compare with
        _profile(leader, _row("cgmax=64", "kernels", at64=1100.0, batch=128)),  # built on another config
        _profile(leader, _row("cgmax=128", "kernels", at64=1100.0)),  # another setting
    ]
    for profile in cases:
        assert nsp.check_kernel(profile, "cgmax=64", level=64, min_gain=0.03) == 1
        assert capsys.readouterr().out.startswith("undecided")
    assert nsp.check_kernel(_profile(leader, _row("cgmax=64", "kernels", at64=1100.0)), "cgmax=64", level=64) == 1


def test_check_kernel_reads_the_setting_in_any_spelling(capsys):
    rows = (_row(), _row("attn=triton,cgmax=64", "kernels", tok_s=110.0))
    assert nsp.check_kernel(_profile(*rows), " cgmax=64 , attn=triton", min_gain=0.03) == 0
    capsys.readouterr()


def test_cli_entry_points(tmp_path, capsys):
    profile = tmp_path / "p.json"
    profile.write_text(json.dumps(_profile(_row(), _row("cgmax=64", "kernels", tok_s=120.0))), encoding="utf-8")
    assert nsp.main(["--check-kernel", str(profile), "--setting", "cgmax=64", "--min-gain", "0.03"]) == 0
    assert nsp.main(["--check-kernel", str(tmp_path / "missing.json"), "--setting", "cgmax=64"]) == 1
    assert nsp.main(["--check-startup", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "undecided: the profile could not be read" in out and str(tmp_path) not in out
