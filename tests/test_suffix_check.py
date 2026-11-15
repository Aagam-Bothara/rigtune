"""Suffix decoding is offered on vLLM unless vLLM's own Python shows Arctic Inference unusable.

On 14 Nov 2026 every suffix trial on an A40 pod failed to start although arctic-inference 0.1.1 had installed and
`import arctic_inference` worked; vLLM's own check imports only that top-level package. The check fails open: only a
positive finding removes the method, and a check that could not run says so and offers it as before. Apart from one
real probe of this interpreter (no vLLM needed), the probe's report is injected or its code runs against fake modules."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from contextlib import redirect_stdout

import pytest

import rigtune.backends.vllm as vl
from rigtune import speculative
from rigtune.backends import get_backend
from rigtune.models import Config

PY = "/opt/vllm/bin/python"
OK = {"python": "3.12.3", "arctic_inference": "0.1.1", "vllm": "0.29.0", "toplevel": "ok", "suffix": "ok"}
ABSENT_ERR = "ModuleNotFoundError: No module named 'arctic_inference'"
ABSENT = {"python": "3.12.3", "arctic_inference": None, "vllm": "0.29.0", "toplevel": ABSENT_ERR, "suffix": ABSENT_ERR}
# What Python 3.12 raised in the 14 Nov pod's image for the suffix tree CMake had built for Python 3.13.
NO_C = "ModuleNotFoundError: No module named 'arctic_inference.suffix_decoding._C'"
ABSENT_NOTE = ("vllm: suffix decoding not tried: arctic-inference is not installed in vLLM's Python; see "
               "docs/usage.md#suffix-decoding, then run `rigtune recalibrate`")


@pytest.fixture
def fresh():
    """Clears the per-process answers before and after; a function a test patched has no cache to clear."""
    def clear():
        for fn in (vl._vllm_python, vl._arctic_report, vl._suffix_refusal, vl._serve_cli):
            getattr(fn, "cache_clear", lambda: None)()

    clear()
    yield
    clear()


# --------------------------------------------------------------------------- the report


def test_a_report_that_suffix_loads_is_accepted():
    assert speculative.suffix_refusal(OK) is None
    assert speculative.suffix_refusal({**OK, "arctic_inference": "0.3.0"}) is None  # same suffix code (0.1.0-0.3.0)
    assert speculative.suffix_refusal({**OK, "plugin_vllm": "0.29.0"}) is None  # an enabled plugin that matches
    assert speculative.suffix_refusal({**OK, "arctic_inference": None}) is None  # no metadata, but the tree loads


def test_each_finding_is_named_without_a_path():
    assert speculative.suffix_refusal(ABSENT) == ABSENT_NOTE[len("vllm: suffix decoding not tried: "):]
    toplevel = "ModuleNotFoundError: No module named 'grpc'"
    broken_top = speculative.suffix_refusal({**OK, "toplevel": toplevel, "suffix": toplevel})
    assert broken_top.startswith("arctic-inference 0.1.1 fails to import in vLLM's Python") and "'grpc'" in broken_top
    so = ("ImportError: /home/alice/env/lib/python3.12/site-packages/arctic_inference/suffix_decoding/"
          "_C.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN8nanobind")
    built = speculative.suffix_refusal({**OK, "suffix": so})
    assert "suffix tree does not load" in built and "_C.cpython-312-x86_64-linux-gnu.so" in built
    assert "alice" not in built and "/home" not in built  # the profile may be committed
    assert NO_C in speculative.suffix_refusal({**OK, "suffix": NO_C})
    assert "predates 0.1.0" in speculative.suffix_refusal({**OK, "arctic_inference": "0.0.9"})
    plugin = speculative.suffix_refusal({**OK, "plugin_vllm": "0.10.1"})
    assert "ARCTIC_INFERENCE_ENABLED=1" in plugin and "vllm==0.10.1" in plugin and "0.29.0" in plugin
    assert all(r.endswith("then run `rigtune recalibrate`") or "plugin" in r
               for r in (broken_top, built, speculative.suffix_refusal(ABSENT)))


def test_the_install_line_quotes_cmake_args():
    assert speculative.arctic_install_line(PY) == (
        f'CMAKE_ARGS="-DPython_EXECUTABLE={PY}" {PY} -m pip install --no-cache-dir --force-reinstall '
        "arctic-inference==0.1.1")


# --------------------------------------------------------------------------- the probe


def _fake_arctic(monkeypatch, cache=None, toplevel_error=None):
    built = []

    class Cache:
        def __init__(self, max_tree_depth, max_cached_requests):
            built.append((max_tree_depth, max_cached_requests))

    pkg = types.ModuleType("arctic_inference")
    pkg.__path__ = []
    sd = types.ModuleType("arctic_inference.suffix_decoding")
    sd.SuffixDecodingCache = cache or Cache
    utils = types.ModuleType("arctic_inference.utils")
    utils.get_compatible_vllm_version = lambda: "0.10.1"
    for name, mod in (("arctic_inference", pkg), ("arctic_inference.suffix_decoding", sd),
                      ("arctic_inference.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    if toplevel_error:  # `import arctic_inference` raises, and so does everything under it
        for name in ("arctic_inference", "arctic_inference.suffix_decoding", "arctic_inference.utils"):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    return built, sd


def _exec_probe() -> dict:
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(speculative.SUFFIX_PROBE, "<probe>", "exec"), {"__name__": "__main__"})
    return json.loads(out.getvalue().strip().splitlines()[-1])


def test_the_probe_imports_the_package_and_the_suffix_tree_and_never_vllm(monkeypatch):
    built, sd = _fake_arctic(monkeypatch)
    monkeypatch.delenv("ARCTIC_INFERENCE_ENABLED", raising=False)
    report = _exec_probe()
    assert report["toplevel"] == report["suffix"] == "ok" and built == [(24, 1)] and "plugin_vllm" not in report
    assert "vllm" not in sys.modules
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "1")
    assert _exec_probe()["plugin_vllm"] == "0.10.1"
    monkeypatch.setenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "1")
    assert "plugin_vllm" not in _exec_probe()
    monkeypatch.delenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK")

    def broken(*a, **kw):
        raise ImportError("undefined symbol: _ZN8nanobind")

    sd.SuffixDecodingCache = broken
    assert _exec_probe()["suffix"] == "ImportError: undefined symbol: _ZN8nanobind"


def test_the_probe_tells_a_package_that_fails_to_import(monkeypatch):
    _fake_arctic(monkeypatch, toplevel_error=True)
    report = _exec_probe()
    assert report["toplevel"].startswith("ModuleNotFoundError") and report["suffix"] != "ok"


def test_the_probe_runs_in_the_given_python_from_an_empty_directory(monkeypatch):
    seen = []

    def fake_run(args, **kw):
        import os

        seen.append((args, os.listdir(kw["cwd"])))
        return subprocess.CompletedProcess(args, 0, stdout="noise\n" + json.dumps(OK) + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert speculative.run_suffix_probe(PY) == OK
    assert seen == [([PY, "-c", speculative.SUFFIX_PROBE], [])]
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 1, stdout="", stderr="Traceback\nSyntaxError: bad"))
    with pytest.raises(RuntimeError, match="exited 1: SyntaxError: bad"):
        speculative.run_suffix_probe(PY)
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, stdout="a banner from sitecustomize\n", stderr=""))
    with pytest.raises(ValueError):
        speculative.run_suffix_probe(PY)  # not JSON
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, stdout="[1]\n"))
    with pytest.raises(ValueError):
        speculative.run_suffix_probe(PY)  # JSON, but no report


def test_a_real_probe_of_this_interpreter_ignores_the_working_directory(monkeypatch, tmp_path):
    """One real subprocess: this interpreter, no vLLM needed. A directory named arctic_inference where rigtune runs
    must not stand in for the package (`python -c` puts the working directory first on sys.path)."""
    fake = tmp_path / "arctic_inference" / "suffix_decoding"
    fake.mkdir(parents=True)
    (tmp_path / "arctic_inference" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "__init__.py").write_text("class SuffixDecodingCache:\n    def __init__(self, **kw): pass\n",
                                      encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    report = speculative.run_suffix_probe(sys.executable)
    assert report["python"] == sys.version.split()[0]
    if report["arctic_inference"] is None:  # the usual case: not installed here
        assert report["toplevel"] == ABSENT_ERR and report["suffix"] != "ok"


# --------------------------------------------------------------------------- vLLM's interpreter


def test_only_a_python_is_taken_from_the_vllm_script(tmp_path, monkeypatch):
    monkeypatch.setattr(vl.shutil, "which", lambda name: f"/opt/env/bin/{name}")
    cases = {
        "#!/opt/env/bin/python3.12\nimport sys\n": "/opt/env/bin/python3.12",
        "#!/usr/bin/env python3.12\n": "/opt/env/bin/python3.12",
        "#!/usr/bin/env -S python3 -I\n": "/opt/env/bin/python3",
        "#!/usr/bin/env PYTHONNOUSERSITE=1 pypy3\n": "/opt/env/bin/pypy3",
        "#!/bin/sh\n'''exec' '/very/long/env/bin/python' \"$0\" \"$@\"\n' '''\n": "/very/long/env/bin/python",
        "#!/bin/sh\n'''exec' /a/bin/python3.12 \"$0\" \"$@\"\n' '''\n": "/a/bin/python3.12",
        "#!/bin/bash\nexec /opt/venv/bin/vllm \"$@\"\n": None,  # a wrapper: probing bash would refuse a working install
        "#!/usr/bin/env bash\nexec /opt/venv/bin/vllm \"$@\"\n": None,
        "#!/bin/sh\nexec /opt/venv/bin/vllm \"$@\"\n": None,
        "#!/bin/sh\n'''exec' /usr/bin/bash \"$0\" \"$@\"\n": None,
    }
    for i, (text, expected) in enumerate(cases.items()):
        script = tmp_path / f"vllm{i}"
        script.write_text(text, encoding="utf-8")
        assert vl.script_python(script) == expected, text
    binary = tmp_path / "vllm-bin"
    binary.write_bytes(b"\x7fELF\x00")
    assert vl.script_python(binary) is None
    assert vl.script_python(tmp_path / "missing") is None


def test_vllm_python_falls_back_to_this_interpreter(fresh, monkeypatch, tmp_path):
    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    assert vl._vllm_python() == sys.executable
    wrapper = tmp_path / "wrap" / "vllm"
    wrapper.parent.mkdir()
    wrapper.write_text("#!/bin/bash\nexec /opt/venv/bin/vllm \"$@\"\n", encoding="utf-8")
    (wrapper.parent / "python3").write_text("", encoding="utf-8")  # beside a shell wrapper: not trusted
    monkeypatch.setattr(vl, "_serve_cli", lambda: str(wrapper))
    vl._vllm_python.cache_clear()
    assert vl._vllm_python() == sys.executable  # rigtune already requires vLLM importable here


def test_a_launcher_without_a_shebang_uses_the_python_beside_it(fresh, monkeypatch, tmp_path):
    launcher = tmp_path / "bin" / "vllm"
    launcher.parent.mkdir()
    launcher.write_bytes(b"MZ\x90\x00 a binary launcher")
    monkeypatch.setattr(vl, "_serve_cli", lambda: str(launcher))
    assert vl._vllm_python() == sys.executable  # none beside it
    (launcher.parent / "python3").write_text("", encoding="utf-8")
    vl._vllm_python.cache_clear()
    assert vl._vllm_python() == str(launcher.parent / "python3")


# --------------------------------------------------------------------------- once per calibration, failing open


def test_this_interpreter_without_the_package_is_answered_without_spawning(fresh, monkeypatch):
    import importlib.util

    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "arctic_inference" else real(name, *a))

    def never(python, timeout=60.0):
        raise AssertionError("spawned a probe for a package that is not there")

    monkeypatch.setattr(speculative, "run_suffix_probe", never)
    assert vl._suffix_refusal() == ABSENT_NOTE[len("vllm: suffix decoding not tried: "):]
    assert vl._suffix_caveat() is None


def test_a_vllm_elsewhere_is_asked_once_in_its_own_python(fresh, monkeypatch, tmp_path):
    cli = tmp_path / "bin" / "vllm"
    cli.parent.mkdir()
    cli.write_text("#!/srv/vllm-env/bin/python\n", encoding="utf-8")
    monkeypatch.setattr(vl, "_serve_cli", lambda: str(cli))
    asked = []

    def probe(python, timeout=60.0):
        asked.append(python)
        return {**OK, "suffix": NO_C}

    monkeypatch.setattr(speculative, "run_suffix_probe", probe)
    first = vl._suffix_refusal()
    assert vl._suffix_refusal() == first and vl._suffix_caveat() is None
    assert asked == ["/srv/vllm-env/bin/python"]  # once per calibration
    assert NO_C in first and "/srv" not in first


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(["python"], 60), OSError("No such file"),
                                     ValueError("Expecting value"), RuntimeError("exited 1: boom")])
def test_a_check_that_could_not_run_offers_suffix_and_says_so(fresh, monkeypatch, tmp_path, failure):
    monkeypatch.setattr(vl, "_vllm_python", lambda: "/srv/vllm-env/bin/python")

    def probe(python, timeout=60.0):
        raise failure

    monkeypatch.setattr(speculative, "run_suffix_probe", probe)
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")
    assert vl._suffix_refusal() is None and speculative.vllm_supports_suffix()
    assert vl._suffix_caveat() == (f"could not check that arctic-inference loads in vLLM's Python "
                                   f"({type(failure).__name__}); suffix decoding offered anyway")


def test_no_interpreter_found_fails_open(fresh, monkeypatch):
    monkeypatch.setattr(vl, "_vllm_python", lambda: None)
    assert vl._suffix_refusal() is None
    assert "no interpreter found" in vl._suffix_caveat()


# --------------------------------------------------------------------------- what the search is offered


def _qwen3b(prepared_vllm):
    return prepared_vllm.model_copy(update={"spec": prepared_vllm.spec.model_copy(
        update={"hf_id": "Qwen/Qwen2.5-3B-Instruct"})})


def test_suffix_is_offered_unless_a_finding_says_otherwise(fresh, monkeypatch, prepared_vllm):
    be, model = get_backend("vllm"), _qwen3b(prepared_vllm)
    cfg = Config(backend="vllm", quant="bf16", batch=64)
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")

    def report(r, why=None):
        vl._suffix_refusal.cache_clear()
        monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, r, why))

    report(OK)
    ok = [c.spec_decode for c in be.spec_variants(cfg, model)]
    assert ok == ["ngram_gpu:4", "suffix:24", "draft:Qwen/Qwen2.5-0.5B-Instruct:4"]  # unchanged where it loads
    assert be.spec_notes(cfg, model) == []
    suffix = next(c for c in be.spec_variants(cfg, model) if c.spec_decode == "suffix:24")
    args = be.launch_spec(suffix, model, 8000).args
    assert json.loads(args[args.index("--speculative-config") + 1]) == {"method": "suffix", "num_speculative_tokens": 24}

    report({**OK, "suffix": NO_C})
    assert [c.spec_decode for c in be.spec_variants(cfg, model)] == [s for s in ok if s != "suffix:24"]
    (note,) = be.spec_notes(cfg, model)
    assert note.startswith("vllm: suffix decoding not tried: arctic-inference 0.1.1's suffix tree") and NO_C in note

    report(ABSENT)
    assert be.spec_notes(cfg, model) == [ABSENT_NOTE]

    report(None, "TimeoutExpired")  # could not ask: offered as before, and said
    assert [c.spec_decode for c in be.spec_variants(cfg, model)] == ok
    assert be.spec_notes(cfg, model) == ["vllm: could not check that arctic-inference loads in vLLM's Python "
                                         "(TimeoutExpired); suffix decoding offered anyway"]
    assert get_backend("vllm-cpu").spec_notes(cfg.model_copy(update={"backend": "vllm-cpu"}), model) == []
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.28.1")  # no suffix decoding there: nothing said
    assert be.spec_notes(cfg, model) == []


def test_an_enabled_plugin_that_refuses_this_vllm_is_said_once_for_every_launch(fresh, monkeypatch):
    monkeypatch.delenv("ARCTIC_INFERENCE_ENABLED", raising=False)
    monkeypatch.setattr(vl, "_arctic_report", lambda: (_ for _ in ()).throw(AssertionError("asked without cause")))
    be = get_backend("vllm")
    assert be.launch_note() is None  # not set: nothing asked
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "1")
    monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, {**OK, "plugin_vllm": "0.10.1"}, None))
    note = be.launch_note()
    assert note == ("vllm: ARCTIC_INFERENCE_ENABLED=1 turns on arctic-inference 0.1.1's vLLM plugin, which refuses "
                    "vLLM 0.29.0 at start-up (it patches vllm==0.10.1 only), so every vLLM launch would fail; unset it")
    monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, {**OK, "plugin_vllm": "0.29.0"}, None))
    assert be.launch_note() is None  # the one release it patches
    monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, ABSENT, None))
    assert be.launch_note() is None  # no package, no plugin
    monkeypatch.setenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "1")
    monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, {**OK, "plugin_vllm": "0.10.1"}, None))
    assert be.launch_note() is None
    assert get_backend("sglang").launch_note() is None


# --------------------------------------------------------------------------- a calibration, end to end


def _calibrate(monkeypatch, report, env=None):
    from rigtune.backends import registry
    from rigtune.calibrate.workload import get_workload
    from rigtune.memory import plan
    from rigtune.models import ModelSpec
    from rigtune.pipeline import PlanResult, SearchOptions, calibrate
    from tests.conftest import make_hw
    from tests.test_hybrid_memory import qwen25
    from tests.test_kv_tier_off import _pin
    from tests.test_objectives_and_search import FakeRunner

    real = speculative.vllm_supports_suffix
    _pin(monkeypatch)
    monkeypatch.setattr(speculative, "vllm_supports_suffix", real)  # _pin says yes; this test asks
    monkeypatch.setattr(speculative, "_vllm_version", lambda: "0.29.0")
    monkeypatch.setattr(vl, "_arctic_report", lambda: (PY, report, None))
    vl._suffix_refusal.cache_clear()
    hw = make_hw("rtx4090")
    reg = registry()
    pm = qwen25("vllm")
    feasible = {"vllm": plan(hw, pm, reg["vllm"].candidate_configs(hw, pm), reg["vllm"].memory_model(hw))}
    result = PlanResult(hw=hw, spec=ModelSpec(hf_id=pm.spec.hf_id), candidates=["vllm"], prepared={"vllm": pm},
                        feasible=feasible)
    runner = FakeRunner()
    profile = calibrate(hw, result.spec, "balanced", result, reg, workload=get_workload("chat"), runner=runner,
                        options=SearchOptions())
    vl._suffix_refusal.cache_clear()
    return profile, runner


@pytest.mark.usefixtures("tmp_home")
def test_a_calibration_without_arctic_inference_tries_no_suffix_and_says_so_once(monkeypatch):
    profile, runner = _calibrate(monkeypatch, ABSENT)
    assert not any("sd=suffix" in k for k in runner.calls)
    assert [n for n in profile.notes if "suffix" in n or "arctic" in n] == [ABSENT_NOTE]
    profile, runner = _calibrate(monkeypatch, OK)
    assert any("sd=suffix" in k for k in runner.calls)
    assert [n for n in profile.notes if "suffix" in n or "arctic" in n] == []


def test_the_profile_records_arctic_inference_only_where_it_is_installed(monkeypatch):
    """A profile lists the engine's libraries beside its version (hardware.ENGINE_LIBS). They are not part of its
    cache key, so recording arctic-inference invalidates no profile; where it is absent nothing changes."""
    from rigtune import hardware

    versions = {"torch": "2.13.0", "transformers": "5.23.0"}
    monkeypatch.setattr(hardware, "_pkg_version", lambda dist: versions.get(dist))
    assert hardware._libs_here() == {"torch": "2.13.0", "transformers": "5.23.0"}
    versions["arctic_inference"] = "0.1.1"
    assert hardware._libs_here()["arctic-inference"] == "0.1.1"
