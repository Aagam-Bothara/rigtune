"""The engine probe: its scripts, run for real in a subprocess against stub engines, and the cache that runs them once
per install. Nothing here needs vLLM, SGLang or a GPU."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rigtune.backends import facts as ef
from rigtune.hardware import ENGINE_LIBS
from tests import engine_stubs


@pytest.fixture
def probing(monkeypatch, tmp_home):
    """Probing on (conftest turns it off for every other test), with rigtune's home in a temporary directory."""
    monkeypatch.delenv("RIGTUNE_ENGINE_PROBE", raising=False)
    monkeypatch.delenv("POLYSERVE_ENGINE_PROBE", raising=False)
    return tmp_home


def _stub(engine, tmp_path, monkeypatch, **kw):
    root = engine_stubs.build(engine, tmp_path / "stub", **kw)
    monkeypatch.setenv("PYTHONPATH", str(root))
    return root


def test_vllm_probe_answers_every_question_from_the_engine(tmp_path, monkeypatch):
    _stub("vllm", tmp_path, monkeypatch)
    f = ef.run_probe("vllm", sys.executable)
    assert f.errors == {}
    assert f.version == "0.33.0"
    # A Literal of Literals, flattened, as vLLM's SpeculativeMethod is.
    assert f.spec_methods == sorted(["ngram", "draft_model", "suffix", "eagle", "eagle3", "deepseek_mtp",
                                     "qwen3_5_mtp", "mtp", "ngram_gpu"])
    assert f.kv_dtypes == ["auto", "fp8", "fp8_e5m2", "int8_per_token_head"]
    assert f.has_flag("--attention-backend") and f.has_flag("--no-enable-prefix-caching") and f.has_flag("-sc")
    assert f.has_flag("--enable-prompt-tokens-details") and not f.has_flag("--no-such-flag")
    assert f.flags["--kv-cache-dtype"] == {"choices": ["auto", "fp8", "fp8_e5m2"], "default": "auto"}
    assert isinstance(f.flags["--speculative-config"]["default"], str)  # an object(), kept as its text
    assert f.features == {"offloading_connector": True, "supports_hma": True, "offload_prompt_only": True,
                          "shm_region": True}
    assert f.archs == ["GptOssForCausalLM", "LlamaForCausalLM", "Qwen2ForCausalLM"]
    assert f.provenance == "probe" and f.probed_at and f.seconds is not None and f.unanswered() == []


def test_sglang_probe_answers_every_question_from_the_engine(tmp_path, monkeypatch):
    _stub("sglang", tmp_path, monkeypatch)
    f = ef.run_probe("sglang", sys.executable)
    assert f.errors == {}
    assert f.version == "0.5.23"
    assert f.spec_methods == ["DSPARK", "EAGLE", "EAGLE3", "NEXTN", "NGRAM", "NONE", "STANDALONE"]  # NEXTN: alias
    assert f.has_flag("--enable-cache-report") and f.has_flag("--tp-size") and f.has_flag("--hicache-size")
    assert f.flags["--hicache-storage-backend"]["choices"] == ["file", "mooncake"]
    assert f.request_fields == ["messages", "model", "return_cached_tokens_details"]
    assert f.env_vars == ["SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"]
    assert f.arch_sets == {"mamba_extra_buffer": ["Qwen3_5ForCausalLM"], "swa_sink": ["GptOssForCausalLM"]}
    assert f.archs == ["GptOssForCausalLM", "LlamaForCausalLM"]


def test_a_question_that_fails_costs_only_its_own_answer(tmp_path, monkeypatch):
    broken = "raise ImportError('needs a GPU')\n"
    _stub("vllm", tmp_path, monkeypatch,
          replace={"vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py": broken,
                   "vllm/config/cache.py": None})
    f = ef.run_probe("vllm", sys.executable)
    assert f.features is None and f.kv_dtypes is None
    assert f.errors["features"] == "ImportError: needs a GPU"
    assert f.errors["kv_dtypes"].startswith("ModuleNotFoundError")
    assert f.flags is not None and f.archs is not None and f.spec_methods is not None
    assert sorted(f.unanswered()) == ["features", "kv_dtypes"]


def test_a_parser_that_exits_is_an_error_not_a_crash(tmp_path, monkeypatch):
    exits = ("class ServeSubcommand:\n"
             "    def subparser_init(self, subparsers):\n"
             "        raise SystemExit(2)\n")
    _stub("vllm", tmp_path, monkeypatch, replace={"vllm/entrypoints/cli/serve.py": exits})
    f = ef.run_probe("vllm", sys.executable)
    assert f.flags is None and f.errors["flags"] == "SystemExit: 2" and f.archs is not None


def test_the_engines_own_log_lines_around_the_answers_are_ignored(tmp_path, monkeypatch):
    noisy = engine_stubs.VLLM["vllm/config/cache.py"] + "import sys\nsys.stdout.write('INFO no newline here')\n"
    _stub("vllm", tmp_path, monkeypatch, replace={"vllm/config/cache.py": noisy})
    f = ef.run_probe("vllm", sys.executable)
    assert f.errors == {} and f.kv_dtypes == ["auto", "fp8", "fp8_e5m2", "int8_per_token_head"]


def test_no_engine_at_all_answers_nothing_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "empty"))
    f = ef.run_probe("sglang", sys.executable)
    assert f.flags is None and f.archs is None and f.version is None
    assert f.errors["flags"].startswith("ModuleNotFoundError")


def test_an_interpreter_that_cannot_run_is_reported_for_every_key(tmp_path):
    f = ef.run_probe("vllm", str(tmp_path / "no-such-python"))
    assert set(f.unanswered()) == {"spec_methods", "kv_dtypes", "flags", "features", "archs"}
    assert all(e.startswith("the probe could not run") for e in f.errors.values())


def test_a_probe_that_runs_out_of_time_keeps_what_it_answered():
    """What a probe printed before it was stopped is kept (subprocess.run hands it over with TimeoutExpired, as bytes
    on POSIX); the rest says it timed out."""
    partial = (b"INFO loading\n" + ef.MARK.encode() + b'{"key": "version", "value": "0.33.0"}\n'
               + ef.MARK.encode() + b'{"key": "flags", "value": {"--port": {"choices": null, "default": 8000}}}\n')

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=partial)

    f = ef.run_probe("vllm", "python", timeout=120, run=run)
    assert f.version == "0.33.0" and f.has_flag("--port")
    assert f.archs is None and f.errors["archs"] == "no answer within 120 s"


def test_the_probe_runs_with_no_gpu_visible_from_an_empty_directory(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(kwargs, argv=argv, listing=os.listdir(kwargs["cwd"]))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PYTHONPATH", "/opt/engine")
    ef.run_probe("sglang", "/opt/sglang/bin/python", run=run)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "" and seen["env"]["PYTHONPATH"] == "/opt/engine"
    assert seen["listing"] == [] and seen["timeout"] == ef.PROBE_TIMEOUT == 120.0
    assert seen["argv"] == ["/opt/sglang/bin/python", str(ef.script_path("sglang"))]


def test_a_relative_pythonpath_entry_finds_nothing_where_rigtune_runs(tmp_path, monkeypatch):
    """The probe runs from an empty folder: a "." on PYTHONPATH (it would resolve to rigtune's own current folder)
    must not import a directory there named like the engine. Run from the current folder, it would."""
    root = _stub("vllm", tmp_path, monkeypatch)
    here = tmp_path / "cwd"
    (here / "vllm").mkdir(parents=True)
    (here / "vllm" / "__init__.py").write_text("raise RuntimeError('the wrong vllm')\n", encoding="utf-8")
    monkeypatch.chdir(here)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([".", str(root)]))
    assert ef.run_probe("vllm", sys.executable).errors == {}
    poisoned = subprocess.run([sys.executable, str(ef.script_path("vllm"))], capture_output=True, cwd=here,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert ef.parse("vllm", ef._text(poisoned.stdout), "not answered").errors  # what the empty folder prevents


def test_answers_of_the_wrong_shape_are_no_answers():
    out = ef.MARK + '{"key": "archs", "value": "LlamaForCausalLM"}\n' + ef.MARK + 'not json\n'
    f = ef.parse("vllm", out, "not answered")
    assert f.archs is None and f.errors["archs"].startswith("unreadable answer")
    assert f.errors["flags"] == "not answered"


# ---------------------------------------------------------------- once per install


def _counting(answer=None):
    calls = []

    def runner(engine, python):
        calls.append((engine, python))
        return answer or ef.EngineFacts(engine=engine, version="0.33.0", archs=["LlamaForCausalLM"])

    return runner, calls


def test_an_install_is_asked_once_and_its_answers_kept(probing, tmp_path):
    pkg = tmp_path / "site-packages" / "vllm"
    pkg.mkdir(parents=True)
    runner, calls = _counting()
    first = ef.cached_facts("vllm", "/opt/vllm/bin/python", pkg, runner=runner)
    again = ef.cached_facts("vllm", "/opt/vllm/bin/python", pkg, runner=runner)
    assert len(calls) == 1 and first == again and again.archs == ["LlamaForCausalLM"]
    assert ef.cache_file("vllm", "/opt/vllm/bin/python", pkg).parent == probing / "engines"


def test_a_reinstall_another_python_or_a_new_probe_asks_again(probing, tmp_path, monkeypatch):
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    runner, calls = _counting()
    ef.cached_facts("vllm", "/a/python", pkg, runner=runner)
    st = pkg.stat()
    os.utime(pkg, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # pip replaced the package
    ef.cached_facts("vllm", "/a/python", pkg, runner=runner)
    ef.cached_facts("vllm", "/b/python", pkg, runner=runner)
    script = tmp_path / "probe.py"
    script.write_text("# asks more\n", encoding="utf-8")
    monkeypatch.setattr(ef, "script_path", lambda engine: script)
    ef.cached_facts("vllm", "/b/python", pkg, runner=runner)
    assert len(calls) == 4


def test_a_failed_probe_is_kept_too_until_asked_again(probing, tmp_path):
    pkg = tmp_path / "sglang"
    pkg.mkdir()
    failed = ef.EngineFacts(engine="sglang", errors={k: "the probe exited 1: ImportError: no module named torch"
                                                     for k in ef.KEYS["sglang"]})
    runner, calls = _counting(failed)
    assert ef.cached_facts("sglang", "/py", pkg, runner=runner).flags is None
    ef.cached_facts("sglang", "/py", pkg, runner=runner)
    ef.forget("sglang", "/py", pkg)
    ef.cached_facts("sglang", "/py", pkg, runner=runner)
    assert len(calls) == 2


def test_a_probe_that_ran_out_of_time_is_asked_again_next_run(probing, tmp_path):
    """A slow first import (a cold disk) is not a property of the install: kept, it would leave the source checks
    answering until --reprobe."""
    pkg = tmp_path / "sglang"
    pkg.mkdir()
    slow = ef.EngineFacts(engine="sglang", version="0.5.23", flags={"--port": {}},
                          errors={"archs": "no answer within 120 s"})
    runner, calls = _counting(slow)
    assert ef.cached_facts("sglang", "/py", pkg, runner=runner).has_flag("--port")  # used for this run
    ef.cached_facts("sglang", "/py", pkg, runner=runner)
    assert len(calls) == 2 and not (probing / "engines").exists()


def test_an_unreadable_kept_answer_is_asked_again(probing, tmp_path):
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    path = ef.cache_file("vllm", "/py", pkg)
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    runner, calls = _counting()
    assert ef.cached_facts("vllm", "/py", pkg, runner=runner).version == "0.33.0" and len(calls) == 1


def test_nothing_is_run_with_probing_off_or_no_engine_found(tmp_home, tmp_path, monkeypatch):
    runner, calls = _counting()
    monkeypatch.setenv("RIGTUNE_ENGINE_PROBE", "off")
    assert ef.cached_facts("vllm", "/py", tmp_path, runner=runner) is None
    monkeypatch.delenv("RIGTUNE_ENGINE_PROBE")
    assert ef.cached_facts("vllm", None, tmp_path, runner=runner) is None
    assert ef.cached_facts("vllm", "/py", None, runner=runner) is None
    assert calls == []


def test_tests_never_probe_unless_they_ask():
    assert not ef.enabled()


# ---------------------------------------------------------------- the backends and the CLI


def test_each_backend_asks_the_python_and_package_it_launches(monkeypatch, tmp_path):
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    from rigtune.backends import registry

    asked = []
    monkeypatch.setattr(ef, "cached_facts", lambda engine, python, package: asked.append((engine, python, package))
                        or ef.EngineFacts(engine=engine))
    monkeypatch.setattr(vl, "_vllm_python", lambda: "/opt/vllm/bin/python")
    monkeypatch.setattr(vl, "_vllm_package", lambda: tmp_path / "vllm")
    monkeypatch.setattr(sg, "sglang_python", lambda: "/opt/sglang/bin/python")
    monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: tmp_path / "sglang")
    vl._engine_facts.cache_clear()
    sg._engine_facts.cache_clear()
    try:
        reg = registry()
        assert reg["vllm"].facts().engine == "vllm" and reg["sglang"].facts().engine == "sglang"
        reg["vllm"].facts()  # kept for the process
        assert reg["llamacpp-cuda"].facts() is None
        assert asked == [("vllm", "/opt/vllm/bin/python", tmp_path / "vllm"),
                         ("sglang", "/opt/sglang/bin/python", tmp_path / "sglang")]
    finally:
        vl._engine_facts.cache_clear()
        sg._engine_facts.cache_clear()


def test_probe_engines_prints_each_engines_answers(monkeypatch):
    from typer.testing import CliRunner

    from rigtune import cli
    from tests.conftest import make_hw

    answers = {"vllm": ef.EngineFacts(engine="vllm", version="0.33.0", flags={"--port": {}}, archs=["A", "B"],
                                      spec_methods=["mtp", "ngram"], errors={"features": "ImportError: x"}),
               "sglang": None}
    monkeypatch.setattr("rigtune.hardware.probe", lambda: make_hw("rtx4090"))
    monkeypatch.setattr(cli, "_engine_facts", lambda refresh: answers)
    out = CliRunner().invoke(cli.app, ["probe", "--engines"], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    assert "0.33.0" in out.output and "mtp, ngram" in out.output and "features: ImportError: x" in out.output
    assert "not asked" in out.output
    as_json = CliRunner().invoke(cli.app, ["probe", "--engines", "--json"])
    assert '"engine_facts"' in as_json.output and '"sglang": null' in as_json.output


# ---------------------------------------------------------------- the scripts


@pytest.mark.parametrize("engine", sorted(ef.SCRIPTS))
def test_the_scripts_use_the_standard_library_and_the_engine_only(engine):
    """rigtune is usually not installed where the engine is, so a script may import nothing else at its top."""
    tree = ast.parse(ef.script_path(engine).read_text(encoding="utf-8"))
    top = {a.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for a in node.names}
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top <= {"argparse", "importlib", "inspect", "json", "sys", "typing"}


@pytest.mark.parametrize("engine", sorted(ef.SCRIPTS))
def test_the_scripts_record_the_libraries_rigtune_records(engine):
    tree = ast.parse(ef.script_path(engine).read_text(encoding="utf-8"))
    libs = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LIBS")
    assert libs == ENGINE_LIBS


def test_the_scripts_ship_in_the_package():
    """rigtune/backends/probes has an __init__.py, so setuptools packages its scripts with the rest of rigtune."""
    assert (Path(ef.__file__).parent / "probes" / "__init__.py").exists()
    assert all(ef.script_path(e).exists() for e in ef.SCRIPTS)


# ---------------------------------------------------------------- answers that must not be trusted, and paths


def test_a_module_that_fails_to_import_is_no_answer_not_a_no(tmp_path, monkeypatch):
    """The offloading answer reads two modules' source. One that fails to import (as a module needing a GPU may with
    none visible) makes the whole answer an error, which leaves it to the source check; read as empty it said
    shm_region False with no error, and that False beat the source's yes."""
    broken = "raise ImportError('libcudart.so: cannot open shared object file')\n"
    _stub("vllm", tmp_path, monkeypatch, replace={"vllm/v1/kv_offload/cpu/shared_offload_region.py": broken})
    f = ef.run_probe("vllm", sys.executable)
    assert f.features is None and f.errors["features"].startswith("ImportError: libcudart.so")


def test_an_empty_answer_is_no_answer(tmp_path, monkeypatch):
    """A release whose SpeculativeMethod is an Enum rather than a Literal flattens to []: taken as an answer, it would
    take every method away."""
    enum_methods = "import enum\nclass SpeculativeMethod(str, enum.Enum):\n    NGRAM = 'ngram'\n"
    _stub("vllm", tmp_path, monkeypatch, replace={"vllm/config/speculative.py": enum_methods})
    f = ef.run_probe("vllm", sys.executable)
    assert f.spec_methods is None and f.errors["spec_methods"] == "an empty answer, taken as none"
    assert f.flags is not None and "spec_methods" in f.unanswered()


def test_one_byte_that_is_not_utf8_loses_no_answer(tmp_path, monkeypatch):
    noisy = ("import sys\nsys.stdout.flush(); sys.stdout.buffer.write(b'\\x81\\xff junk\\n'); sys.stdout.flush()\n"
             + engine_stubs.VLLM["vllm/config/cache.py"])
    _stub("vllm", tmp_path, monkeypatch, replace={"vllm/config/cache.py": noisy})
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    f = ef.run_probe("vllm", sys.executable)
    assert f.errors == {} and f.kv_dtypes and f.archs


def _an_install(tmp_path, version="0.33.0"):
    site = tmp_path / "other-env" / "site-packages"
    pkg = site / "vllm"
    pkg.mkdir(parents=True)
    (site / f"vllm-{version}.dist-info").mkdir()
    (site / f"vllm-{version}.dist-info" / "RECORD").write_text("vllm/__init__.py,,\n", encoding="utf-8")
    return pkg


def test_answers_from_another_install_than_the_one_launched_are_not_used(probing, tmp_path):
    """A `vllm` on PATH that is a shell wrapper into another environment names no Python, so rigtune asks its own
    while the package it finds is the other environment's: the versions differ, and the answers are dropped."""
    pkg = _an_install(tmp_path, "0.33.0")
    assert ef.installed_version("vllm", pkg) == "0.33.0"
    runner, calls = _counting(ef.EngineFacts(engine="vllm", version="0.29.0", archs=["X"]))
    assert ef.cached_facts("vllm", "/py", pkg, runner=runner) is None
    assert not (probing / "engines").exists()  # not kept under the other install's name
    same, _ = _counting(ef.EngineFacts(engine="vllm", version="0.33.0+cu129", archs=["X"]))
    assert ef.cached_facts("vllm", "/py", pkg, runner=same).archs == ["X"]


def test_an_install_changed_below_its_package_folder_is_asked_again(tmp_path):
    """An editable install updated by `git pull` replaces files in subfolders, which does not move the package
    folder's own time; a reinstall rewrites its dist-info RECORD. Both change the key."""
    pkg = _an_install(tmp_path)
    spec = pkg / "config" / "speculative.py"
    spec.parent.mkdir()
    spec.write_text("SpeculativeMethod = 'old'\n", encoding="utf-8")
    before = ef.cache_key("vllm", "/py", pkg)
    st = spec.stat()
    os.utime(spec, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    after_pull = ef.cache_key("vllm", "/py", pkg)
    record = pkg.parent / "vllm-0.33.0.dist-info" / "RECORD"
    st = record.stat()
    os.utime(record, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    after_reinstall = ef.cache_key("vllm", "/py", pkg)
    assert len({before, after_pull, after_reinstall}) == 3


def test_answers_are_kept_through_a_file_of_the_writers_own(probing, tmp_path):
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    runner, _ = _counting()
    ef.cached_facts("vllm", "/py", pkg, runner=runner)
    assert [p.suffix for p in (probing / "engines").iterdir()] == [".json"]  # no shared .tmp left behind


def _path_forms(*paths):
    """Each path as it could appear in a file: native, POSIX, and JSON-escaped (doubled backslashes)."""
    out = set()
    for p in paths:
        for form in (str(p), Path(p).as_posix()):
            out |= {form, json.dumps(form)[1:-1]}
    return out


def test_kept_answers_name_no_folder_of_the_user(probing, tmp_path, monkeypatch):
    """An ImportError carries the module's file, a missing interpreter (on POSIX) its path, and a flag's default can
    be a folder in the user's home: every one is cut to its last part before it is kept."""
    site = tmp_path / "home" / "alice" / "venv" / "lib" / "site-packages"
    serve = engine_stubs.VLLM["vllm/entrypoints/cli/serve.py"].replace(
        "        return p\n", f"        p.add_argument('--download-dir', default={str(site / 'hub')!r})\n        return p\n")
    root = engine_stubs.build("vllm", site, replace={"vllm/config/cache.py": "X = 1\n",
                                                     "vllm/entrypoints/cli/serve.py": serve})
    monkeypatch.setenv("PYTHONPATH", str(root))
    kept = ef.cached_facts("vllm", sys.executable, root / "vllm")
    assert "kv_dtypes" in kept.errors and kept.flags["--download-dir"]["default"] == "<path>/hub"
    missing = ef.run_probe("vllm", str(tmp_path / "home" / "alice" / "envs" / "vllm" / "bin" / "python"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in (probing / "engines").glob("*.json"))
    text += "\n" + missing.model_dump_json()
    forms = _path_forms(tmp_path, site, Path(sys.executable).parent) | {"alice", "site-packages"}
    assert sorted(f for f in forms if f in text) == []
