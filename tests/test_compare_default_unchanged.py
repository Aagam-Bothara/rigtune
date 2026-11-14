"""`rigtune compare` without any of the comparison-harness options writes what it wrote before them.

tests/golden/compare/default.json and default.md were captured from deecc92, before `compare` had any option
for added rows: the full command, run with fakes (a cached-profile stand-in for calibration and a fake trial
runner for every launch), three interleaved repeats, results saved to --out. The test runs the same command on
today's code and compares the saved file and its Markdown with them byte for byte, and checks that nothing
else was written beside the results.

    python -m tests.test_compare_default_unchanged    # rewrite the golden files from the code as it is
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

GOLDEN = Path(__file__).parent / "golden" / "compare"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def run_default_compare(out: Path, monkeypatch) -> Path:
    """`rigtune compare MODEL --backend vllm --repeats 3 --out OUT` against fakes; the saved results file."""
    import importlib

    import rigtune.hardware as hwmod
    import rigtune.pipeline as pl
    from rigtune.cli import app
    from rigtune.pipeline import calibrate, prepare_and_plan, select
    from tests.conftest import make_hw
    from tests.test_compare_repeats import DriftingRunner
    from tests.test_objectives_and_search import FakeRunner

    import rigtune.backends.vllm as vl

    cmp = importlib.import_module("rigtune.bench.compare")  # the package exports a function of the same name
    # A launch command starts with the `vllm` CLI where one is installed, else with this interpreter: a fact
    # about the machine, not the code. Pin it to "no CLI", as on the machine that captured the golden files.
    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    hw = make_hw("a100")
    monkeypatch.setattr(hwmod, "probe", lambda: hw)

    def cached_profile(spec, objective, force_backend=None, **kwargs):
        candidates, reg = select(hw, spec, force=force_backend)
        planned = prepare_and_plan(hw, spec, candidates, reg)
        profile = calibrate(hw, spec, objective, planned, reg, runner=FakeRunner())
        profile.calibration_seconds = 1234.0  # wall time of the fake calibration, which differs between runs
        return profile

    monkeypatch.setattr(pl, "resolve_profile", cached_profile)
    runner = DriftingRunner()
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kwargs: runner)
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--repeats", "3", "--out", str(out)])
    assert r.exit_code == 0, r.output
    files = sorted(p.name for p in out.iterdir())
    assert len(files) == 1, files  # the results file and nothing beside it
    return out / files[0]


PYTHON = "<sys.executable>"


def normalised(path: Path) -> str:
    """The saved results with the fields that differ between runs or machines fixed: when, which rigtune, and
    the interpreter path each launch command starts with (C:\\...\\python.exe on Windows,
    /usr/local/bin/python in a Linux container). That path is replaced only where it is exactly this
    interpreter, so any other first argument still shows up as a difference."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["created_at"] = 0.0
    data["rigtune_version"] = "x"
    for row in data["rows"]:
        args = row["launch_args"]
        if args and args[0] == sys.executable:
            args[0] = PYTHON
    return json.dumps(data, indent=2) + "\n"


def markdown(path: Path) -> str:
    from rigtune.bench.compare import ComparisonResult, to_markdown

    return to_markdown(ComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))) + "\n"


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_default_compare_writes_what_it_wrote_before_the_harness_options(tmp_path, monkeypatch):
    path = run_default_compare(tmp_path / "out", monkeypatch)
    assert path.name == (GOLDEN / "name.txt").read_text(encoding="utf-8").strip()
    assert normalised(path) == (GOLDEN / "default.json").read_text(encoding="utf-8")
    assert markdown(path) == (GOLDEN / "default.md").read_text(encoding="utf-8")


def _rewrite() -> None:  # pragma: no cover - run by hand
    import tempfile

    from _pytest.monkeypatch import MonkeyPatch

    from tests.conftest import LLAMA_3B_CONFIG

    mp = MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        mp.setenv("RIGTUNE_HOME", str(Path(tmp) / "home"))
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.delenv("HF_TOKEN", raising=False)
        import rigtune.backends.llamacpp as lc
        import rigtune.backends.sglang as sg
        import rigtune.backends.vllm as vl
        import rigtune.backends.vllm_cpu as vc
        import rigtune.quantized as qz
        import rigtune.speculative as sp
        from rigtune.hfconfig import arch_from_config

        arch = arch_from_config(LLAMA_3B_CONFIG)
        for mod in (lc, sg, vl, vc):
            mp.setattr(mod, "load_arch", lambda spec, token=None: arch)
        mp.setattr(vl, "vllm_registry_archs", lambda: None)
        mp.setattr(sg, "sglang_registry_archs", lambda: None)
        mp.setattr(lc, "search_hub_gguf", lambda spec, quants, **kw: {})
        mp.setattr(lc, "llama_server_binary", lambda: "/usr/local/bin/llama-server")
        mp.setattr(qz, "find_prequantized_repos", lambda spec, **kw: {})
        mp.setattr(sp, "find_eagle3_head", lambda spec, arch, **kw: None)
        mp.setattr(sp, "repo_bytes", lambda repo, token=None: None)
        path = run_default_compare(Path(tmp) / "out", mp)
        GOLDEN.mkdir(parents=True, exist_ok=True)
        for name, text in (("name.txt", path.name + "\n"), ("default.json", normalised(path)),
                           ("default.md", markdown(path))):
            (GOLDEN / name).write_bytes(text.encode("utf-8"))
        mp.undo()
    print(f"wrote {GOLDEN}", file=sys.stderr)


if __name__ == "__main__":
    _rewrite()
