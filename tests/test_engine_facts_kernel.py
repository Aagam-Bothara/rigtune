"""vLLM's own Python is asked which graph and compile settings it has; an answer that is missing, empty or of the
wrong shape is no answer, and never "this engine has none".

Run against a stub package shaped like vLLM at the module paths its source was read at (v0.29.0 and v0.33.0); the
probe has not been run against a real install.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

from rigtune.backends import facts as ef
from tests import engine_stubs

ANSWER = {"cudagraph_modes": ["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
          "compilation_modes": ["NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE"],
          "compilation_fields": ["cudagraph_capture_sizes", "cudagraph_mode", "max_cudagraph_capture_size", "mode"],
          "pass_fields": ["eliminate_noops", "fuse_norm_quant"],
          "attention_backends": ["FLASH_ATTN", "TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION"],
          "opt_levels": ["O0", "O1", "O2", "O3"]}


def _line(key: str, value) -> str:
    return ef.MARK + json.dumps({"key": key, "value": value})


def _probe(tmp_path, monkeypatch, replace=None) -> ef.EngineFacts:
    root = engine_stubs.build("vllm", tmp_path / "stub", replace=replace)
    monkeypatch.setenv("PYTHONPATH", str(root))
    return ef.run_probe("vllm", sys.executable)


def test_facts_expose_cudagraph_modes(tmp_path, monkeypatch):
    f = _probe(tmp_path, monkeypatch)
    assert f.kernel == ANSWER and f.answered("kernel") and f.unanswered() == []
    assert "FULL_AND_PIECEWISE" in f.kernel["cudagraph_modes"]  # aliases of a value are still names the engine takes


def test_kernel_answer_parsed():
    f = ef.parse("vllm", _line("kernel", ANSWER), "no answer")
    assert f.kernel == ANSWER and "kernel" not in f.errors


@pytest.mark.parametrize("part", ef.KERNEL_PARTS)
def test_empty_modes_list_is_an_error_not_an_answer(part):
    for broken in ({**ANSWER, part: []}, {k: v for k, v in ANSWER.items() if k != part}):
        f = ef.parse("vllm", _line("kernel", broken), "no answer")
        assert f.kernel is None and not f.answered("kernel")
        assert f.errors["kernel"] == f"an empty answer, taken as none: {part}"
    assert ef.parse("vllm", _line("kernel", {}), "no answer").errors["kernel"] == "an empty answer, taken as none"


def test_missing_kernel_line_leaves_other_answers(tmp_path, monkeypatch):
    f = _probe(tmp_path, monkeypatch, replace={"vllm/config/compilation.py": None})
    assert f.kernel is None and f.errors["kernel"].startswith("ModuleNotFoundError")
    assert f.unanswered() == ["kernel"]  # every other question still has its answer
    assert f.flags is not None and f.kv_dtypes is not None and f.archs is not None


def test_a_renamed_enum_is_no_answer(tmp_path, monkeypatch):
    renamed = engine_stubs.VLLM["vllm/config/compilation.py"].replace("class CUDAGraphMode", "class GraphMode")
    f = _probe(tmp_path, monkeypatch, replace={"vllm/config/compilation.py": renamed})
    assert f.kernel is None and f.errors["kernel"].startswith("ImportError")


@pytest.mark.parametrize("value", [["FULL", "NONE"], "FULL", {"cudagraph_modes": "FULL"}, {"cudagraph_modes": [1]}])
def test_wrong_shape_is_unreadable_answer(value):
    f = ef.parse("vllm", _line("kernel", value), "no answer")
    assert f.kernel is None and f.errors["kernel"].startswith("unreadable answer")


def test_old_kept_answers_read_unanswered():
    """Answers kept before this question existed say nothing about it, and nothing else of theirs is lost."""
    f = ef.EngineFacts.model_validate({"engine": "vllm", "version": "0.33.0", "kv_dtypes": ["auto"]})
    assert f.kernel is None and not f.answered("kernel") and f.kv_dtypes == ["auto"]


def test_sglang_is_not_asked():
    assert "kernel" not in ef.KEYS["sglang"] and "kernel" in ef.KEYS["vllm"]


def test_cache_key_changes_with_compilation_source_mtime(tmp_path):
    root = engine_stubs.build("vllm", tmp_path / "stub")
    package = root / "vllm"
    before = ef.cache_key("vllm", sys.executable, package)
    source = package / "config" / "compilation.py"
    stamp = source.stat().st_mtime + 100
    os.utime(source, (stamp, stamp))
    assert ef.cache_key("vllm", sys.executable, package) != before  # an install edited in place is asked again


def test_probe_script_is_stdlib_only_and_compiles():
    tree = ast.parse(ef.script_path("vllm").read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    top |= {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert top <= set(sys.stdlib_module_names)  # the engine is imported inside each question, never at the top
