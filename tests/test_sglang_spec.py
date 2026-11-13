"""Speculative decoding on SGLang: which methods the installed release has, read from its source."""

from __future__ import annotations

import subprocess
import sys

import pytest

import rigtune.backends.sglang as sg

# The lines that matter, as each release's argument parser has them.
SERVER_ARGS_0_5_2 = '''
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"],
            help="Speculative algorithm.",
        )
'''
SERVER_ARGS_0_5_3 = '''
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_bfs_breadth: int = 10
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],
        )
'''
SERVER_ARGS_0_5_19 = '''
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            help="Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK",
        )
'''
SPECULATIVE_HOOK_0_5_19 = '''
    if server_args.speculative_algorithm == "NGRAM":
        server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
'''
SERVER_ARGS_0_4 = '''choices=["EAGLE", "EAGLE3", "NEXTN"]'''


def fake_sglang(root, server_args: str, hooks: str = ""):
    srt = root / "sglang" / "srt"
    srt.mkdir(parents=True)
    (root / "sglang" / "__init__.py").write_text("raise ImportError('a scan must not import sglang')\n")
    (srt / "server_args.py").write_text(server_args)
    if hooks:
        (srt / "arg_groups").mkdir()
        (srt / "arg_groups" / "speculative_hook.py").write_text(hooks)
    return root / "sglang"


@pytest.fixture
def fresh():
    sg.sglang_spec_methods.cache_clear()
    yield
    sg.sglang_spec_methods.cache_clear()


@pytest.mark.parametrize("server_args, hooks, expected", [
    (SERVER_ARGS_0_4, "", set()),  # before 0.5.2: not checked, so nothing
    (SERVER_ARGS_0_5_2, "", set()),  # a second model only where 0.5.19's behaviour was read
    (SERVER_ARGS_0_5_3, "", {"ngram"}),
    (SERVER_ARGS_0_5_19, SPECULATIVE_HOOK_0_5_19, {"eagle3", "draft", "ngram"}),
])
def test_the_methods_come_from_the_installed_source(monkeypatch, tmp_path, fresh, server_args, hooks, expected):
    fake_sglang(tmp_path, server_args, hooks)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    monkeypatch.setattr(sg, "sglang_python", lambda: sys.executable)
    assert sg.sglang_spec_methods() == expected
    assert "sglang" not in sys.modules  # found and read, never imported


def test_sglang_in_its_own_environment_is_located_by_its_python(monkeypatch, tmp_path, fresh):
    pkg = fake_sglang(tmp_path, SERVER_ARGS_0_5_19, SPECULATIVE_HOOK_0_5_19)
    calls = []

    def run(args, **kw):
        calls.append((args, kw.get("timeout")))
        return subprocess.CompletedProcess(args, 0, stdout=f"{pkg}\n", stderr="")

    monkeypatch.setattr(sg, "sglang_python", lambda: "/envs/sgl/bin/python")
    monkeypatch.setattr(sg.subprocess, "run", run)
    assert sg.sglang_spec_methods() == {"eagle3", "draft", "ngram"}
    assert calls[0][0][0] == "/envs/sgl/bin/python" and calls[0][1] == 60


def test_no_sglang_offers_nothing(monkeypatch, fresh):
    monkeypatch.setattr(sg, "sglang_python", lambda: None)
    assert sg.sglang_spec_methods() == frozenset()
    sg.sglang_spec_methods.cache_clear()
    monkeypatch.setattr(sg, "sglang_python", lambda: "/envs/sgl/bin/python")
    monkeypatch.setattr(sg.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "boom"))
    assert sg.sglang_spec_methods() == frozenset()
