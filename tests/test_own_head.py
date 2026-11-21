"""rigtune.backends.own_head: the engine itself says whether it has an MTP draft for a model's architecture.

The fake engines below do what the real ones were read to do (vLLM v0.33.0 config/speculative.py hf_config_override
and transformers_utils/config.py get_config; SGLang v0.5.23 configs/model_config.py ModelConfig), including the parts
that would fool a careless probe: vLLM's Inkling rule returns a different object, get_config turns every failure to
load into a ValueError and fills in a missing `architectures`, and SGLang's __init__ can raise after the rename.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from rigtune.backends import facts as ef
from rigtune.backends import own_head as oh

# What each fake engine's config loader returns, by model id: (architectures, model_type, extra fields).
CONFIGS = {
    "Qwen/Qwen3.5-35B-A3B": (["Qwen3_5MoeForConditionalGeneration"], "qwen3_5_moe", {"mtp_num_hidden_layers": 1}),
    "thinkingmachines/Inkling-Small": (["InklingForConditionalGeneration"], "inkling_mm_model",
                                       {"mtp_config": {"num_nextn_predict_layers": 8}}),
    "acme/inkling-no-mtp": (["InklingForConditionalGeneration"], "inkling_mm_model", {"mtp_config": {}}),
    "acme/deepseek-no-nextn": (["DeepseekV3ForCausalLM"], "deepseek_v3", {}),
    "acme/no-architectures": (None, "llama", {}),
    "acme/raises-later": (["InklingForConditionalGeneration"], "inkling_mm_model",
                          {"mtp_config": {"num_nextn_predict_layers": 8}}),
    "thinkingmachines/Inkling-Small-banded": (["InklingForConditionalGeneration"], "inkling_mm_model",
                                              {"mtp_config": {"num_nextn_predict_layers": 8,
                                                              "local_layer_ids": [0, 2, 4, 5, 6, 7]}}),
    "meta-llama/Llama-3.1-8B-Instruct": (["LlamaForCausalLM"], "llama", {}),
}

FAKE_VLLM = {
    "vllm/__init__.py": "",
    "vllm/config/__init__.py": "",
    "vllm/config/speculative.py": """
        import types
        class SpeculativeConfig:
            @staticmethod
            def hf_config_override(c):
                if c.model_type == "qwen3_5_moe":
                    c.model_type = "qwen3_5_mtp"
                    c.n_predict = c.mtp_num_hidden_layers
                    c.architectures = ["Qwen3_5MoeMTP"]
                if c.model_type == "deepseek_v3":
                    c.model_type = "deepseek_mtp"
                    c.n_predict = getattr(c, "num_nextn_predict_layers", None)
                    c.architectures = ["DeepSeekMTPModel"]
                if c.model_type == "inkling_mm_model":  # returns its text config, a different object (line 1089)
                    depths = (getattr(c, "mtp_config", None) or {}).get("num_nextn_predict_layers", 0)
                    if depths < 1:
                        raise ValueError("The Inkling checkpoint does not contain MTP weights")
                    c.architectures = ["not-the-draft"]
                    return types.SimpleNamespace(model_type="inkling_mtp", n_predict=depths,
                                                 architectures=["InklingMTPModel"])
                return c
        """,
    "vllm/transformers_utils/__init__.py": "",
    "vllm/transformers_utils/config.py": """
        import json, os, types
        FILL = {"llama": "LlamaForCausalLM"}  # get_config fills a missing architectures (lines 939-949)
        def get_config(model, trust_remote_code, revision=None, **kw):
            assert trust_remote_code is False, "the probe must not run a repository's code"
            want = os.environ.get("EXPECT_REVISION")
            assert want is None or revision == want, f"revision {revision!r}"
            configs = json.loads(os.environ["FAKE_CONFIGS"])
            if model not in configs:  # every failure to load (lines 889-923)
                raise ValueError(f"Invalid repository ID or local directory specified: '{model}'.")
            archs, model_type, extra = configs[model]
            return types.SimpleNamespace(architectures=list(archs or [FILL[model_type]]), model_type=model_type,
                                         **extra)
        """,
}

FAKE_SGLANG = {
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/configs/__init__.py": "",
    "sglang/srt/configs/model_config.py": """
        import json, os, types
        RENAME = {"InklingForConditionalGeneration": "InklingForConditionalGenerationMTP",
                  "Qwen3_5MoeForConditionalGeneration": "Qwen3_5ForCausalLMMTP",
                  "Qwen3MoeForCausalLM": "Qwen3MoeForCausalLMMTP"}
        class ModelConfig:
            def __init__(self, model_path, trust_remote_code=True, revision=None, is_draft_model=False,
                         speculative_algorithm=None, **kw):
                assert trust_remote_code is False, "the probe must not run a repository's code"
                assert is_draft_model and speculative_algorithm == "EAGLE"
                configs = json.loads(os.environ["FAKE_CONFIGS"])
                if model_path not in configs:
                    raise ValueError(f"Unrecognized model in {model_path}")
                archs, model_type, extra = configs[model_path]
                self.hf_config = types.SimpleNamespace(architectures=list(archs or []), model_type=model_type)
                # configs/inkling.py copies mtp_config.local_layer_ids onto the text config (lines 337-342)
                local = (extra.get("mtp_config") or {}).get("local_layer_ids", [])
                self.hf_text_config = types.SimpleNamespace(mtp_local_layer_ids=local)
                self.is_draft_model = is_draft_model
                self._config_draft_model()
                if model_path == "acme/raises-later":  # a check after the rename (_verify_quantization and others)
                    raise ValueError("Unknown quantization method")
            def _config_draft_model(self):
                arch = self.hf_config.architectures[0]
                if self.is_draft_model and arch in RENAME:
                    self.hf_config.architectures[0] = RENAME[arch]
                    self.hf_text_config.num_nextn_predict_layers = 1  # set whatever the checkpoint holds
        """,
}

BROKEN = {"vllm/__init__.py": "raise ImportError('vllm is broken here')",
          "sglang/__init__.py": "raise ImportError('sglang is broken here')"}


@pytest.fixture
def engine_tree(tmp_path, monkeypatch):
    """Write a fake engine package and put it first on PYTHONPATH for the probe script."""
    def make(files):
        root = tmp_path / "site"
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text), encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(root))
        monkeypatch.setenv("FAKE_CONFIGS", json.dumps(CONFIGS))
        return root
    return make


def _probe(engine, model, revision=None):
    return oh.run_head_probe(engine, sys.executable, model, revision)


# ---------------------------------------------------------------- vLLM


def test_vllm_has_a_draft_for_qwen35_moe(engine_tree):
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "Qwen/Qwen3.5-35B-A3B")
    assert (a.target_arch, a.draft_arch, a.layers, a.errors) == ("Qwen3_5MoeForConditionalGeneration",
                                                                 "Qwen3_5MoeMTP", 1, {})
    assert oh.verdict(a, {"Qwen3_5MoeMTP"}).loads is True and a.settled()


def test_vllm_reads_the_draft_its_rewrite_returns_not_the_config_it_was_given(engine_tree):
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "thinkingmachines/Inkling-Small")
    assert (a.draft_arch, a.layers) == ("InklingMTPModel", 8)  # the rewrite's returned object, from mtp_config


def test_a_refusal_by_vllms_rewrite_is_a_no_and_kept(engine_tree):
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "acme/inkling-no-mtp")
    v = oh.verdict(a)
    assert v.loads is False and "does not contain MTP weights" in v.why and a.settled()


def test_a_config_that_cannot_be_loaded_is_unknown_and_never_kept(engine_tree):
    """vLLM's get_config raises ValueError for every failure to load (a gated repo, a bad revision, the Hub down):
    that must never read as vLLM refusing the head."""
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "acme/not-on-the-hub")
    assert a.target_arch is None and a.errors["target_arch"].startswith("ValueError")
    assert oh.verdict(a).loads is None and not a.settled() and not a.refused()


def test_the_architecture_compared_is_the_one_the_engine_serves(engine_tree):
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "acme/no-architectures")  # config.json names none; vLLM fills LlamaForCausalLM
    assert a.target_arch == a.draft_arch == "LlamaForCausalLM"
    assert oh.verdict(a).loads is False


def test_a_vllm_rule_that_finds_no_layer_count_is_unknown(engine_tree):
    engine_tree(FAKE_VLLM)
    a = _probe("vllm", "acme/deepseek-no-nextn")
    assert a.draft_arch == "DeepSeekMTPModel" and a.layers is None
    assert oh.verdict(a).loads is None


def test_the_revision_reaches_the_engines_config_loader(engine_tree, monkeypatch):
    engine_tree(FAKE_VLLM)
    monkeypatch.setenv("EXPECT_REVISION", "2508b24")
    assert _probe("vllm", "Qwen/Qwen3.5-35B-A3B", "2508b24").draft_arch == "Qwen3_5MoeMTP"
    assert _probe("vllm", "Qwen/Qwen3.5-35B-A3B", "c202236").errors["target_arch"].startswith("AssertionError")


# ---------------------------------------------------------------- SGLang


def test_sglang_has_a_draft_for_inkling(engine_tree):
    engine_tree(FAKE_SGLANG)
    a = _probe("sglang", "thinkingmachines/Inkling-Small")
    assert (a.target_arch, a.draft_arch) == ("InklingForConditionalGeneration", "InklingForConditionalGenerationMTP")
    assert oh.verdict(a, {"InklingForConditionalGenerationMTP"}).loads is True


def test_sglang_says_when_a_head_is_banded(engine_tree):
    """Inkling's banded head runs only on SGLang's multi-layer EAGLE worker (models/inkling.py 1792-1797)."""
    engine_tree(FAKE_SGLANG)
    banded = _probe("sglang", "thinkingmachines/Inkling-Small-banded")
    plain = _probe("sglang", "thinkingmachines/Inkling-Small")
    assert (banded.banded, plain.banded) == (True, False)
    assert oh.verdict(banded).banded is True and oh.verdict(banded).loads is True
    kept = oh.HeadAnswer.model_validate_json(oh.HeadAnswer(engine="sglang", target_arch="X", draft_arch="Y")
                                             .model_dump_json(exclude={"banded"}))
    assert oh.verdict(kept).banded is False  # an answer kept before the field reads as not banded


def test_sglangs_answer_stands_when_its_init_raises_after_the_rename(engine_tree):
    engine_tree(FAKE_SGLANG)
    a = _probe("sglang", "acme/raises-later")
    assert a.draft_arch == "InklingForConditionalGenerationMTP" and oh.verdict(a).loads is True


def test_no_sglang_error_is_a_refusal(engine_tree):
    """SGLang's rename never raises (model_config.py lines 1010-1214 at v0.5.23); a ValueError is something else."""
    engine_tree(FAKE_SGLANG)
    a = _probe("sglang", "acme/not-on-the-hub")
    assert a.errors["target_arch"].startswith("ValueError") and oh.verdict(a).loads is None and not a.refused()


def test_sglang_has_no_draft_for_a_plain_llama(engine_tree):
    engine_tree(FAKE_SGLANG)
    assert oh.verdict(_probe("sglang", "meta-llama/Llama-3.1-8B-Instruct")).loads is False


def test_an_sglang_that_never_reaches_the_rename_is_unknown(engine_tree):
    files = dict(FAKE_SGLANG)
    files["sglang/srt/configs/model_config.py"] = """
        class ModelConfig:
            def __init__(self, *a, **kw):
                pass  # a release that builds the draft's config elsewhere
        """
    engine_tree(files)
    a = _probe("sglang", "thinkingmachines/Inkling-Small")
    assert "not reached" in a.errors["target_arch"] and oh.verdict(a).loads is None and not a.settled()


def test_a_valueerror_while_vllm_loads_is_no_refusal(engine_tree):
    """vLLM's config package rejects an environment variable outside its choices with a ValueError at import: that
    happens in the config step, never in the rewrite's."""
    files = dict(FAKE_VLLM)
    files["vllm/config/speculative.py"] = "raise ValueError(\"VLLM_SOMETHING='x' is not one of its choices\")\n"
    engine_tree(files)
    a = _probe("vllm", "Qwen/Qwen3.5-35B-A3B")
    assert a.errors["target_arch"].startswith("ValueError") and not a.refused() and oh.verdict(a).loads is None


def test_the_probe_says_which_config_libraries_answered(engine_tree):
    """The versions the engine's Python has; none when one of them is missing (CI installs no transformers), and an
    answer without them is never kept."""
    engine_tree(FAKE_VLLM)
    assert _probe("vllm", "Qwen/Qwen3.5-35B-A3B").libs == (oh.config_libs(sys.executable) or {})


def test_a_step_that_was_due_and_left_no_answer_is_an_error():
    class Out:
        returncode, stderr = -9, b"Killed"
        stdout = ("\n" + ef.MARK + '{"key": "target_arch", "value": "Qwen3_5MoeForConditionalGeneration"}\n'
                  + ef.MARK + '{"key": "draft_arch", "value": "Qwen3_5MoeMTP"}\n').encode()

    a = oh.run_head_probe("vllm", sys.executable, "Qwen/Qwen3.5-35B-A3B", "2508b24", run=lambda *a, **k: Out())
    assert "exited -9" in a.errors["layers"] and not a.settled() and oh.verdict(a).loads is None


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_broken_engine_install_is_unknown(engine, engine_tree):
    engine_tree(BROKEN)
    a = _probe(engine, "Qwen/Qwen3.5-35B-A3B")
    assert "ImportError" in a.errors["target_arch"] and oh.verdict(a).loads is None and not a.settled()


# ---------------------------------------------------------------- the verdict alone


LIBS = {"transformers": "5.20.2", "huggingface_hub": "1.2.0"}


def _answer(**kw):
    return oh.HeadAnswer(**{"engine": "vllm", "target_arch": "Qwen3_5MoeForConditionalGeneration", "libs": LIBS,
                            **kw})


@pytest.mark.parametrize("answer, registered, loads, says", [
    (_answer(draft_arch="Qwen3_5MoeMTP", layers=1), {"Qwen3_5MoeMTP"}, True, "1 MTP layer"),
    (_answer(draft_arch="Qwen3_5MoeMTP", layers=1), None, True, "builds Qwen3_5MoeMTP"),  # registry unanswered
    (_answer(draft_arch="Qwen3_5MoeForConditionalGeneration"), {"Qwen3_5MoeForConditionalGeneration"}, False,
     "no MTP draft for"),
    # The architecture has no class of its own: vLLM would run it through its generic backend, not asked here.
    (_answer(draft_arch="Qwen3_5MoeForConditionalGeneration"), {"LlamaForCausalLM"}, None, "generic backend"),
    (_answer(draft_arch="Qwen3_5MoeMTP", layers=0), None, False, "reads no MTP layers"),
    (_answer(draft_arch="Qwen3_5MoeMTP", layers=1), {"LlamaForCausalLM"}, None, "registry does not list"),
    (_answer(errors={"draft_arch": "AttributeError: no hf_config_override"}), None, None, "could not be asked"),
    (oh.HeadAnswer(engine="sglang", target_arch="Qwen3MoeForCausalLM", draft_arch="Qwen3MoeForCausalLMMTP", layers=1),
     None, True, "builds Qwen3MoeForCausalLMMTP"),
    (None, None, None, "not asked"),
])
def test_only_a_positive_finding_says_no(answer, registered, loads, says):
    v = oh.verdict(answer, registered)
    assert v.loads is loads and says in v.why


# ---------------------------------------------------------------- what is kept


@pytest.fixture
def probing(monkeypatch, tmp_path, tmp_home):
    monkeypatch.delenv("RIGTUNE_ENGINE_PROBE", raising=False)
    pkg = tmp_path / "site-packages" / "vllm"
    pkg.mkdir(parents=True)
    return pkg


def _counting(answer):
    calls = []

    def runner(engine, python, model, revision):
        calls.append((model, revision))
        return answer
    return calls, runner


def _ask(pkg, runner, model="Qwen/Qwen3.5-35B-A3B", revision="2508b24", libs=LIBS):
    return oh.ask("vllm", sys.executable, pkg, model, revision, runner=runner, libs_of=lambda python: libs)


def test_a_settled_answer_at_a_pinned_revision_is_asked_once_and_kept(probing, tmp_home):
    calls, runner = _counting(_answer(draft_arch="Qwen3_5MoeMTP", layers=1))
    for _ in range(2):
        assert _ask(probing, runner)[0].loads is True
    _ask(probing, runner, revision="c202236")
    assert calls == [("Qwen/Qwen3.5-35B-A3B", "2508b24"), ("Qwen/Qwen3.5-35B-A3B", "c202236")]
    assert len(list((tmp_home / "engines" / "heads").glob("*.json"))) == 2


@pytest.mark.parametrize("answer", [
    _answer(errors={"target_arch": "ValueError: Invalid repository ID"}, target_arch=None),  # could not load
    _answer(errors={"draft_arch": f"{ef.TIMED_OUT} 180 s"}),
    _answer(errors={"draft_arch": "ImportError: no module"}),
])
def test_an_answer_that_settles_nothing_is_asked_again(probing, answer):
    calls, runner = _counting(answer)
    for _ in range(2):
        assert _ask(probing, runner)[0].loads is None
    assert len(calls) == 2


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "refs/pr/3"])
def test_a_model_at_no_commit_is_asked_again(probing, revision):
    calls, runner = _counting(_answer(draft_arch="Qwen3_5MoeMTP", layers=1))
    for _ in range(2):
        _ask(probing, runner, revision=revision)
    assert len(calls) == 2


def test_an_answer_that_does_not_say_which_libraries_gave_it_is_not_kept(probing):
    calls, runner = _counting(_answer(draft_arch="Qwen3_5MoeMTP", layers=1, libs={}))
    for _ in range(2):
        _ask(probing, runner)
    assert len(calls) == 2


def test_a_new_probe_script_or_config_library_asks_again(probing, tmp_path, monkeypatch):
    calls, runner = _counting(_answer(draft_arch="Qwen3_5MoeMTP", layers=1))
    _ask(probing, runner)
    _ask(probing, runner, libs={**LIBS, "transformers": "5.21.0"})  # upgraded, wherever it is installed
    _ask(probing, runner, libs=None)  # the Python cannot say: ask again rather than trust the kept answer
    probes = tmp_path / "probes"
    shutil.copytree(ef.PROBES, probes)
    monkeypatch.setattr(ef, "PROBES", probes)
    (probes / "vllm_mtp_probe.py").write_text("# asks otherwise\n", encoding="utf-8")
    _ask(probing, runner)
    assert len(calls) == 4


def test_config_libs_asks_the_engines_python():
    found = oh.config_libs(sys.executable)
    assert found is None or set(found) == set(oh.CONFIG_LIBS)
    assert oh.config_libs("/no/such/python") is None


def test_a_local_checkpoint_is_asked_by_its_absolute_path_and_again_when_its_config_changes(probing, tmp_path,
                                                                                             monkeypatch):
    model = tmp_path / "models" / "mine"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls, runner = _counting(_answer(draft_arch="Qwen3_5MoeMTP", layers=1))
    _ask(probing, runner, model=os.path.join("models", "mine"), revision=None)
    _ask(probing, runner, model=os.path.join("models", "mine"), revision=None)  # kept: a local directory
    os.utime(model / "config.json", ns=(1, 1))
    _ask(probing, runner, model=os.path.join("models", "mine"), revision=None)
    assert calls == [(str(model.resolve()), None)] * 2


def test_probing_off_asks_nothing(tmp_path):
    def runner(*a):
        raise AssertionError("asked although probing is off")

    v, answer = oh.ask("vllm", sys.executable, tmp_path, "Qwen/Qwen3.5-35B-A3B", "2508b24", runner=runner)
    assert answer is None and v.loads is None


def test_errors_kept_on_disk_name_no_folder(engine_tree, tmp_path):
    root = engine_tree(FAKE_VLLM)
    (root / "vllm" / "transformers_utils" / "config.py").write_text(
        f"raise ImportError('cannot import from {tmp_path / 'secret' / 'home'}')\n", encoding="utf-8")
    a = _probe("vllm", "Qwen/Qwen3.5-35B-A3B")
    assert "secret" not in a.errors["target_arch"] and a.target_arch is None


def test_the_probe_scripts_take_the_model_and_its_revision():
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        raise OSError("not run")

    a = oh.run_head_probe("sglang", "/opt/sglang/bin/python", "Qwen/Qwen3.5-35B-A3B", "2508b24", run=run)
    assert seen["argv"][0] == "/opt/sglang/bin/python" and Path(seen["argv"][1]).name == "sglang_mtp_probe.py"
    assert seen["argv"][2:] == ["Qwen/Qwen3.5-35B-A3B", "2508b24"]
    assert "could not run" in a.errors["target_arch"] and oh.verdict(a).loads is None
