"""Feature checks answered by the engine probe, falling back to the engine's source text.

Absent, failed and partial probes leave every check answering what it answered before the probe existed; where the
probe did answer, its answer wins, and it can take a speculative method away but never offer one that a version gate
or a measured failure holds back."""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from rigtune import speculative
from rigtune.backends import registry
from rigtune.backends.facts import KEYS, EngineFacts
from tests import golden_cases
from tests.engine_trees import installed
from tests.test_golden_launch import RELEASE_ANSWERS, _probed_answers

VLLM, SGLANG = ("vllm", "0.33.0"), ("sglang", "0.5.23")


def _failed(engine):
    return EngineFacts(engine=engine, errors={k: "no answer within 120 s" for k in KEYS[engine]})


def _flags(*options):
    return {o: {"choices": None, "default": None} for o in options}


@pytest.mark.parametrize("engine,version", [VLLM, SGLANG])
@pytest.mark.parametrize("probe", ["absent", "failed", "version only", "raises"])
def test_without_an_answer_every_check_reads_the_source_as_before(engine, version, probe, tmp_path):
    mod = vl if engine == "vllm" else sg
    answers = {"absent": lambda: None, "failed": lambda: _failed(engine),
               "version only": lambda: EngineFacts(engine=engine, version=version)}

    def raising():
        raise OSError("disk full")

    with installed(engine, version, tmp_path), mock.patch.object(mod, "_engine_facts", answers.get(probe, raising)):
        assert _probed_answers(engine) == RELEASE_ANSWERS[engine]


def test_a_partial_answer_is_used_where_it_answers_and_the_source_elsewhere(tmp_path):
    """SGLang's HiCache check needs flags and environment variables both; with flags alone it reads the source, while
    the cache-report check, which needs only flags, takes the probe's word (here: no such flag)."""
    partial = EngineFacts(engine="sglang", flags=_flags("--enable-hierarchical-cache", "--hicache-size"))
    with installed(*SGLANG, tmp_path), mock.patch.object(sg, "_engine_facts", lambda: partial):
        assert sorted(sg.feature("hicache")) == ["file", "file_cap", "hicache"]  # the source's answer
        assert sg.feature("cache_report") is False  # the probe's
        assert sg.feature("tier_details") is True and sorted(sg.feature("spec_methods")) == ["draft", "eagle3", "ngram"]


def test_the_probe_finds_a_flag_the_source_check_cannot(tmp_path):
    """vLLM 0.33 moved its server arguments; a source check that looks only where 0.29 kept them finds nothing, while
    the parser has the flag wherever it is defined."""
    from tests import golden_cases as gc

    facts = gc.facts_fixture(*VLLM)
    with installed(*VLLM, tmp_path) as pkg:
        (pkg / "entrypoints" / "common" / "cli_args.py").unlink()
        vl._prompt_tokens_details_flag.cache_clear()
        assert vl._prompt_tokens_details_flag() is False
        pm = gc.prepared("llama-3.1-8b", "vllm")
        be = registry(report_reuse=True)["vllm"]
        cfg = be.default_config(gc.hw("rtx4090", "vllm", "0.33.0"), pm)
        assert "--enable-prompt-tokens-details" not in be.launch_spec(cfg, pm, 8000).args
        with mock.patch.object(vl, "_engine_facts", lambda: facts):
            assert "--enable-prompt-tokens-details" in be.launch_spec(cfg, pm, 8000).args


def test_the_probes_no_wins_over_the_sources_yes_and_the_refusal_says_who_said_so(tmp_path):
    without = EngineFacts(engine="vllm", flags=_flags("--attention-backend"), features={
        "offloading_connector": True, "offload_prompt_only": False, "supports_hma": True, "shm_region": True})
    be = registry()["vllm"]
    pm = golden_cases.prepared("llama-3.1-8b", "vllm")
    cfg = be.default_config(golden_cases.hw("rtx4090", "vllm", "0.33.0"), pm).model_copy(update={"kv_tier": "host:16"})
    with installed(*VLLM, tmp_path):
        assert be.text_only_refusal() is None and be.tier_refusal(cfg, pm) is None
        with mock.patch.object(vl, "_engine_facts", lambda: without):
            assert be.text_only_refusal() == ("the vLLM that would run has no --language-model-only (asked in its "
                                              "own Python)")
            assert be.tier_refusal(cfg, pm).endswith("offload_prompt_only (asked in its own Python)")
            assert be.tier_host_cap() is None
    with installed(*VLLM, tmp_path / "bare") as pkg:
        (pkg / "engine" / "arg_utils.py").unlink()
        vl._language_model_only_flag.cache_clear()
        assert be.text_only_refusal() == "the vLLM that would run has no --language-model-only (read from its source)"


# ---------------------------------------------------------------- speculative methods


def _llama_with_head():
    pm = golden_cases.prepared("llama-3.1-8b", "vllm")
    return pm


def _kinds(variants):
    return sorted({speculative.parse(v.spec_decode)[0] for v in variants})


def test_a_method_vllm_does_not_list_is_not_offered(tmp_path):
    facts = golden_cases.facts_fixture(*VLLM)
    trimmed = facts.model_copy(update={"spec_methods": [m for m in facts.spec_methods
                                                        if m not in ("suffix", "eagle3", "ngram_gpu")]})
    be = registry()["vllm"]
    pm = _llama_with_head()
    card = golden_cases.hw("rtx4090", "vllm", "0.33.0")
    with installed(*VLLM, tmp_path):
        cfg = be.default_config(card, pm)
        full = be.spec_variants(cfg, pm, card, all_methods=True)
        with mock.patch.object(vl, "_engine_facts", lambda: facts):
            assert be.spec_variants(cfg, pm, card, all_methods=True) == full
        assert _kinds(full) == ["draft", "eagle3", "ngram_gpu", "suffix"]
        with mock.patch.object(vl, "_engine_facts", lambda: trimmed):
            # n-gram falls back to the CPU proposer, as on a release without the GPU one.
            assert _kinds(be.spec_variants(cfg, pm, card, all_methods=True)) == ["draft", "ngram"]
            assert "suffix decoding" not in " ".join(be.spec_notes(cfg, pm, card, all_methods=True))


def test_a_listed_method_is_still_held_back_by_its_version_gate(tmp_path):
    """vLLM 0.11's V1 engine refused a draft model and was never checked with EAGLE3 or suffix decoding: a probe that
    lists them does not change that, since the gate records what was checked, which a Literal cannot say."""
    facts = golden_cases.facts_fixture(*VLLM)
    with installed(*VLLM, tmp_path), mock.patch.object(vl, "_engine_facts", lambda: facts), \
            mock.patch.object(speculative, "_vllm_version", lambda: "0.11.0"):
        assert not speculative.vllm_supports_draft()
        assert not speculative.vllm_supports_eagle3()
        assert not speculative.vllm_suffix_release() and not speculative.vllm_supports_ngram_gpu()
    # A version passed explicitly is a question about that release, not the installed one: the probe is not asked.
    with mock.patch.object(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", spec_methods=[])):
        assert speculative.vllm_supports_draft("0.29.0") and speculative.vllm_supports_eagle3("0.33.0")


def test_a_measured_failure_stays_refused_whatever_the_probe_lists(tmp_path):
    """EAGLE3 with the fp8_e5m2 cache through FlashInfer failed on an A40 (VllmBackend.known_failure); the probe can say
    the method exists, never that the pair works."""
    facts = golden_cases.facts_fixture("vllm", "0.29.0")
    be = registry()["vllm"]
    pm = _llama_with_head()
    card = golden_cases.hw("a40", "vllm", "0.29.0")
    with installed("vllm", "0.29.0", tmp_path), mock.patch.object(vl, "_engine_facts", lambda: facts):
        cfg = be.default_config(card, pm).model_copy(update={"kv_dtype": vl.AMPERE_FP8_KV})
        assert "eagle3" not in _kinds(be.spec_variants(cfg, pm, card, all_methods=True))
        assert "fp8" not in be.precisions(card)  # FP8_ON_AMPERE_BROKEN_FROM, measured on 0.29


def test_a_kv_type_vllm_does_not_list_is_not_offered(tmp_path):
    facts = golden_cases.facts_fixture(*VLLM)
    no_int8 = facts.model_copy(update={"kv_dtypes": [d for d in facts.kv_dtypes if d != vl.INT8_KV]})
    card = golden_cases.hw("rtx4090", "vllm", "0.33.0")
    be = registry()["vllm"]
    with installed(*VLLM, tmp_path):
        assert vl.INT8_KV in be.kv_dtypes(card)
        with mock.patch.object(vl, "_engine_facts", lambda: facts):
            assert vl.INT8_KV in be.kv_dtypes(card)
        with mock.patch.object(vl, "_engine_facts", lambda: no_int8):
            assert be.kv_dtypes(card) == ["fp8"]


@pytest.mark.parametrize("algorithms,kinds", [
    (["DSPARK", "EAGLE", "EAGLE3", "NGRAM", "STANDALONE"], ["draft", "eagle3", "ngram"]),
    (["EAGLE", "EAGLE3", "NGRAM", "STANDALONE"], ["ngram"]),  # no DSPARK: second-model behaviour not read there
    (["DSPARK", "EAGLE3", "STANDALONE"], ["draft", "eagle3"]),
    (["EAGLE", "EAGLE3", "NGRAM"], []),  # before STANDALONE
])
def test_sglangs_algorithms_map_to_rigtunes_kinds_under_the_source_checks_gates(algorithms, kinds):
    assert sorted(sg.spec_methods_from(algorithms)) == kinds
    assert sg.spec_methods_from(None) is None


def test_probe_engines_shows_each_checks_two_answers_and_any_disagreement(tmp_path, monkeypatch):
    """Where the probe and the source both answer and differ, one of them is wrong: `rigtune probe --engines` says so
    (the GPU session's check that the two agree on real installs)."""
    from typer.testing import CliRunner

    from rigtune import cli
    from tests.conftest import make_hw

    disagree = golden_cases.facts_fixture(*VLLM).model_copy(update={"flags": _flags("--attention-backend")})
    monkeypatch.setattr("rigtune.hardware.probe", lambda: make_hw("rtx4090"))
    monkeypatch.setattr(cli, "_engine_facts", lambda refresh: {"vllm": disagree, "sglang": None})
    with installed(*VLLM, tmp_path), mock.patch.object(vl, "_engine_facts", lambda: disagree):
        report = vl.feature_report()
        assert report["attention_backend"] == (True, True)
        assert report["prompt_tokens_details"] == (False, True)
        out = CliRunner().invoke(cli.app, ["probe", "--engines"], env={"COLUMNS": "300"})
        as_json = CliRunner().invoke(cli.app, ["probe", "--engines", "--json"])
    assert out.exit_code == 0, out.output
    assert "the source disagrees" in out.output and "source (no probe answer)" in out.output  # SGLang: no probe
    checks = json.loads(as_json.output)["feature_checks"]
    assert checks["vllm"]["prompt_tokens_details"] == {"probe": False, "source": True}
    assert checks["sglang"]["hicache"] == {"probe": None, "source": []}


# ---------------------------------------------------------------- a failed or empty answer never beats the source


def test_an_offloading_module_that_fails_to_import_leaves_the_tier_to_the_source(tmp_path, monkeypatch):
    """The probe could not read /dev/shm's region (its module failed to import with no GPU visible): the source
    check's answer stands, and the host tier is still capped by free /dev/shm."""
    from rigtune.backends import facts as ef
    from tests import engine_stubs

    root = engine_stubs.build("vllm", tmp_path / "stub", replace={
        "vllm/v1/kv_offload/cpu/shared_offload_region.py": "raise ImportError('libcudart.so: cannot open')\n"})
    monkeypatch.setenv("PYTHONPATH", str(root))
    probed = ef.run_probe("vllm", sys.executable)
    assert probed.features is None
    with installed(*VLLM, tmp_path / "tree"), mock.patch.object(vl, "_engine_facts", lambda: probed), \
            mock.patch.object(vl, "_shm_free", lambda: 8 * 2**30):
        assert sorted(vl.feature("offload")) == ["cpu", "hma", "shm"]
        assert registry()["vllm"].tier_host_cap() == 8 * 2**30


def test_an_empty_list_of_methods_takes_none_away(tmp_path, monkeypatch):
    """A release whose SpeculativeMethod is no longer a Literal flattens to []: that is no answer, so the version
    gates decide as before, rather than every gated method disappearing."""
    from rigtune.backends import facts as ef
    from tests import engine_stubs

    enum_methods = "import enum\nclass SpeculativeMethod(str, enum.Enum):\n    NGRAM = 'ngram'\n"
    root = engine_stubs.build("vllm", tmp_path / "stub", replace={"vllm/config/speculative.py": enum_methods})
    monkeypatch.setenv("PYTHONPATH", str(root))
    probed = ef.run_probe("vllm", sys.executable)
    with mock.patch.object(vl, "_engine_facts", lambda: probed), \
            mock.patch.object(speculative, "_vllm_version", lambda: "0.33.0"):
        assert speculative.vllm_supports_draft() and speculative.vllm_supports_eagle3()
        assert speculative.vllm_supports_ngram_gpu() and speculative.vllm_suffix_release()
