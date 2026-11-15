"""Supported architectures from the engine probe, before the built-in list.

The probe lists the registry of the engine that will run, in its own Python: gpt-oss is in both engines' registries
at every release rigtune was read against, and in neither rigtune's fallback list, so without the probe SGLang behind
$SGLANG_PYTHON (whose registry rigtune cannot import) refuses it. The probe only ever adds an architecture."""

from __future__ import annotations

import hashlib
from unittest import mock

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from rigtune.backends.facts import EngineFacts
from rigtune.models import ModelSpec
from tests import golden_cases
from tests.conftest import make_hw

GPT_OSS = "GptOssForCausalLM"
RELEASES = [("vllm", "0.29.0"), ("vllm", "0.33.0"), ("sglang", "0.5.19"), ("sglang", "0.5.23")]


def _supports(engine, arch_name, hw=None):
    """`engine`'s answer for a model of architecture `arch_name`, with no importable registry here (conftest's
    no_network does the same)."""
    mod = vl if engine == "vllm" else sg
    arch = golden_cases.prepared("llama-3.1-8b", engine).arch.model_copy(update={"architecture": arch_name})
    backend = vl.VllmBackend() if engine == "vllm" else sg.SglangBackend()
    with mock.patch.object(mod, "load_arch", lambda spec, token=None: arch), \
            mock.patch.object(vl, "vllm_registry_archs", lambda: None), \
            mock.patch.object(sg, "sglang_registry_archs", lambda: None):
        return backend.supports(hw or make_hw("rtx4090"), ModelSpec(hf_id="org/model"))


@pytest.mark.parametrize("engine,version", RELEASES)
def test_an_architecture_only_the_engines_registry_lists_is_offered(engine, version):
    facts = golden_cases.facts_fixture(engine, version)
    assert GPT_OSS in facts.archs and GPT_OSS not in vl.FALLBACK_ARCHS
    mod = vl if engine == "vllm" else sg
    assert not _supports(engine, GPT_OSS)
    with mock.patch.object(mod, "_engine_facts", lambda: facts):
        assert _supports(engine, GPT_OSS)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_probe_never_takes_an_architecture_away(engine):
    """A registry listed with no GPU visible can miss a model whose module fails to import (SGLang skips it), and no
    vLLM plugin is loaded: an architecture the fallback list or an importable registry has is still offered."""
    mod = vl if engine == "vllm" else sg
    narrow = EngineFacts(engine=engine, archs=[GPT_OSS])
    with mock.patch.object(mod, "_engine_facts", lambda: narrow):
        assert _supports(engine, "LlamaForCausalLM")
        assert _supports(engine, "unknown")  # a config with no architectures, as before
        assert not _supports(engine, "SomethingElseForCausalLM")


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("answer", ["absent", "failed"])
def test_without_a_probe_answer_the_registry_then_the_list_decide_as_before(engine, answer):
    mod = vl if engine == "vllm" else sg
    facts = None if answer == "absent" else EngineFacts(engine=engine, errors={"archs": "no answer within 120 s"})
    with mock.patch.object(mod, "_engine_facts", lambda: facts):
        assert _supports(engine, "Qwen3_5ForConditionalGeneration")
        assert not _supports(engine, GPT_OSS)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_listed_architecture_still_needs_a_card_the_engine_runs_on(engine):
    mod = vl if engine == "vllm" else sg
    with mock.patch.object(mod, "_engine_facts", lambda: EngineFacts(engine=engine, archs=[GPT_OSS])):
        assert not _supports(engine, GPT_OSS, make_hw("gtx1080"))  # compute capability 6.1
        assert not _supports(engine, GPT_OSS, make_hw("cpu"))


def test_an_importable_registry_that_lacks_it_is_overruled_by_the_engine_that_will_run():
    """rigtune's own interpreter may import another vLLM than the one the `vllm` command runs; the probe asks that
    one."""
    arch = golden_cases.prepared("llama-3.1-8b", "vllm").arch.model_copy(update={"architecture": GPT_OSS})
    with mock.patch.object(vl, "load_arch", lambda spec, token=None: arch), \
            mock.patch.object(vl, "vllm_registry_archs", lambda: {"LlamaForCausalLM"}):
        assert not vl.VllmBackend().supports(make_hw("rtx4090"), ModelSpec(hf_id="openai/gpt-oss-20b"))
        with mock.patch.object(vl, "_engine_facts", lambda: EngineFacts(engine="vllm", archs=[GPT_OSS])):
            assert vl.VllmBackend().supports(make_hw("rtx4090"), ModelSpec(hf_id="openai/gpt-oss-20b"))


def test_the_fallback_list_is_frozen():
    """No name is ever added to it: a new architecture comes from the probe. Changing the list changes this digest."""
    assert isinstance(vl.FALLBACK_ARCHS, frozenset) and len(vl.FALLBACK_ARCHS) == 37
    digest = hashlib.sha256("\n".join(sorted(vl.FALLBACK_ARCHS)).encode()).hexdigest()[:16]
    assert digest == FALLBACK_DIGEST


FALLBACK_DIGEST = "a14244da1181d3d8"


@pytest.mark.parametrize("engine,version", RELEASES)
def test_every_fallback_architecture_is_in_each_releases_registry(engine, version):
    """Read from the four release tags (tests/fixtures/facts/). The list holds names these registries do not: vLLM
    dropped MPTForCausalLM in 0.28.0 (model_executor/models/registry.py _PREVIOUSLY_SUPPORTED_MODELS, line 889 at
    v0.33.0), and no SGLang module defines Bloom, Falcon, GPT-NeoX, MPT or SmolLM3. The list still offers them, as it
    did before the probe (it never takes a name away); a launch of one fails and is reported."""
    facts = golden_cases.facts_fixture(engine, version)
    missing = sorted(a for a in vl.FALLBACK_ARCHS if a not in facts.archs)
    assert missing == KNOWN_ABSENT.get(engine, [])


KNOWN_ABSENT = {"vllm": ["MPTForCausalLM"],
                "sglang": ["BloomForCausalLM", "FalconForCausalLM", "GPTNeoXForCausalLM", "MPTForCausalLM",
                           "SmolLM3ForCausalLM"]}
