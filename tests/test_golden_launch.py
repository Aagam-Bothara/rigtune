"""Launch commands for about forty configurations per engine release are the ones in tests/golden/launch/.

Each release is seen through a source tree shaped like it (tests/engine_trees.py), so the feature checks that read
the engine's source answer as on that release. To change a launch command on purpose, regenerate with
`python -m tests.golden_cases` and say in the commit which entries moved and why.

Each release is also seen with the engine probe answering as its source says it would (tests/fixtures/facts/,
hand-built from the release tags until a GPU session records the probe's real output): the launch commands must not
move, and every feature check must give the release's answer from the probe alone, with no source to read.
"""

from __future__ import annotations

from unittest import mock

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from tests import golden_cases
from tests.engine_trees import VERSIONS, installed


def _differences(got: dict, golden: dict) -> list:
    return sorted(k for k in set(got) | set(golden) if got.get(k) != golden.get(k))


@pytest.mark.parametrize("probed", [False, True], ids=["source", "probe"])
@pytest.mark.parametrize("release", VERSIONS)
def test_launch_commands_are_the_golden_ones(release, probed, tmp_path):
    engine, version = release.split("-")
    facts = golden_cases.facts_fixture(engine, version) if probed else None
    got = golden_cases.launches(engine, version, tmp_path, facts)
    golden = golden_cases.load(f"launch/{release}.json")
    assert _differences(got, golden) == []
    assert len(golden) >= 35


def _answers(engine: str) -> dict:
    if engine == "vllm":
        return {"attention_backend": vl._attention_backend_flag(),
                "prompt_tokens_details": vl._prompt_tokens_details_flag(),
                "language_model_only": vl._language_model_only_flag(),
                "offload": sorted(vl._offload_features())}
    return {"spec_methods": sorted(sg.sglang_spec_methods()), "cache_report": sg._cache_report_flag(),
            "hicache": sorted(sg._hicache_features()), "tier_details": sg._tier_details_flag()}


# What each feature check returns on the real source at each release tag (read from the four tags' files); the trees
# must give the same answers.
RELEASE_ANSWERS = {
    "vllm": {"attention_backend": True, "prompt_tokens_details": True, "language_model_only": True,
             "offload": ["cpu", "hma", "shm"]},
    "sglang": {"spec_methods": ["draft", "eagle3", "ngram"], "cache_report": True,
               "hicache": ["file", "file_cap", "hicache"], "tier_details": True},
}


def _probed_answers(engine: str) -> dict:
    """The same checks as _answers, through what the backends consult: the probe's answer, else the source's."""
    mod = vl if engine == "vllm" else sg
    return {k: sorted(v) if isinstance(v, frozenset) else v
            for k, v in ((name, mod.feature(name)) for name in sorted(mod.FEATURE_CHECKS))}


@pytest.mark.parametrize("release", VERSIONS)
def test_each_releases_facts_answer_every_feature_check_as_the_release_does(release, tmp_path):
    """With an empty package where the engine's source would be, every answer must come from the probe's."""
    engine, version = release.split("-")
    facts = golden_cases.facts_fixture(engine, version)
    assert facts.provenance.startswith("source:") and facts.version == version
    mod = vl if engine == "vllm" else sg
    with installed(engine, version, tmp_path) as pkg:
        for path in sorted(pkg.rglob("*.py"), reverse=True):
            path.unlink()
        with mock.patch.object(mod, "_engine_facts", lambda: facts):
            assert _probed_answers(engine) == RELEASE_ANSWERS[engine]
        with mock.patch.object(mod, "_engine_facts", lambda: None):
            answers = _probed_answers(engine)
        nothing = {"vllm": {"attention_backend": False, "prompt_tokens_details": False, "language_model_only": False,
                            "offload": []},
                   "sglang": {"spec_methods": [], "cache_report": False, "hicache": [], "tier_details": False}}
        assert answers == nothing[engine]  # the check above was the probe's doing


@pytest.mark.parametrize("release", VERSIONS)
def test_each_releases_tree_answers_every_feature_check_as_the_release_does(release, tmp_path):
    engine, version = release.split("-")
    with installed(engine, version, tmp_path):
        assert _answers(engine) == RELEASE_ANSWERS[engine]


def test_sglang_0523_answers_through_its_field_order_not_its_argument_definitions(tmp_path):
    """0.5.23 moved the argument definitions to srt/arg_groups/fields/, which rigtune does not read; its answers
    come from arg_groups/field_order.py and the hooks. Without those files nothing would be found."""
    with installed("sglang", "0.5.23", tmp_path) as pkg:
        (pkg / "srt" / "arg_groups" / "field_order.py").unlink()
        sg._cache_report_flag.cache_clear()
        sg._hicache_features.cache_clear()
        assert not sg._cache_report_flag() and sg._hicache_features() == frozenset()
