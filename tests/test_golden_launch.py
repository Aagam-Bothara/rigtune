"""Launch commands for about forty configurations per engine release are the ones in tests/golden/launch/.

Each release is seen through a source tree shaped like it (tests/engine_trees.py), so the feature checks that read
the engine's source answer as on that release. To change a launch command on purpose, regenerate with
`python -m tests.golden_cases` and say in the commit which entries moved and why.
"""

from __future__ import annotations

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
from tests import golden_cases
from tests.engine_trees import VERSIONS, installed


def _differences(got: dict, golden: dict) -> list:
    return sorted(k for k in set(got) | set(golden) if got.get(k) != golden.get(k))


@pytest.mark.parametrize("release", VERSIONS)
def test_launch_commands_are_the_golden_ones(release, tmp_path):
    engine, version = release.split("-")
    got = golden_cases.launches(engine, version, tmp_path)
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
