"""Candidates, precisions, KV dtypes, speculation and prefill variants for three cards and four models per engine
release are the ones in tests/golden/candidates/. Regenerate on purpose with `python -m tests.golden_cases`.

Each runs twice: with the engine's source alone, and with the engine probe answering as each release's source says it
would (tests/fixtures/facts/): rigtune must pick the same candidates either way."""

from __future__ import annotations

import pytest

from tests import golden_cases
from tests.engine_trees import VERSIONS


@pytest.mark.parametrize("probed", [False, True], ids=["source", "probe"])
@pytest.mark.parametrize("release", VERSIONS)
def test_candidates_are_the_golden_ones(release, probed, tmp_path):
    engine, version = release.split("-")
    facts = golden_cases.facts_fixture(engine, version) if probed else None
    got = golden_cases.candidates(engine, version, tmp_path, facts)
    golden = golden_cases.load(f"candidates/{release}.json")
    assert sorted(got) == sorted(golden) and len(golden) == len(golden_cases.CARDS) * len(golden_cases.MODELS)
    assert sorted(k for k in golden if got[k] != golden[k]) == []
