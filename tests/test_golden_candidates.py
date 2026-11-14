"""Candidates, precisions, KV dtypes, speculation and prefill variants for three cards and four models per engine
release are the ones in tests/golden/candidates/. Regenerate on purpose with `python -m tests.golden_cases`."""

from __future__ import annotations

import pytest

from tests import golden_cases
from tests.engine_trees import VERSIONS


@pytest.mark.parametrize("release", VERSIONS)
def test_candidates_are_the_golden_ones(release, tmp_path):
    engine, version = release.split("-")
    got = golden_cases.candidates(engine, version, tmp_path)
    golden = golden_cases.load(f"candidates/{release}.json")
    assert sorted(got) == sorted(golden) and len(golden) == len(golden_cases.CARDS) * len(golden_cases.MODELS)
    assert sorted(k for k in golden if got[k] != golden[k]) == []
