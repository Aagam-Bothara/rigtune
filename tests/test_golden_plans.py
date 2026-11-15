"""The memory planner's estimate and verdict for every configuration of golden_cases.plan_grid, on three cards and
four models per engine release, are the ones in tests/golden/plans/. Regenerate on purpose with
`python -m tests.golden_cases`; the rows that moved are the change. Each runs with the engine's source alone and with
the engine probe answering as the release's source says it would (tests/fixtures/facts/)."""

from __future__ import annotations

import pytest

from tests import golden_cases
from tests.engine_trees import VERSIONS


@pytest.mark.parametrize("probed", [False, True], ids=["source", "probe"])
@pytest.mark.parametrize("release", VERSIONS)
def test_plans_are_the_golden_ones(release, probed, tmp_path):
    engine, version = release.split("-")
    facts = golden_cases.facts_fixture(engine, version) if probed else None
    got = golden_cases.plans(engine, version, tmp_path, facts)
    golden = golden_cases.load(f"plans/{release}.json")
    assert sorted(got) == sorted(golden)
    moved = []
    for entry in sorted(golden):
        old = {r[0]: r for r in golden[entry]}
        new = {r[0]: r for r in got[entry]}
        moved += [f"{entry}: {key}" for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)]
    assert moved == []
