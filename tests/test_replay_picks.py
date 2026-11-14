"""Every recorded profile under benchmarks/strategies replays through today's search to the pick, trials and
class in tests/golden/replay.json (benchmarks/replay_search.py). Only five replay exactly to their recorded pick;
this pins all of them, so a change to the search shows which recorded calibration it would have moved."""

from __future__ import annotations

from tests import golden_cases


def test_every_recorded_profile_replays_as_pinned(no_network):
    got = golden_cases.replays()
    golden = golden_cases.load("replay.json")
    assert sorted(got) == sorted(golden)
    assert sorted(k for k in golden if got[k] != golden[k]) == []
    verdicts = [v.get("verdict") for v in golden.values()]
    assert (verdicts.count("exact"), verdicts.count("inconclusive"), verdicts.count("different")) == (5, 13, 3)
