"""Session files: no level, re-measure or power point is flattered by conversations another one cached."""

from __future__ import annotations

import json
import logging

from rigtune.calibrate.measure import _level_workload
from rigtune.calibrate.search import _point_workload
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import get_workload, workload_from_file


def _sessions(tmp_path, n, name="chats.jsonl"):
    p = tmp_path / name
    lines = [json.dumps({"session": f"c{i}", "messages": [{"role": "user", "content": f"q{i}"},
                                                          {"role": "assistant", "content": "a"},
                                                          {"role": "user", "content": "more"}]})
             for i in range(n)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _ids(wl, k, c):
    level = _level_workload(wl, k, TokenCounter())
    level = level.head(level.level_requests(c))
    return {s.id for s in level.session_list}


def test_levels_and_their_re_measures_get_sessions_of_their_own(tmp_path):
    wl = workload_from_file(_sessions(tmp_path, 300), get_workload("chat")).ensure_prompts()
    levels = len(wl.concurrencies)
    sent = [_ids(wl, k, c) for k, c in enumerate(wl.concurrencies)]
    sent += [_ids(wl, k + levels, c) for k, c in enumerate(wl.concurrencies)]  # the close-call re-measures
    assert [len(s) for s in sent] == [8, 24, 48, 8, 24, 48]  # twice each turn budget's worth of 2-turn sessions
    assert all(not (a & b) for i, a in enumerate(sent) for b in sent[i + 1:])


def test_a_small_file_gets_fewer_sessions_per_level_and_says_so(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        wl = workload_from_file(_sessions(tmp_path, 30), get_workload("chat"))
    assert wl.n_prompts == 5  # 30 sessions over 3 levels and their 3 re-measures, rather than reusing any
    assert "30 sessions" in caplog.text and "below their concurrency" in caplog.text


def _replayed(points):
    return [{s.id for k in range(6) for s in _level_workload(p, k, TokenCounter()).session_list} for p in points]


def test_each_power_point_replays_sessions_of_its_own(tmp_path):
    wl = workload_from_file(_sessions(tmp_path, 1000), get_workload("chat")).ensure_prompts()
    points = [_point_workload(wl, i, 3).ensure_prompts() for i in range(3)]
    assert points[0] is wl and all(p.n_prompts == wl.n_prompts for p in points)
    seen = _replayed(points)
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])


def test_power_points_draw_on_fewer_sessions_rather_than_wrap(tmp_path, caplog):
    wl = workload_from_file(_sessions(tmp_path, 300), get_workload("chat")).ensure_prompts()  # 48 per level
    with caplog.at_level(logging.WARNING):
        points = [_point_workload(wl, i, 4).ensure_prompts() for i in range(4)]
    assert all(p.n_prompts == 300 // (4 * 6) for p in points)  # 12 each: 4 points x 6 slices fit in 300
    seen = _replayed(points)
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])
    assert "instead of 48" in caplog.text


def test_power_points_of_a_flat_workload_keep_their_prompts():
    wl = get_workload("chat")
    assert _point_workload(wl, 2, 4) is wl
