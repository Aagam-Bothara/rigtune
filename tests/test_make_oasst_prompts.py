"""Held-out prompts from OpenAssistant: the opening user turn of English conversations, deduplicated."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_oasst_prompts", Path(__file__).resolve().parents[1] / "benchmarks" / "make_oasst_prompts.py")
mop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mop)

ROWS = [
    {"text": "How do tides work?", "role": "prompter", "parent_id": None, "lang": "en"},
    {"text": "The Moon pulls the oceans.", "role": "assistant", "parent_id": "m1", "lang": "en"},  # a reply
    {"text": "And why twice a day?", "role": "prompter", "parent_id": "m2", "lang": "en"},  # a later turn
    {"text": "¿Qué es la marea?", "role": "prompter", "parent_id": None, "lang": "es"},  # another language
    {"text": " How do tides work? ", "role": "prompter", "parent_id": None, "lang": "en"},  # a duplicate opener
    {"text": "Write a haiku about rain.", "role": "prompter", "parent_id": None, "lang": "en"},
]


def test_only_distinct_english_openers_are_kept():
    assert sorted(mop.first_turns(ROWS, 10)) == ["How do tides work?", "Write a haiku about rain."]


def test_the_selection_is_seeded_and_capped():
    assert mop.first_turns(ROWS, 1) == mop.first_turns(ROWS, 1) and len(mop.first_turns(ROWS, 1)) == 1


THREAD = [
    {"message_id": "m1", "text": "How do tides work?", "role": "prompter", "parent_id": None, "lang": "en"},
    {"message_id": "m2", "text": "The Moon pulls the oceans.", "role": "assistant", "parent_id": "m1", "lang": "en",
     "rank": 0},
    {"message_id": "m2b", "text": "Magic.", "role": "assistant", "parent_id": "m1", "lang": "en", "rank": 1},
    {"message_id": "m3", "text": "And why twice a day?", "role": "prompter", "parent_id": "m2", "lang": "en"},
    {"message_id": "m4", "text": "Two bulges.", "role": "assistant", "parent_id": "m3", "lang": "en", "rank": 0},
    {"message_id": "h1", "text": "Write a haiku about rain.", "role": "prompter", "parent_id": None, "lang": "en"},
    {"message_id": "h2", "text": "Rain falls...", "role": "assistant", "parent_id": "h1", "lang": "en"},
]


THIRD = [
    {"message_id": "m5", "text": "Does the Sun pull too?", "role": "prompter", "parent_id": "m4", "lang": "en"},
    {"message_id": "m6", "text": "Yes, less.", "role": "assistant", "parent_id": "m5", "lang": "en", "rank": 0},
    {"message_id": "m7", "text": "How much less?", "role": "prompter", "parent_id": "m6", "lang": "en"},
]


def test_sessions_follow_the_best_ranked_reply_and_need_three_user_turns(tmp_path):
    from rigtune.calibrate.datasets import read_sessions

    assert mop.sessions(THREAD, 10) == []  # two user turns: too few to reuse two turns of history
    got = mop.sessions(THREAD + THIRD, 10)
    assert [line["session"] for line in got] == ["m1"]
    assert [m["content"] for m in got[0]["messages"]] == [
        "How do tides work?", "The Moon pulls the oceans.", "And why twice a day?", "Two bulges.",
        "Does the Sun pull too?", "Yes, less.", "How much less?"]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in got) + "\n", encoding="utf-8")
    (s,) = read_sessions(p)  # what it writes is a session file
    assert s.users[:2] == ("How do tides work?", "And why twice a day?") and s.turns == 4


def test_the_flattened_control_holds_the_same_conversations_and_system_prompt(tmp_path):
    from rigtune.calibrate.workload import get_workload, workload_from_file

    lines = mop.sessions(THREAD + THIRD, 10, max_turns=3, system="Be brief.")
    assert lines[0]["system"] == "Be brief." and lines[0]["messages"][-1]["content"] == "Does the Sun pull too?"
    flat = mop.flattened(lines)
    assert flat == [{"messages": [{"role": "system", "content": "Be brief."}] + lines[0]["messages"]}]
    p, q = tmp_path / "s.jsonl", tmp_path / "s-flat.jsonl"
    p.write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
    q.write_text(json.dumps(flat[0]) + "\n", encoding="utf-8")
    sessions, control = workload_from_file(p, get_workload("chat")), workload_from_file(q, get_workload("chat"))
    assert sessions.sessions and not control.sessions  # the control is flattened, one prompt per conversation
    assert control.ensure_prompts().prompts[0].startswith("Be brief.\n\nHow do tides work?")
