"""Session files: conversations read whole, so a later turn can carry the earlier ones."""

from __future__ import annotations

import hashlib
import json
import logging

import pytest

from rigtune.calibrate import datasets
from rigtune.calibrate.datasets import Session, Step
from rigtune.calibrate.workload import get_workload, turn_budget, workload_from_file


def _write(tmp_path, lines, name="sessions.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines) + "\n", encoding="utf-8")
    return p


def _u(text):
    return {"role": "user", "content": text}


def _a(text):
    return {"role": "assistant", "content": text}


def test_a_conversation_per_line_and_a_turn_per_line_both_parse(tmp_path):
    p = _write(tmp_path, [
        {"session": "s1", "system": "Be brief.", "messages": [_u("Hi"), _a("Hello"), _u("Why?")]},
        {"session": "s2", "turn": 1, "role": "user", "content": "First", "system": "Be kind."},
        {"session": "s3", "role": "user", "content": "Alone"},
        {"session": "s2", "turn": 2, "role": "assistant", "content": "Answer"},
        {"session": "s2", "turn": 3, "role": "user", "content": "Second"},
        {"messages": [{"role": "system", "content": "Sys"}, _u("one"), _a("two"), _u("three")]},
        '"a plain prompt"',
    ])
    got = datasets.read_sessions(p)
    assert [s.id for s in got] == ["s1", "s2", "s3", "line6", "line7"]  # the order each first appears in
    assert got[0] == Session(id="s1", system="Be brief.", steps=(Step("Hi", answer="Hello"), Step("Why?")))
    assert got[1].system == "Be kind." and got[1].users == ("First", "Second") and got[1].answers == ("Answer",)
    assert got[2].users == ("Alone",) and got[2].answers == ()
    assert got[3].system == "Sys" and got[3].users == ("one", "three") and got[3].answers == ("two",)
    assert got[4].users == ("a plain prompt",) and got[4].system == ""
    assert not any(step.explicit for s in got for step in s.steps)


def test_multi_turn_messages_without_a_session_key_stay_flat(tmp_path, caplog):
    """Such files were flattened before sessions existed: same bytes, same name, same cached profile."""
    lines = [{"messages": [{"role": "system", "content": "Be brief."}, _u("Hi?")]},
             {"messages": [_u("q1"), _a("a1"), _u("q2")]}, {"prompt": "x"}]
    p = _write(tmp_path, lines, name="flat.jsonl")
    assert datasets.read_sessions(p) == []
    with caplog.at_level(logging.INFO):
        wl = workload_from_file(p, get_workload("chat"))
    assert not wl.sessions and wl.name == f"file-flat-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}"
    assert set(wl.ensure_prompts().prompts) <= {"Be brief.\n\nHi?", "q1\n\na1\n\nq2", "x"}
    assert "1 conversations are flattened" in caplog.text
    forced = workload_from_file(p, get_workload("chat"), sessions=True)  # --sessions replays them instead
    assert forced.sessions and forced.name.endswith("-sessions")
    assert len(datasets.read_sessions(p, force=True)) == 3


def test_a_session_key_beside_a_prompt_is_ignored_as_before(tmp_path, caplog):
    """A flat line {"prompt": ..., "session": "u1"} predates session files: it must neither switch modes nor fail."""
    p = _write(tmp_path, [{"prompt": "first", "session": "u1"}, {"text": "second", "session": "u2"}, {"prompt": "x"}],
               name="flat.jsonl")
    assert datasets.read_sessions(p) == []
    with caplog.at_level(logging.INFO):
        wl = workload_from_file(p, get_workload("chat"))
    assert not wl.sessions and set(wl.ensure_prompts().prompts) <= {"first", "second", "x"}
    assert "2 lines have a \"session\" key beside a prompt" in caplog.text
    turn = _write(tmp_path, [{"session": "s", "role": "user", "content": "q"}], name="turns.jsonl")
    assert [s.id for s in datasets.read_sessions(turn)] == ["s"]  # one turn per line still makes a session file


def test_a_byte_order_mark_and_crlf_are_read(tmp_path):
    p = tmp_path / "bom.jsonl"
    lines = [json.dumps({"session": f"s{i}", "messages": [_u("q"), _a("a"), _u("more")]}) for i in range(3)]
    p.write_bytes(b"\xef\xbb\xbf" + "\r\n".join(lines).encode() + b"\r\n")
    assert [s.users for s in datasets.read_sessions(p)] == [("q", "more")] * 3
    shared = workload_from_file(p, session_system="\ufeffShared.")
    assert shared.session_system == "Shared."


def test_a_trailing_answer_is_dropped(tmp_path):
    p = _write(tmp_path, [{"session": "x", "messages": [_u("q1"), _a("a1"), _u("q2"), _a("a2")]}])
    (s,) = datasets.read_sessions(p)
    assert s.users == ("q1", "q2") and s.answers == ("a1",) and s.steps[-1].answer is None


@pytest.mark.parametrize("lines, where", [
    ([{"session": "x", "messages": [_a("first"), _u("then")]}], ":1:"),
    ([{"session": "x", "messages": [_u("q")]}, {"messages": [_u("a"), _u("b")]}], ":2:"),  # two users in a row
    ([{"session": "x", "turn": 2, "role": "user", "content": "q"},
      {"session": "x", "turn": 1, "role": "assistant", "content": "a"}], ":2:"),           # turns go backwards
    ([{"session": "x", "messages": [_u("q")]}, {"session": "x", "messages": [_u("q")]}], ":2:"),  # appears twice
    ([{"session": "x", "messages": [_u("q")]}, {"role": "user", "content": "no session"}], ":2:"),
    ([{"session": "x", "messages": [_u("q")]}, {"answer": "?"}], ":2:"),
])
def test_errors_name_the_line(tmp_path, lines, where):
    with pytest.raises(ValueError, match=where):
        datasets.read_sessions(_write(tmp_path, lines))


def test_sessions_are_cut_to_their_first_turns(tmp_path):
    p = _write(tmp_path, [{"session": f"s{i}", "messages": [_u("q1"), _a("a1"), _u("q2"), _a("a2"), _u("q3")]}
                          for i in range(3)])
    got = datasets.sample_sessions(datasets.FILE_PREFIX + str(p), 3, 0, turns_cap=2)
    assert all(s.users == ("q1", "q2") and s.answers == ("a1",) and s.steps[-1].answer is None for s in got)
    assert {s.id for s in got} == {"s0", "s1", "s2"}
    override = datasets.sample_sessions(datasets.FILE_PREFIX + str(p), 1, 0, system="Shared.")
    assert override[0].system == "Shared." and override[0].turns == 3


def _conversations(tmp_path, n=60, turns=3, answer_words=30, name="chats.jsonl"):
    lines = []
    for i in range(n):
        messages = []
        for t in range(turns):
            messages += [_u(f"session {i} question {t} " + "word " * 20), _a("answer " * answer_words)]
        lines.append({"session": f"c{i}", "system": "You are terse.", "messages": messages})
    return _write(tmp_path, lines, name)


def test_a_session_file_becomes_a_session_workload(tmp_path):
    p = _conversations(tmp_path)
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.sessions and wl.history == "generated" and wl.turns_cap == 8 and wl.session_pool == 0
    assert wl.name == f"file-chats-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}-sessions"
    wl.ensure_prompts()
    assert len(wl.session_list) == len(wl.prompts) == wl.n_prompts
    assert all(p == f"You are terse.\n\n{s.users[0]}" for p, s in zip(wl.prompts, wl.session_list))
    spec = wl.spec()
    assert spec["sessions"] is True and spec["turns_cap"] == 8 and spec["history"] == "generated"
    assert "turns per slot" in wl.describe()


def test_answers_are_budgeted_at_their_recorded_length_so_4k_contexts_survive(tmp_path):
    """Budgeting every earlier answer at the 512-token cap put an 8-turn session past 4096 and dropped 4k contexts."""
    p = _conversations(tmp_path, turns=8, answer_words=40)
    wl = workload_from_file(p, get_workload("chat"))
    # 8 user turns and 7 answers of ~40 words fit in 4096; at 7 x 512 for the answers alone they would not.
    assert wl.min_ctx <= 4096 < 7 * 512 + wl.decode_tokens + 64
    long = _conversations(tmp_path, turns=3, answer_words=1000, name="long.jsonl")  # recorded answers past the cap
    assert workload_from_file(long).prefill_tokens >= 2 * 512


def test_a_level_has_a_turn_budget_and_enough_sessions_to_fill_it(tmp_path):
    wl = workload_from_file(_conversations(tmp_path, n=400), get_workload("chat")).ensure_prompts()
    assert [wl.turn_budget(c) for c in wl.concurrencies] == [8, 24, 48]
    # Twice the budget's worth of 3-turn sessions: still admitting new ones when the budget runs out.
    assert [wl.level_requests(c) for c in wl.concurrencies] == [6, 16, 32]
    pooled = workload_from_file(_conversations(tmp_path, n=600, name="p.jsonl"), get_workload("chat"),
                                session_pool=40).ensure_prompts()
    assert all(pooled.level_requests(c) == 80 for c in pooled.concurrencies)  # twice the pool
    # A pool needs turns as well as sessions: about three for each live session, or none would get a second.
    assert [pooled.turn_budget(c) for c in pooled.concurrencies] == [120, 120, 120]
    assert turn_budget(8, 6, 64) == 192 and turn_budget(8) == 48 and turn_budget(1) == 8


def test_a_level_that_cannot_reach_its_budget_says_so(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        workload_from_file(_conversations(tmp_path, n=300), get_workload("chat"), session_pool=64)
    assert "short of its turn budget of 192" in caplog.text and "pool of 64" in caplog.text


def test_unrecorded_answers_are_budgeted_at_the_files_median(tmp_path):
    """A file of user turns with empty answers: budgeting each at the 512-token cap would need 8k contexts."""
    lines = []
    for i in range(30):
        for k in range(8):
            lines.append({"session": f"s{i}", "role": "user", "content": f"question {k} " + "w " * 10})
            if k < 7:
                lines.append({"session": f"s{i}", "role": "assistant", "content": "" if i else "answer " * 40})
    wl = workload_from_file(_write(tmp_path, lines))
    assert wl.min_ctx <= 4096


def test_explicit_steps_are_sized_by_their_own_messages(tmp_path):
    from rigtune.calibrate.workload import _session_workload

    p = _write(tmp_path, [{"session": "x", "messages": [_u("q")]}])
    big = tuple(("user", "word " * 4000) for _ in range(2))  # ~10k tokens, none of it in `users`
    s = Session(id="x", system="", steps=(Step("hi", answer="ok"), Step(messages=big)))
    wl = _session_workload(p, [s], get_workload("chat"), None, None, None, None, False)
    assert wl.prefill_tokens >= 2 * 4000 * 5 // 4


def test_session_options_shape_the_name_and_flat_files_refuse_them(tmp_path):
    p = _conversations(tmp_path)
    base = workload_from_file(p).name
    assert workload_from_file(p, turns_cap=6).name == base + "-t6"
    assert workload_from_file(p, session_pool=32).name == base + "-pool32"
    assert workload_from_file(p, history="recorded").name == base + "-recorded"
    assert workload_from_file(p, turns_cap=6, session_pool=32, history="recorded").name == base + "-t6-pool32-recorded"
    shared = workload_from_file(p, session_system="One shared system prompt.")
    assert shared.name != base and shared.name.endswith("-sessions") and shared.session_system
    with pytest.raises(ValueError, match="--history"):
        workload_from_file(p, history="imagined")
    flat = _write(tmp_path, ['{"prompt": "x"}'], name="flat.jsonl")
    with pytest.raises(ValueError, match="--session-pool"):
        workload_from_file(flat, session_pool=8)
