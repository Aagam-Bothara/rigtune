"""Session files of explicit requests: an agent's every request given whole, read without touching other files."""

from __future__ import annotations

import hashlib
import json

import pytest

from rigtune.calibrate import datasets
from rigtune.calibrate.datasets import Session, Step
from rigtune.calibrate.workload import get_workload, workload_from_file


def _write(tmp_path, lines, name="agent.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines) + "\n", encoding="utf-8")
    return p


def _m(role, content):
    return {"role": role, "content": content}


SYS = _m("system", "You are an agent. ")


def test_both_forms_parse_and_keep_messages_byte_for_byte(tmp_path):
    p = _write(tmp_path, [
        {"session": "a1", "gen": {"ignored": True}, "requests": [
            {"messages": [SYS, _m("user", "  hi  ")]},
            {"messages": [SYS, _m("user", "  hi  "), _m("assistant", "call search"), _m("user", "result\n")],
             "chain": True, "tag": "heartbeat", "max_tokens": 32},
            {"messages": [SYS, _m("user", "rewritten")], "tag": "evict"}]},
        {"session": "a2", "request": 1, "messages": [_m("user", "first")]},
        {"session": "a2", "request": 5, "messages": [_m("user", "first"), _m("assistant", ""), _m("user", "second")],
         "chain": True},
        {"requests": [{"messages": [_m("user", "no session key")]}]},
    ])
    a1, a2, own = datasets.read_sessions(p)
    assert (a1.id, a2.id, own.id) == ("a1", "a2", "line4")
    bare = Session(id=a1.id, system=a1.system, steps=tuple(
        Step(messages=s.messages, chain=s.chain, tag=s.tag, max_tokens=s.max_tokens) for s in a1.steps))
    assert bare == Session(id="a1", system="You are an agent. ", steps=(
        Step(messages=(("system", "You are an agent. "), ("user", "  hi  ")), chain=True, tag="first"),
        Step(messages=(("system", "You are an agent. "), ("user", "  hi  "), ("assistant", "call search"),
                       ("user", "result\n")), tag="heartbeat", max_tokens=32),
        Step(messages=(("system", "You are an agent. "), ("user", "rewritten")), tag="evict")))
    assert a2.system == "" and [s.chain for s in a2.steps] == [True, False] and a2.steps[1].messages[1] == ("assistant", "")
    assert all(step.explicit for s in (a1, a2, own) for step in s.steps)
    assert a1.first_turn == "You are an agent. \n\n  hi  "  # the system prompt once, as the request holds it


@pytest.mark.parametrize("lines, where", [
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")]}]}, {"session": "b", "messages": [_m("user", "q")]}],
     ":2: line 1 gives requests whole"),                                                    # mixed with turns
    ([{"session": "a", "requests": [{"messages": []}]}], ":1: request 1: session 'a': a request needs"),
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")]}, {"messages": [_m("tool", "r")]}]}],
     ":1: request 2: each message needs"),                                                  # a tool role
    ([{"session": "a", "requests": [{"messages": [{"role": "user", "content": [{"type": "text"}]}]}]}],
     ":1: request 1: each message needs"),                                                  # content parts
    ([{"session": "a", "requests": [{"messages": [{"role": "user", "content": "q", "name": "x"}]}]}],
     ":1: request 1: each message needs"),                                                  # a key not sent
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")], "chain": True}]}], ":1: request 1: session 'a' opens"),
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")], "max_tokens": 0}]}], ":1: request 1: session 'a': \"max"),
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")], "chain": "yes"}]}], ":1: request 1: session 'a': \"chain"),
    ([{"session": "a", "request": 2, "messages": [_m("user", "q")]},
      {"session": "a", "request": 1, "messages": [_m("user", "q")]}], ":2: session 'a' request 1 does not follow"),
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")]}]},
      {"session": "a", "request": 1, "messages": [_m("user", "q")]}], ":2: session 'a' was already given whole"),
    ([{"session": "a", "requests": [{"messages": [_m("user", "q")]}]},
      {"session": "a", "requests": [{"messages": [_m("user", "q")]}]}], ":2: session 'a' appears twice"),
    ([{"session": "a", "system": "S", "requests": [{"messages": [_m("user", "q")]}]}], ":1: a \"system\" key"),
    ([{"session": "a", "messages": [_m("user", "q")], "requests": [{"messages": [_m("user", "q")]}]}],
     ":1: give \"requests\" or \"messages\""),
    ([{"session": "a", "request": 1, "messages": [_m("user", "q")], "requests": [{"messages": [_m("user", "q")]}]}],
     ":1: give \"request\" with its \"messages\", or \"requests\""),
])
def test_errors_name_the_line(tmp_path, lines, where):
    with pytest.raises(ValueError, match=where.replace("'", ".").replace("(", r"\(")):
        datasets.read_sessions(_write(tmp_path, lines))


def test_only_a_requests_or_request_key_makes_a_file_explicit(tmp_path):
    """Turn-based session files and flat files are read as before; see test_flat_requests_unchanged for their bytes."""
    turns = _write(tmp_path, [{"session": "s", "messages": [_m("user", "q"), _m("assistant", "a"), _m("user", "r")]},
                              {"session": "t", "turn": 1, "role": "user", "content": "q"}], name="turns.jsonl")
    assert not any(step.explicit for s in datasets.read_sessions(turns) for step in s.steps)
    flat = _write(tmp_path, [{"prompt": "x", "requests": [{"messages": [_m("user", "q")]}]},
                             {"prompt": "y", "session": "u", "request": 1}], name="flat.jsonl")
    assert datasets.read_sessions(flat) == []
    # A "request" or "requests" key of another shape is metadata on a conversation, as it always was.
    convo = [_m("user", "q"), _m("assistant", "a"), _m("user", "r")]
    meta = _write(tmp_path, [{"session": "s", "request": "req-7", "messages": convo},
                             {"session": "t", "request": True, "messages": convo},
                             {"session": "u", "requests": [], "messages": convo},
                             {"session": "v", "requests": ["r1"], "messages": convo},
                             {"session": "w", "requests": [{"no": "messages"}], "messages": convo}], name="meta.jsonl")
    got = datasets.read_sessions(meta)
    assert [s.users for s in got] == [("q", "r")] * 5 and not any(st.explicit for s in got for st in s.steps)
    assert datasets.read_sessions(_write(tmp_path, [{"session": "x", "requests": []}], name="empty.jsonl")) == []


def _agent_file(tmp_path, n=40, name="agent.jsonl"):
    lines = []
    for i in range(n):
        reqs, convo = [], [SYS]
        for t in range(3):
            convo = convo + [_m("user", f"agent {i} step {t} " + "word " * 20)]
            reqs.append({"messages": convo, **({"chain": True} if t == 2 else {})})
            convo = convo + [_m("assistant", "thinking " * 10)]
        lines.append({"session": f"a{i}", "requests": reqs})
    return _write(tmp_path, lines, name)


def test_an_explicit_file_becomes_a_session_workload(tmp_path):
    p = _agent_file(tmp_path)
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.sessions and wl.history == "explicit"
    assert wl.name == f"file-agent-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}-sessions-explicit"
    assert wl.spec()["history"] == "explicit" and "requests given whole" in wl.describe()
    wl.ensure_prompts()
    assert all(not p.startswith("You are an agent. \n\nYou are an agent.") for p in wl.prompts)
    # Sized by the largest request: the system prompt, three user turns and two answers.
    words = len(SYS["content"]) + 3 * len("agent 0 step 0 " + "word " * 20) + 2 * len("thinking " * 10)
    assert wl.prefill_tokens >= words // 4


def test_history_and_a_replacement_system_prompt_are_refused(tmp_path):
    p = _agent_file(tmp_path)
    for kwargs, flag in (({"history": "recorded"}, "--history"), ({"history": "generated"}, "--history"),
                         ({"session_system": "Other."}, "--session-system")):
        with pytest.raises(ValueError, match=f"gives each request's messages whole, so {flag}"):
            workload_from_file(p, get_workload("chat"), **kwargs)
    assert workload_from_file(p, session_pool=16).name.endswith("-sessions-explicit-pool16")  # the pool still applies


# --------------------------------------------------------------------------- what each request changed

S, CORE, CORE2 = ("system", "instructions"), ("system", "core: likes tea"), ("system", "core: likes coffee")
U1, A1, U2, A2, U3 = ("user", "hello there"), ("assistant", "hi"), ("user", "what now"), ("assistant", "ok"), \
    ("user", "and then")
SUMMARY = ("user", "summary: they talked")


def _len(*messages):
    return len(datasets.rendered(messages))


def test_the_rendering_is_the_one_replay_compares():
    from rigtune.calibrate.measure import _render

    msgs = (S, U1, A1)
    assert datasets.rendered(msgs) == _render([{"role": r, "content": c} for r, c in msgs])


def test_an_append_shares_everything_it_held_and_the_rest_is_new():
    prev, cur = (S, CORE, U1), (S, CORE, U1, A1, U2)
    lcp, new = datasets.prefix_shares(prev, cur)
    assert datasets.request_kind(prev, cur) == "append"
    assert lcp == _len(*prev) / _len(*cur) and lcp + new == pytest.approx(1.0)


def test_a_repeated_message_appended_is_new_and_a_displaced_one_counts_once():
    ok, again = ("user", "[result] ok"), ("assistant", "again")
    prev, cur = (S, ok, A1, ok), (S, ok, A1, ok, again, ok)
    assert datasets.prefix_shares(prev, cur)[0] + datasets.prefix_shares(prev, cur)[1] == pytest.approx(1.0)
    # Dropping U1 displaces the rest: the kept A1 and "ok" were sent before; the second "ok", appended, was not.
    prev, cur = (S, U1, A1, ok), (S, A1, ok, again, ok)
    lcp, new = datasets.prefix_shares(prev, cur)
    assert lcp == (_len(S) + 1) / _len(*cur)  # "\x00", then user against assistant
    assert new == _len(again, ok) / _len(*cur) and 1 - lcp - new == pytest.approx((_len(A1, ok) - 1) / _len(*cur))


# Hand-counted cases. A rendered message is 2 + len(role) + len(content) characters: "\x00", role, "\x01", content.
HS = ("system", "S" * 100)                                                     # 108
HU1, HA1, HU2, HA2, HU3 = ("user", "u" * 20), ("assistant", "a" * 20), ("user", "v" * 20), ("assistant", "b" * 20),     ("user", "w" * 20)                                                         # 26, 31, 26, 31, 26
OK = ("user", "[function result] ok")                                          # 26


def _chars(prev, cur):
    lcp, new = datasets.prefix_shares(prev, cur)
    n = len(datasets.rendered(cur))
    return round(lcp * n), round(new * n), round((1 - lcp - new) * n)


def test_an_edit_inside_a_message_leaves_what_follows_it_behind_the_change():
    """The system message changed at its 51st character: its last 49 characters, unchanged, were sent before."""
    edited = ("system", "S" * 50 + "X" + "S" * 49)
    # lcp: 8 + 50. new: the "X", a2 and u3 (1 + 31 + 26). Behind: the rest of the message, u1, a1, u2 (49 + 83).
    assert _chars((HS, HU1, HA1, HU2), (edited, HU1, HA1, HU2, HA2, HU3)) == (58, 58, 132)


def test_a_displaced_message_that_does_not_come_back_is_not_behind_the_change():
    """An eviction keeps a2; the "ok" after it is new, although an evicted message said the same."""
    summary = ("user", "summary" * 3)                                          # 27
    # lcp: 108 + "\x00user\x01" (6). Behind: a2 (31). New: the rest of the summary and "ok" (21 + 26).
    assert _chars((HS, OK, HA1, HU2, HA2), (HS, summary, HA2, OK)) == (114, 47, 31)


def test_a_request_that_drops_only_the_end_is_a_truncation():
    assert datasets.request_kind((HS, HU1, HA1, HU2), (HS, HU1)) == "truncate"
    assert _chars((HS, HU1, HA1, HU2), (HS, HU1)) == (134, 0, 0)
    assert datasets.request_kind((HS, HU1), (HS, HU1)) == "append"  # the same again appends nothing


def test_a_first_request_is_all_new():
    assert datasets.prefix_shares((), (S, U1)) == (0.0, 1.0) and datasets.request_kind(None, (S, U1)) == "first"


def test_a_memory_edit_in_the_middle_puts_everything_after_it_behind_the_change():
    prev, cur = (S, CORE, U1, A1, U2), (S, CORE2, U1, A1, U2, A2, U3)
    lcp, new = datasets.prefix_shares(prev, cur)
    assert datasets.request_kind(prev, cur) == "rewrite_mid"
    common = _len(S) + len("\x00system\x01core: likes ")
    assert lcp == common / _len(*cur)
    assert new == (_len(CORE2) - len("\x00system\x01core: likes ") + _len(A2, U3)) / _len(*cur)
    assert 1 - lcp - new == pytest.approx(_len(U1, A1, U2) / _len(*cur))  # sent before, prefilled again


def test_an_edit_to_the_first_message_is_a_head_rewrite():
    prev, cur = (("system", "instructions; core: tea"), U1), (("system", "instructions; core: coffee"), U1, A1)
    assert datasets.request_kind(prev, cur) == "rewrite_head"
    lcp, new = datasets.prefix_shares(prev, cur)
    assert lcp == len("\x00system\x01instructions; core: ") / _len(*cur)
    assert 1 - lcp - new == pytest.approx(_len(U1) / _len(*cur))


def test_an_eviction_keeps_only_the_system_and_core_share():
    """Old messages replaced by a summary right after the memory: the kept tail was sent before, behind the change."""
    prev, cur = (S, CORE, U1, A1, U2, A2), (S, CORE, SUMMARY, U2, A2, U3)
    lcp, new = datasets.prefix_shares(prev, cur)
    assert datasets.request_kind(prev, cur) == "rewrite_mid"
    assert lcp == (_len(S, CORE) + len("\x00user\x01")) / _len(*cur)  # the summary's role matches the dropped turn's
    assert new == (_len(SUMMARY) - len("\x00user\x01") + _len(U3)) / _len(*cur)
    assert 1 - lcp - new == pytest.approx(_len(U2, A2) / _len(*cur))


def test_tags_default_to_the_kind_and_each_step_carries_its_shares(tmp_path):
    def req(*msgs, **kw):
        return {"messages": [{"role": r, "content": c} for r, c in msgs], **kw}

    p = _write(tmp_path, [{"session": "a", "requests": [
        req(S, CORE, U1), req(S, CORE, U1, A1, U2, chain=True), req(S, CORE2, U1, A1, U2, A2, U3, tag="core_edit"),
        req(("system", "other"), U3)]}])
    (s,) = datasets.read_sessions(p)
    assert [st.kind for st in s.steps] == ["first", "append", "rewrite_mid", "rewrite_head"]
    assert [st.tag for st in s.steps] == ["first", "append", "core_edit", "rewrite_head"]
    lcp, new = datasets.prefix_shares((S, CORE, U1, A1, U2), (S, CORE2, U1, A1, U2, A2, U3))
    assert (s.steps[2].lcp, s.steps[2].new) == (lcp, new) and s.steps[0].lcp == 0.0
    turns = _write(tmp_path, [{"session": "t", "messages": [{"role": "user", "content": "q"}]}], name="t.jsonl")
    (t,) = datasets.read_sessions(turns)
    assert (t.steps[0].kind, t.steps[0].lcp, t.steps[0].new, t.steps[0].tag) == ("", None, None, "")


def test_a_level_records_its_refill_behind_edits():
    from rigtune.calibrate.measure import RequestOutcome, _metrics_from

    def step(turn, prompt, lcp, new):
        return RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.2, tokens=4, token_source="usage", prompt_tokens=prompt,
                              session="a", turn=turn, request_chars=100, lcp=lcp, new=new)

    m = _metrics_from([step(1, 100, 0.0, 1.0), step(2, 200, 0.9, 0.1), step(3, 300, 0.2, 0.3)], 1.0, 1)
    assert m.refill_behind_edit == round(300 * 0.5 / 600, 4)  # the third sent half its prompt again
    turn_based = _metrics_from([step(1, 100, None, None)], 1.0, 1)
    assert turn_based.refill_behind_edit is None
