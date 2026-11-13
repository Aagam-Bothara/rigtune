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
    assert a1 == Session(id="a1", system="You are an agent. ", steps=(
        Step(messages=(("system", "You are an agent. "), ("user", "  hi  ")), chain=True),
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
