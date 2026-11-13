"""Session replay over HTTP against a fake chat engine whose replies are gated, so order is exact, not timed."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.datasets import Session, Step
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload, get_workload, workload_from_file
from tests.test_measure_and_proxy import _Server
from tests.test_session_scheduler import simulate


def _words(messages) -> int:
    return sum(len(m["content"].split()) for m in messages)


def _key(body):
    users = [m for m in body["messages"] if m["role"] == "user"]
    return users[0]["content"].split()[0], len(users)


def gated_engine(c: int, total: int, expected=None, fail=()) -> FastAPI:
    """Holds every request until c are waiting (or all that remain), then answers one, the earliest in `expected`
    when given, else the earliest to arrive: the level's closed loop, made exact."""
    app = FastAPI()
    app.state.order, app.state.bodies, app.state.peak = [], [], 0
    st = {"waiting": [], "streaming": False, "done": 0, "cond": None}
    rank = {k: i for i, k in enumerate(expected or [])}

    def releasable(key):
        if st["streaming"] or len(st["waiting"]) < min(c, total - st["done"]):
            return False
        return min(st["waiting"], key=lambda k: rank.get(k, len(rank))) == key

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if st["cond"] is None:
            st["cond"] = asyncio.Condition()
        cond = st["cond"]
        body = await req.json()
        key = _key(body)
        app.state.order.append(key)
        app.state.bodies.append(body)
        async with cond:
            st["waiting"].append(key)
            app.state.peak = max(app.state.peak, len(st["waiting"]))
            cond.notify_all()
            try:
                await asyncio.wait_for(cond.wait_for(lambda: releasable(key)), 10)
            except asyncio.TimeoutError:
                pass  # never hang a test: a wrong order shows up in the assertions instead
            st["waiting"].remove(key)
            st["streaming"] = True
        answer = f"reply {key[0]} {key[1]}"

        async def gen():
            try:
                if key in fail:
                    return
                yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"
                for word in answer.split():
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': word + ' '}}]})}\n\n"
                usage = {"prompt_tokens": _words(body["messages"]), "completion_tokens": 3}
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                async with cond:
                    st["streaming"] = False
                    st["done"] += 1
                    cond.notify_all()

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sessions(n, turns=3):
    return [Session(id=f"s{i}", system="You are terse.",
                    steps=tuple(Step(user=f"s{i} asks question {t}", answer=f"recorded answer {i}.{t}")
                                for t in range(turns)))
            for i in range(n)]


def _level(n=20, turns=3, c=2, budget_per_slot=6, pool=0, history="generated"):
    sessions = _sessions(n, turns)
    return Workload(sessions=True, source="file:unused", session_list=sessions, prompts=[s.first_turn for s in sessions],
                    n_prompts=n, concurrencies=(c,), prefill_tokens=4096, decode_tokens=64, natural_stop=True,
                    turns_per_slot=budget_per_slot, session_pool=pool, history=history, fitted=True)


def _drive(app, wl, c):
    with _Server(app) as s:
        return asyncio.run(M._drive(s.url, LlmtraceHooks(model_name="m"), wl, c, 30.0, TokenCounter()))


def _expected(wl, c):
    sched = M._SessionScheduler(wl.session_list, c, wl.session_pool, wl.turn_budget(c), lambda s, m: True,
                                wl.history)
    return simulate(sched, c)[0]


def _same_order(got, want, c):
    """Exact, except that the first c requests of a level race each other to the server."""
    return sorted(got[:c]) == sorted(want[:c]) and got[c:] == want[c:]


def test_the_level_holds_its_concurrency_in_the_scheduler_order():
    c = 4
    wl = _level(n=30, c=c)
    want = _expected(wl, c)
    app = gated_engine(c, len(want), expected=want)
    out = _drive(app, wl, c)
    assert app.state.peak == c and len(want) == wl.turn_budget(c) == 24
    assert _same_order(app.state.order, want, c)
    sent = [o for o in out if o.skipped is None]
    assert len(sent) == 24 and all(o.ok for o in sent)
    assert [o.seq for o in sorted(sent, key=lambda o: o.seq)] == list(range(24))


def test_a_pool_spaces_a_sessions_steps_over_http():
    c, pool = 2, 6
    wl = _level(n=30, turns=4, c=c, budget_per_slot=12, pool=pool)
    want = _expected(wl, c)
    app = gated_engine(c, len(want), expected=want)
    out = _drive(app, wl, c)
    assert _same_order(app.state.order, want, c) and app.state.peak == c
    assert {o.distance for o in out if o.skipped is None and o.distance is not None} == {pool - 1}


def test_each_step_resends_the_last_one_plus_its_answer_and_the_next_question():
    wl = _level(n=4, c=2, budget_per_slot=50)
    want = _expected(wl, 2)
    app = gated_engine(2, len(want), expected=want)
    _drive(app, wl, 2)
    by_session = defaultdict(list)
    for b in app.state.bodies:
        by_session[_key(b)[0]].append(b["messages"])
    for sid, turns in by_session.items():
        assert [len(t) for t in turns] == [2, 4, 6]
        for n, (before, after) in enumerate(zip(turns, turns[1:]), 1):
            assert after[:len(before)] == before  # the engine sees the exact prefix it rendered last step
            assert after[len(before)] == {"role": "assistant", "content": f"reply {sid} {n} "}
    assert all(b["stream"] and b["max_tokens"] == 64 and not b["ignore_eos"] for b in app.state.bodies)


def test_recorded_history_carries_the_files_answers_over_http():
    wl = _level(n=2, c=1, budget_per_slot=50, history="recorded")
    want = _expected(wl, 1)
    app = gated_engine(1, len(want), expected=want)
    _drive(app, wl, 1)
    second = [b["messages"] for b in app.state.bodies if len(b["messages"]) == 4]
    assert second and all(m[2]["content"].startswith("recorded answer") for m in second)


def test_a_failed_step_ends_its_session_over_http():
    wl = _level(n=2, c=1, budget_per_slot=50)
    app = gated_engine(1, 4, fail={("s0", 2)})
    out = _drive(app, wl, 1)
    keys = [_key(b) for b in app.state.bodies]
    assert ("s0", 3) not in keys and [t for s, t in keys if s == "s1"] == [1, 2, 3]
    assert [o.ok for o in out if o.session == "s0" and o.skipped is None] == [True, False]
    s1_last = [b["messages"] for b in app.state.bodies if _key(b) == ("s1", 3)][0]
    assert [m["content"] for m in s1_last if m["role"] == "assistant"] == ["reply s1 1 ", "reply s1 2 "]


def test_a_step_can_cap_its_own_answer_and_a_hook_can_add_fields():
    capped = Session(id="s0", system="", steps=(Step("s0 short please", max_tokens=5), Step("s0 then long")))
    wl = replace(_level(n=1, c=1, budget_per_slot=50), session_list=[capped], prompts=[capped.first_turn], n_prompts=1)
    app = gated_engine(1, 2)

    def hook(body, state, messages):
        body["return_cached_tokens_details"] = True
        body["step"] = state.step

    with _Server(app) as s:
        asyncio.run(M._drive_sessions(s.url, LlmtraceHooks(model_name="m"), wl, 1, 30.0, TokenCounter(),
                                      body_hook=hook))
    assert [b["max_tokens"] for b in app.state.bodies] == [5, 64]
    assert [(b["return_cached_tokens_details"], b["step"]) for b in app.state.bodies] == [(True, 0), (True, 1)]


def test_load_generator_processes_share_the_levels_budget_floor_included():
    wl = _level(n=6, c=1, budget_per_slot=1)  # a level of one slot: the floor of 8 turns, in all
    shards = [M._shard(wl, wl.session_list[i::3], 3, concurrency=1) for i in range(3)]
    assert wl.turn_budget(1) == 8 and [s.turn_budget(1) for s in shards] == [3, 3, 3]  # not 8 each


def test_usage_prompt_tokens_are_kept_and_answers_are_not():
    wl = _level(n=2, c=1, budget_per_slot=50)
    want = _expected(wl, 1)
    out = _drive(gated_engine(1, len(want), expected=want), wl, 1)
    sent = [o for o in out if o.skipped is None]
    assert all(o.prompt_tokens and o.prompt_tokens > 0 for o in sent) and all(o.text == "" for o in out)


def test_cut_and_truncated_sessions_are_not_failed_requests():
    outcomes = [M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.2, tokens=4, session="a", turn=1),
                M.RequestOutcome(ok=False, skipped="cut", session="a", turn=2),
                M.RequestOutcome(ok=False, skipped="truncated", session="b", turn=3)]
    m = M._metrics_from(outcomes, 1.0, 1)
    assert m.requests == 1 and m.failed == 0


def _file(tmp_path, n=40, turns=3, name="chats.jsonl"):
    p = tmp_path / name
    lines = []
    for i in range(n):
        messages = []
        for t in range(turns):
            messages += [{"role": "user", "content": f"s{i} asks question {t}"},
                         {"role": "assistant", "content": f"recorded answer {i}.{t}"}]
        lines.append(json.dumps({"session": f"s{i}", "system": "You are terse.", "messages": messages}))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_the_warmup_sends_the_shared_system_prompt_and_no_session(tmp_path, monkeypatch):
    seen = []

    async def fake(base_url, hooks, workload, concurrency, timeout, counter=None):
        seen.append(workload)
        return [M.RequestOutcome(ok=True, ttft_s=0.01, duration_s=0.02, tokens=4, token_source="usage")]

    monkeypatch.setattr(M, "_drive", fake)
    wl = workload_from_file(_file(tmp_path), get_workload("chat"))
    M.run_trial("http://x", LlmtraceHooks(), wl, warmup=True, counter=TokenCounter(), enough=lambda m: True)
    warm = seen[0]
    ids = {s.id for s in wl.session_list}
    assert warm.sessions and len(warm.session_list) == 8 and all(s.turns == 1 for s in warm.session_list)
    assert warm.turn_budget(8) >= 8  # every warm-up session is sent
    assert all(s.system == "You are terse." and s.id not in ids for s in warm.session_list)
    assert not any("asks question" in u for s in warm.session_list for u in s.users)


def test_a_system_prompt_no_two_sessions_share_is_not_warmed():
    wl = Workload(sessions=True, source="file:x", session_list=[Session("a", "only a", (Step("q"),)),
                                                                 Session("b", "only b", (Step("q"),))])
    synthetic = Workload(n_prompts=2, prefill_tokens=32, decode_tokens=16, concurrencies=(2,))
    assert all(s.system == "" for s in M._session_warmup(wl, synthetic).session_list)


def test_load_generator_processes_get_whole_sessions():
    wl = replace(_level(n=6, c=3, budget_per_slot=50))
    app = gated_engine(1, 18)  # one answer at a time; three processes keep their own slots filled
    with _Server(app) as s:
        out, wall = M._drive_clients(s.url, LlmtraceHooks(model_name="m"), wl, 3, 30.0, clients=3)
    sent = [o for o in out if o.skipped is None]
    assert len(sent) == 18 and all(o.ok for o in sent) and wall > 0
    turns = defaultdict(list)
    for sid, t in app.state.order:
        turns[sid].append(t)
    assert sorted(turns) == sorted(s.id for s in wl.session_list)
    assert all(sorted(t) == [1, 2, 3] for t in turns.values())  # every step of every session, from one process
