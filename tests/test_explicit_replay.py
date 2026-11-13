"""Replaying explicit requests over HTTP: the file's messages exactly, with chained requests kept back to back."""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate import datasets
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload, get_workload, workload_from_file
from tests.test_measure_and_proxy import _Server
from tests.test_session_scheduler import simulate

SYSTEM = {"role": "system", "content": "You are an agent with tools."}


def agent_lines(n=30, steps=5):
    """Sessions whose every request is given whole: step 2 rewrites the middle of the prompt, and steps 3 and 4 follow
    the step before them at once (a tool call's result, then a heartbeat)."""
    lines = []
    for i in range(n):
        reqs = []
        for k in range(steps):
            head = [SYSTEM, {"role": "system", "content": f"memory of a{i}: " + ("edited " if k >= 2 else "first ")}]
            tail = [{"role": "user", "content": f"a{i} says {j} " + "word " * 5} for j in range(k + 1)]
            req = {"messages": head + tail, "tag": "edit" if k == 2 else ""}
            if k in (3, 4):
                req["chain"] = True
            if k == 1:
                req["max_tokens"] = 7
            reqs.append(req)
        lines.append({"session": f"a{i}", "requests": reqs})
    return lines


def write_agent_file(tmp_path, n=30, steps=5, name="agent.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in agent_lines(n, steps)) + "\n", encoding="utf-8")
    return p


def _key(body):
    """(session, request number) of a body agent_lines wrote: request k holds k user messages."""
    memory = body["messages"][1]["content"]
    return memory.split()[2].rstrip(":"), sum(m["role"] == "user" for m in body["messages"])


def recording_engine(delay_s=0.0, serial=False) -> FastAPI:
    """Records every body, and with `serial` answers one request at a time, logging when each arrives and when its
    answer is done, so the order requests reach the engine can be checked against when slots came free."""
    app = FastAPI()
    app.state.bodies, app.state.events = [], []
    lock = {"lock": None}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.bodies.append(body)
        app.state.events.append(("arrive", _key(body)))
        if lock["lock"] is None:
            lock["lock"] = asyncio.Lock()

        async def gen():
            if serial:
                await lock["lock"].acquire()
            try:
                await asyncio.sleep(delay_s)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': 'done'}}]})}\n\n"
                usage = {"prompt_tokens": sum(len(m["content"]) for m in body["messages"]), "completion_tokens": 1}
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                app.state.events.append(("done", _key(body)))
                if serial:
                    lock["lock"].release()

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def level(path, c, pool=0, per_slot=5) -> Workload:
    sessions = datasets.read_sessions(path)
    return Workload(sessions=True, source="file:unused", session_list=sessions, prompts=[s.first_turn for s in sessions],
                    n_prompts=len(sessions), concurrencies=(c,), prefill_tokens=4096, decode_tokens=64,
                    natural_stop=True, turns_per_slot=per_slot, session_pool=pool, history="explicit", fitted=True)


def drive(app, wl, c):
    with _Server(app) as s:
        return asyncio.run(M._drive(s.url, LlmtraceHooks(model_name="m"), wl, c, 30.0, TokenCounter()))


def test_every_body_is_the_files_messages_exactly(tmp_path):
    p = write_agent_file(tmp_path, n=6)
    wl = level(p, 2, per_slot=15)  # a budget of every request of all six sessions
    app = recording_engine()
    out = drive(app, wl, 2)
    assert all(o.ok for o in out if o.skipped is None)
    index = {json.dumps(r["messages"]): (line["session"], k, r) for line in agent_lines(6)
             for k, r in enumerate(line["requests"])}
    sent = defaultdict(list)
    for body in app.state.bodies:
        sid, k, req = index[json.dumps(body["messages"])]  # a KeyError would be a body the file never gave
        sent[sid].append(k)
        assert body["max_tokens"] == req.get("max_tokens", 64)
    assert dict(sent) == {f"a{i}": [0, 1, 2, 3, 4] for i in range(6)}  # whole sessions, in order
    tags = defaultdict(list)
    for o in sorted(out, key=lambda o: o.seq):
        tags[o.session].append(o.tag)
        assert o.lcp is not None and o.new is not None and (o.lcp == 0.0) == (o.turn == 1)
    assert set(map(tuple, tags.values())) == {("first", "append", "edit", "append", "append")}


def test_a_chained_request_is_sent_at_once_and_the_rest_wait_their_turn(tmp_path):
    c, pool = 2, 8
    ss = datasets.read_sessions(write_agent_file(tmp_path, n=40))
    sched = M._SessionScheduler(ss, c, pool, budget=120, fits=lambda s, m: True, history="explicit")
    _, _, sent, _ = simulate(sched, c)
    chained = [meta for _, _, meta in sent.values() if meta.chained]
    waiting = [meta for _, _, meta in sent.values() if meta.idle_gap is not None and not meta.chained]
    assert chained and {(m.turn, m.idle_gap) for m in chained} <= {(4, 0), (5, 0)}  # nothing in between
    full = [m.idle_gap for m in waiting if m.live == pool]
    assert full and min(full) >= pool - c  # an unchained step waits behind the rest of the pool


def test_over_http_a_chained_request_is_the_next_to_arrive_once_its_predecessor_is_done(tmp_path):
    """On an engine that answers one request at a time, a slot comes free only when an answer is done: what reaches
    the engine next is its chained successor, or another session's chained request freed at the same moment, never
    a request that waits in the queue."""
    c, pool = 2, 8
    wl = level(write_agent_file(tmp_path, n=40), c, pool)
    app = recording_engine(delay_s=0.03, serial=True)
    drive(app, wl, c)
    events = app.state.events
    arrived = [key for kind, key in events if kind == "arrive"]
    checked = 0
    for i, (kind, (sid, k)) in enumerate(events):
        if kind != "done" or k not in (3, 4) or (sid, k + 1) not in arrived:  # requests 3 and 4 chain to the next
            continue
        after = [key for what, key in events[i + 1:] if what == "arrive"]
        between = after[:after.index((sid, k + 1))]
        assert all(n in (4, 5) for _, n in between), (sid, k, after[:3])  # requests 4 and 5 are the chained ones
        checked += 1
    assert checked >= 4


def _window_session():
    """Ten explicit steps; step 3 follows step 2 at once, and steps 7 and 8 follow 6."""
    chains = {2, 6, 7}
    steps = tuple(datasets.Step(messages=(("user", f"r{i}"),), chain=i in chains) for i in range(10))
    return datasets.Session(id="w", system="", steps=steps)


def test_a_window_is_a_primer_then_whole_chains():
    s = _window_session()

    def shape(where):
        w = datasets.window(s, 3, where)
        return [(st.messages[0][1], st.primer) for st in w.steps]

    # Valid starts are 0, 1, 2, 5 and 6: a primer neither follows the step before it nor chains to the next. The
    # target is the point's step less half the window, at most 7 (room for 3); the nearest start wins, the earlier
    # on a tie.
    assert shape(0.0) == [("r0", False), ("r1", False), ("r2", False), ("r3", False)]  # no primer; the chain ends at 3
    assert shape(0.3) == [("r1", True), ("r2", False), ("r3", False), ("r4", False)]  # target 2
    assert shape(0.55) == [("r4", True), ("r5", False), ("r6", False), ("r7", False), ("r8", False)]  # target 4
    assert shape(0.75) == [("r5", True), ("r6", False), ("r7", False), ("r8", False)]  # target 6; to the chain's end
    assert shape(0.95) == shape(0.75)  # target 7: never past the last start that leaves room for the window


def test_a_levels_windows_hold_the_files_mix_of_early_and_late_requests(tmp_path):
    lines = []
    for i in range(90):
        reqs = [{"messages": [{"role": "user", "content": f"s{i} r{k}"}],
                 "tag": ("early", "middle", "late")[3 * k // 30]} for k in range(30)]
        lines.append({"session": f"s{i}", "requests": reqs})
    p = tmp_path / "phases.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    source = datasets.FILE_PREFIX + str(p)
    for offset in (0, 30, 60):
        got = datasets.sample_sessions(source, 30, offset, turns_cap=6)
        measured = [st.tag for s in got for st in s.steps if not st.primer]
        share = {t: measured.count(t) / len(measured) for t in ("early", "middle", "late")}
        assert all(abs(v - 1 / 3) < 0.07 for v in share.values()), share
        assert all(1 <= sum(st.primer for st in s.steps) + (s.steps[0].messages[0][1].endswith(" r0")) for s in got)
    assert datasets.sample_sessions(source, 30, 0, 6) == datasets.sample_sessions(source, 30, 0, 6)  # seeded


def test_a_level_is_a_window_per_slot_and_a_pool_is_capped_by_the_concurrency(tmp_path):
    p = write_agent_file(tmp_path, n=200, steps=12)
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.turns_cap == 6 and wl.turns_per_slot == 7 and wl.name.endswith("-sessions-explicit")
    assert [wl.turn_budget(c) for c in wl.concurrencies] == [8, 28, 56]  # a primer and six measured, per slot
    pooled = workload_from_file(p, get_workload("chat"), session_pool=64)
    assert [pooled.pool_at(c) for c in pooled.concurrencies] == [8, 32, 64]
    assert [pooled.turn_budget(c) for c in pooled.concurrencies] == [56, 224, 448]  # a window per live session
    assert workload_from_file(p, turns_cap=6).name.endswith("-sessions-explicit-t6")  # asked for, so named
    assert workload_from_file(p, turns_cap=4).turns_per_slot == 5
    ws = pooled.ensure_prompts().session_list
    assert ws and all(s.turns <= 12 for s in ws) and any(s.steps[0].primer for s in ws)


def test_requests_past_the_budget_only_finish_chains_begun_inside_it(tmp_path):
    ss = [datasets.window(s, 6, (i % 10) / 10) for i, s in enumerate(datasets.read_sessions(
        write_agent_file(tmp_path, n=60, steps=12)))]
    for budget in (13, 20, 27):
        sched = M._SessionScheduler(ss, 4, 16, budget=budget, fits=lambda s, m: True, history="explicit")
        _, _, sent, _ = simulate(sched, 4)
        by_session = defaultdict(dict)
        for (sid, turn), (_, _, meta) in sent.items():
            by_session[sid][turn] = meta
        past = [meta for _, _, meta in sent.values() if meta.seq >= budget]
        for meta in past:
            assert meta.chained
            turn = meta.turn
            while by_session[meta.session][turn].chained:
                turn -= 1
            assert by_session[meta.session][turn].seq < budget  # the chain's first request was inside the budget


def test_a_primer_loads_the_level_but_is_not_measured():
    def out(ttft, primer):
        return M.RequestOutcome(ok=True, ttft_s=ttft, duration_s=ttft + 0.5, tokens=11, token_source="usage",
                                prompt_tokens=100, session="a", turn=1, primer=primer)

    m = M._metrics_from([out(9.0, True), out(0.1, False), out(0.2, False)], 1.0, 1)
    assert m.ttft_samples_ms == [100.0, 200.0] and m.output_tokens == 33 and m.requests == 3


def test_the_shared_system_prompt_is_the_common_start_of_first_messages():
    def session(sid, memory):
        msgs = (("system", "Instructions for every agent. Memory: " + memory), ("user", "hi"))
        return datasets.Session(id=sid, system=msgs[0][1], steps=(datasets.Step(messages=msgs),))

    wl = Workload(sessions=True, source="file:x", history="explicit",
                  session_list=[session("a", "likes tea"), session("b", "likes coffee"), session("c", "lives here")])
    assert M._shared_system(wl) == "Instructions for every agent. Memory: li"
    synthetic = Workload(n_prompts=2, prefill_tokens=32, decode_tokens=16, concurrencies=(2,))
    assert {s.system for s in M._session_warmup(wl, synthetic).session_list} == {M._shared_system(wl)}
    alone = Workload(sessions=True, source="file:x", history="explicit", session_list=[session("a", "x")])
    assert M._shared_system(alone) == ""


# --------------------------------------------------------------------------- by kind


def _out(sid, turn, tag, prompt, cached, ttft, lcp, new, started, chained=False, reusable=0):
    return M.RequestOutcome(ok=True, ttft_s=ttft, duration_s=ttft + 0.1, tokens=3, token_source="usage",
                            prompt_tokens=prompt, cached_tokens=cached, session=sid, turn=turn, tag=tag, lcp=lcp,
                            new=new, started_s=started, chained=chained, request_chars=prompt, reusable_chars=reusable)


def test_reuse_and_time_to_first_token_are_split_by_kind():
    outs = [_out("a", 1, "first", 100, 0, 0.10, 0.0, 1.0, 10.0),
            _out("a", 2, "append", 200, 100, 0.02, 0.5, 0.5, 10.2, reusable=100),
            _out("a", 3, "heartbeat", 300, 200, 0.03, 0.66, 0.34, 10.25, chained=True, reusable=200),
            _out("a", 4, "core_edit", 300, 30, 0.30, 0.1, 0.2, 10.4, reusable=30),
            _out("b", 1, "first", 100, None, 0.12, 0.0, 1.0, 20.0),  # no count: zero, as in the level
            _out("b", 2, "append", 200, 100, 0.04, 0.5, 0.5, 20.3, chained=True, reusable=100),
            _out("b", 3, "append", 300, 200, 0.06, 0.6, 0.4, 20.5, chained=True, reusable=200),
            M.RequestOutcome(ok=False, skipped="truncated", session="b", turn=4)]
    m = M._metrics_from(outs, 1.0, 2)
    by = m.reuse_by_kind
    assert list(by) == ["append", "core_edit", "first", "heartbeat"]
    assert by["core_edit"] == {"requests": 1, "prompt_tokens": 300, "ttft_ms": 300.0, "reuse_expected": 0.1,
                               "prefix_reuse": 0.1, "refill_behind_edit": 0.7}
    assert by["append"]["requests"] == 3 and by["append"]["ttft_ms"] == 40.0
    assert by["append"]["prefix_reuse"] == round(400 / 700, 4) == by["append"]["reuse_expected"]
    assert by["first"]["prefix_reuse"] == 0.0 and by["first"]["refill_behind_edit"] == 0.0
    # a's chain: turn 2 sent at 10.2, the heartbeat's first token at 10.25 + 0.03; b's: 20.0 to 20.5 + 0.06.
    assert m.chain_latency_ms == round(statistics.median([80.0, 560.0]), 1)


def test_a_turn_based_level_reports_nothing_by_kind():
    m = M._metrics_from([M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.2, tokens=3, prompt_tokens=50,
                                          cached_tokens=10, session="s", turn=1, request_chars=10)], 1.0, 1)
    assert m.reuse_by_kind == {} and m.chain_latency_ms is None and m.refill_behind_edit is None


def scripted_engine(counts) -> FastAPI:
    """Answers "done" and reports, for the n-th request it receives, the (prompt, cached) counts it is given: written
    by hand below, so what replay records is checked against arithmetic rather than against a second copy of it."""
    app = FastAPI()
    app.state.bodies = []

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        prompt, cached = counts[len(app.state.bodies)]
        app.state.bodies.append(body)

        async def gen():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'done'}}]})}\n\n"
            usage = {"prompt_tokens": prompt, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": cached}}
            yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def test_by_kind_matches_hand_counted_characters(tmp_path):
    """One session of three requests, counted by hand. A rendered message is 2 + len(role) + len(content) characters.

      1  [system "Agent rules.", user "hello"]                           20 + 11 = 31      first
      2  1 + [assistant "ok", user "more"]                               31 + 13 + 10 = 54 append
      3  2 with the system message "Agent rulez.", sent at once (chain)  54                rewrite_head

    Reusable, as a never-evicting engine would cache it: 0 for the first (no other session shares its system prompt);
    for the second, the first and the header of the engine's answer, "\\x00assistant\\x01" (31 + 11 = 42); for the
    third, "\\x00system\\x01Agent rule" (18). The third's refill behind the edit: the "." after the edit and the three
    messages it kept (1 + 11 + 13 + 10 = 35 of 54); its only new character is the "z". The engine reports
    characters as tokens."""
    def m(role, content):
        return {"role": role, "content": content}

    one = [m("system", "Agent rules."), m("user", "hello")]
    two = one + [m("assistant", "ok"), m("user", "more")]
    three = [m("system", "Agent rulez.")] + two[1:]
    p = tmp_path / "hand.jsonl"
    p.write_text(json.dumps({"session": "h", "requests": [{"messages": one}, {"messages": two},
                                                          {"messages": three, "chain": True}]}) + "\n",
                 encoding="utf-8")
    out = drive(scripted_engine([(31, 0), (54, 42), (54, 18)]), level(p, 1, per_slot=10), 1)
    got = M._metrics_from(out, 1.0, 1)
    by = got.reuse_by_kind
    assert {k: (v["requests"], v["prompt_tokens"], v["reuse_expected"], v["prefix_reuse"], v["refill_behind_edit"])
            for k, v in by.items()} == {"first": (1, 31, 0.0, 0.0, 0.0), "append": (1, 54, 0.7778, 0.7778, 0.0),
                                        "rewrite_head": (1, 54, 0.3333, 0.3333, 0.6481)}
    assert got.refill_behind_edit == round(35 / 139, 4) and got.prefix_reuse == round(60 / 139, 4)
    assert got.chain_latency_ms is not None and got.chain_latency_ms > 0  # requests 2 and 3


def test_the_profile_records_refill_and_chain_latency_beside_reuse(tmp_path):
    served = M.TrialMetrics(refill_behind_edit=0.12, chain_latency_ms=345.6, prefix_reuse=0.7)
    spec = workload_from_file(write_agent_file(tmp_path), get_workload("chat")).spec(served)
    assert (spec["refill_behind_edit"], spec["chain_latency_ms"], spec["reuse_achieved"]) == (0.12, 345.6, 0.7)
    turn_based = get_workload("chat").spec(served)
    assert "refill_behind_edit" not in turn_based
