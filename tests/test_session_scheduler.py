"""_SessionScheduler on its own: dispatch order from scripted completions, with no HTTP and no clock."""

from __future__ import annotations

from rigtune.calibrate.datasets import Session, Step
from rigtune.calibrate.measure import RequestOutcome, _render, _SessionScheduler
from rigtune.calibrate.tokens import TokenCounter


def _sessions(n, turns=4, system="Be terse."):
    return [Session(id=f"s{i}", system=system,
                    steps=tuple(Step(user=f"s{i} q{t}", answer=None if t == turns - 1 else f"recorded {i}.{t}")
                                for t in range(turns)))
            for i in range(n)]


def simulate(sched, c, fail=()):
    """Fill c slots, finish the oldest step, repeat: the closed loop the driver runs, in dispatch order."""
    flight, order, fills, sent = [], [], [], {}
    while True:
        while len(flight) < c:
            item = sched.next()
            if item is None:
                break
            flight.append(item)
            order.append((item[2].session, item[2].turn))
            sent[(item[2].session, item[2].turn)] = item
        fills.append((sched.dispatched, len(flight)))
        if not flight:
            break
        state, messages, meta = flight.pop(0)
        key = (meta.session, meta.turn)
        out = RequestOutcome(ok=key not in fail, tokens=3, text=f"reply {meta.session} {meta.turn}",
                             prompt_tokens=10 * meta.turn, seq=meta.seq)
        sched.done(state, messages, out)
    return order, fills, sent, sched.close()


def _always(state, messages):
    return True


def test_the_level_stays_at_its_concurrency_until_the_budget_is_spent():
    sched = _SessionScheduler(_sessions(40), concurrency=8, pool=0, budget=48, fits=_always)
    order, fills, _, skipped = simulate(sched, 8)
    assert len(order) == 48
    assert all(n == 8 for dispatched, n in fills if dispatched < 48)  # closed loop at c, every time it refills
    assert all(s.skipped == "truncated" for s in skipped) and skipped  # live sessions with steps left


def test_one_session_per_slot_follows_its_last_step_at_once():
    sched = _SessionScheduler(_sessions(40), concurrency=2, pool=0, budget=24, fits=_always)
    _, _, sent, _ = simulate(sched, 2)
    distances = [meta.distance for _, _, meta in sent.values() if meta.distance is not None]
    assert distances and set(distances) == {1}  # only the other slot's step comes between
    assert {meta.live for _, _, meta in sent.values()} == {2}


def test_a_pool_sets_the_reuse_distance():
    sched = _SessionScheduler(_sessions(40), concurrency=2, pool=8, budget=60, fits=_always)
    order, _, sent, _ = simulate(sched, 2)
    distances = [meta.distance for _, _, meta in sent.values() if meta.distance is not None]
    assert set(distances) == {7}  # the other 7 live sessions each go once between two steps of one session
    assert {meta.live for _, _, meta in sent.values()} == {8}
    assert order[:9] == [(f"s{i}", 1) for i in range(8)] + [("s0", 2)]


def test_the_idle_gap_is_the_pool_less_the_concurrency():
    sched = _SessionScheduler(_sessions(40), concurrency=2, pool=8, budget=60, fits=_always)
    _, _, sent, _ = simulate(sched, 2)
    gaps = [meta.idle_gap for _, _, meta in sent.values() if meta.idle_gap is not None]
    assert set(gaps) == {6}  # P - c requests start while a session sits idle; its distance is P - 1


def test_a_pool_with_the_real_turn_budget_gets_second_turns(tmp_path):
    """A budget of slots alone (6c = 48 at c = 8) let a pool of 64 send nothing but first turns."""
    import json

    from rigtune.calibrate.measure import _level_workload
    from rigtune.calibrate.workload import get_workload, workload_from_file

    p = tmp_path / "chats.jsonl"
    p.write_text("\n".join(json.dumps({"session": f"c{i}", "messages": [
        {"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"},
        {"role": "user", "content": "more"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "last"}]}) for i in range(1000)) + "\n", encoding="utf-8")
    wl = workload_from_file(p, get_workload("chat"), session_pool=64).ensure_prompts()
    k, c = 2, 8
    level = _level_workload(wl, k, TokenCounter())
    level = level.head(level.level_requests(c))
    assert wl.turn_budget(c) == 192 and len(level.session_list) >= 128
    sched = _SessionScheduler(level.session_list, c, level.session_pool, level.turn_budget(c), _always)
    order, _, sent, _ = simulate(sched, c)
    later = [meta for _, _, meta in sent.values() if meta.turn > 1]
    assert len(order) == 192 and len(later) >= 64
    assert {meta.distance for meta in later} == {63} and {meta.live for _, _, meta in sent.values()} == {64}


def test_a_chained_step_goes_next_even_past_the_budget():
    tool = Session(id="agent", system="", steps=(Step("plan", chain=True), Step("tool result", chain=True),
                                                 Step("answer", tag="final")))
    sched = _SessionScheduler([tool] + _sessions(5), concurrency=1, pool=3, budget=2, fits=_always)
    order, _, sent, skipped = simulate(sched, 1)
    assert order == [("agent", 1), ("agent", 2), ("agent", 3)]  # never interleaved, and the chain finished
    assert sent[("agent", 3)][2].tag == "final"
    assert {s.session for s in skipped} == {"s0", "s1"}  # the other live sessions never got a turn


def test_the_working_set_is_what_live_sessions_last_sent_and_received():
    sched = _SessionScheduler(_sessions(2, turns=3), concurrency=2, pool=0, budget=100, fits=_always)
    _, _, sent, _ = simulate(sched, 2)
    assert sent[("s0", 1)][2].working_set == 0
    # Each finished step leaves its prompt (10 x turn) and its 3-token answer: s1 after turn 1 (13) and s0 after
    # turn 2 (23) when s0's third step goes out.
    assert sent[("s1", 2)][2].working_set == 13 + 13 and sent[("s0", 3)][2].working_set == 13 + 23


def test_never_two_steps_of_a_session_in_flight():
    sched = _SessionScheduler(_sessions(3), concurrency=8, pool=0, budget=100, fits=_always)
    order, fills, _, _ = simulate(sched, 8)
    assert max(n for _, n in fills) == 3 and len(order) == 12  # three sessions, four steps each


def test_each_step_carries_the_last_one_and_its_answer():
    sched = _SessionScheduler(_sessions(1, turns=3), concurrency=1, pool=0, budget=10, fits=_always)
    _, _, sent, _ = simulate(sched, 1)
    first, second, third = (sent[("s0", t)][1] for t in (1, 2, 3))
    assert first == [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "s0 q0"}]
    assert second == first + [{"role": "assistant", "content": "reply s0 1"}, {"role": "user", "content": "s0 q1"}]
    assert third[:len(second)] == second
    # Generated history: everything but the new question is a prefix of what the engine saw.
    meta = sent[("s0", 2)][2]
    assert meta.reusable_chars == len(_render(second[:-1])) and meta.request_chars == len(_render(second))


def test_recorded_history_carries_the_files_answer_and_counts_only_what_the_engine_saw():
    sched = _SessionScheduler(_sessions(1, turns=2), concurrency=1, pool=0, budget=10, fits=_always,
                              history="recorded")
    _, _, sent, _ = simulate(sched, 1)
    first, second = sent[("s0", 1)][1], sent[("s0", 2)][1]
    assert second[2] == {"role": "assistant", "content": "recorded 0.0"}
    seen = _render(first + [{"role": "assistant", "content": "reply s0 1"}])
    # The engine cached its own answer, "reply s0 1"; the file's "recorded 0.0" shares only "re" with it.
    assert sent[("s0", 2)][2].reusable_chars == len(_render(first)) + len("\x00assistant\x01") + len("re")
    assert seen.startswith(_render(first))


def test_a_shared_system_prompt_is_what_a_first_step_can_reuse():
    sched = _SessionScheduler(_sessions(2), concurrency=1, pool=0, budget=2, fits=_always, shared_system="Be terse.")
    _, _, sent, _ = simulate(sched, 1)
    meta = sent[("s0", 1)][2]
    assert meta.reusable_chars == len(_render([{"role": "system", "content": "Be terse."}])) > 0


def test_a_failed_step_ends_its_session_and_leaves_the_others_alone():
    sched = _SessionScheduler(_sessions(2, turns=3), concurrency=2, pool=0, budget=100, fits=_always)
    order, _, sent, _ = simulate(sched, 2, fail={("s0", 2)})
    assert ("s0", 3) not in order and [t for s, t in order if s == "s1"] == [1, 2, 3]
    third = sent[("s1", 3)][1]
    assert [m["content"] for m in third if m["role"] == "assistant"] == ["reply s1 1", "reply s1 2"]


def test_a_step_that_would_not_fit_stops_its_session_as_a_cut():
    def fits(state, messages):
        return state.step == 0  # only first steps fit

    sched = _SessionScheduler(_sessions(3, turns=3), concurrency=1, pool=0, budget=100, fits=fits)
    order, _, _, skipped = simulate(sched, 1)
    assert order == [("s0", 1), ("s1", 1), ("s2", 1)]
    assert [(s.session, s.turn, s.skipped) for s in skipped] == [("s0", 2, "cut"), ("s1", 2, "cut"), ("s2", 2, "cut")]


def test_the_next_prompt_is_estimated_from_what_the_engine_counted():
    sched = _SessionScheduler(_sessions(1, turns=2), concurrency=1, pool=0, budget=10, fits=_always,
                              history="recorded")
    state, messages, meta = sched.next()
    sched.done(state, messages, RequestOutcome(ok=True, tokens=300, text="x", prompt_tokens=50, seq=meta.seq))
    # Recorded history carries the file's 12-character answer, not the 300 tokens generated.
    words = TokenCounter()
    assert state.next_prompt_tokens(state.request(), words) == 50 + (12 // 4 + 1) + (len("s0 q1") // 4 + 1) + 32
