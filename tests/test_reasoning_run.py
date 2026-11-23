"""rigtune.reasoning.run: a method over many problems, limited in sequences, graded after its clock stops."""

from __future__ import annotations

import asyncio

import pytest

from rigtune.reasoning import methods as rm
from rigtune.reasoning import problems as rp
from rigtune.reasoning import run as rr
from rigtune.reasoning.methods import Settings
from rigtune.reasoning.problems import Problem

PROBLEMS = [Problem(id=f"{i:016x}", subject="algebra", level=5, problem=f"p{i}", answer=str(i)) for i in range(12)]


@pytest.fixture(autouse=True)
def _no_math_verify(monkeypatch):
    rp._parsed.cache_clear()  # readings cached under another stub
    monkeypatch.setattr(rp, "_math_verify", lambda: None)


class Slow:
    """Answers every problem correctly after a short wait; records sequences in flight and the order calls began."""

    def __init__(self, fail=(), fail_turn=None):
        self.running = self.peak = 0
        self.fail, self.fail_turn, self.order = set(fail), fail_turn, []

    async def __call__(self, body):
        assert rm.PRIORITY not in body, "the limit strips rigtune's own fields"
        text = body["messages"][0]["content"].rsplit("p", 1)[1]
        turn = (len(body["messages"]) + 1) // 2
        self.order.append((text, turn))
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        if text in self.fail or (self.fail_turn == turn):
            raise RuntimeError("engine said 400")
        return {"choices": [{"index": 0, "message": {"content": "\\boxed{" + text + "}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def run(coro):
    return asyncio.run(coro)


def test_sequences_in_flight_are_limited_and_the_limit_stays_full():
    ep = Slow()
    r = run(rr.run_method(ep, Settings(model="m"), PROBLEMS, "self-consistency", k=4, sequences=8))
    assert ep.peak <= 8 and r.peak_sequences == 8
    assert len(r.attempts) == 12 and all(a.correct for a in r.attempts)  # graded after the batch
    assert r.name == "self-consistency-4" and 0.5 < r.utilisation <= 1.0


def test_a_follow_up_turn_goes_before_new_problems():
    ep = Slow()
    run(rr.run_method(ep, Settings(model="m"), PROBLEMS[:3], "self-correction", sequences=1))
    assert ep.order == [("0", 1), ("0", 2), ("1", 1), ("1", 2), ("2", 1), ("2", 2)]  # no tail of second turns


def test_service_time_leaves_out_the_wait_for_a_place():
    r = run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS, "direct", sequences=1))
    assert max(c.seconds for a in r.attempts for c in a.calls) < 0.1  # each ran ~0.01 s; the last waited ~0.11 s


def test_energy_is_the_counter_s_difference_over_the_batch():
    readings = iter([1000.0, 1450.5])
    r = run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS, "direct", sequences=4, energy=lambda: next(readings)))
    assert r.energy_joules == pytest.approx(450.5) and r.k is None and r.name == "direct"


def test_a_failed_problem_is_kept_as_wrong_never_dropped():
    r = run(rr.run_method(Slow(fail={"3"}), Settings(model="m"), PROBLEMS, "direct", sequences=4))
    assert len(r.attempts) == 12 and len(r.errors) == 1 and r.errors[0].startswith(PROBLEMS[3].id)
    failed = next(a for a in r.attempts if a.error)
    assert (failed.correct, failed.grader) == (False, "error")


def test_a_failed_critique_turn_counts_as_a_harmful_revision_with_its_first_answer_graded():
    r = run(rr.run_method(Slow(fail_turn=2), Settings(model="m"), PROBLEMS[:2], "self-correction", sequences=4))
    assert all((a.first_correct, a.correct, a.grader) == (True, False, "error") for a in r.attempts)


def test_methods_and_k_are_checked_before_anything_runs():
    with pytest.raises(ValueError, match="unknown method"):
        run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS, "beam"))
    with pytest.raises(ValueError, match="k = 1 is direct"):
        run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS, "self-consistency", k=1))
    with pytest.raises(ValueError, match="always takes 2"):
        run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS, "adaptive", k=2))


def test_the_run_is_written_with_every_attempt_and_its_problem_set():
    r = run(rr.run_method(Slow(), Settings(model="m", seed=3), PROBLEMS[:2], "self-correction", sequences=4,
                          problem_set="abc"))
    out = r.to_json()
    assert out["name"] == "self-correction" and out["settings"]["seed"] == 3 and out["problem_set"] == "abc"
    assert len(out["attempts"]) == 2 and len(out["attempts"][0]["calls"]) == 2 and "utilisation" in out


def test_majorities_are_taken_after_the_clock():
    r = run(rr.run_method(Slow(), Settings(model="m"), PROBLEMS[:3], "self-consistency", k=3, sequences=8))
    assert [a.final for a in r.attempts] == ["0", "1", "2"] and all(a.correct for a in r.attempts)


def test_an_urgent_request_leaves_no_free_place_idle():
    async def scenario():
        limit = rr.SequenceLimit(4)
        for _ in range(4):
            await limit.acquire()
        got = []

        async def waiter(name, priority):
            await limit.acquire(1, priority)
            got.append(name)

        tasks = [asyncio.create_task(waiter(n, p)) for n, p in (("a0", 0), ("b0", 0), ("u1", 1))]
        await asyncio.sleep(0)
        await limit.release()
        await limit.release()
        await asyncio.sleep(0.01)
        for t in tasks:
            t.cancel()
        return got, limit.used

    got, used = asyncio.run(scenario())
    assert got[0] == "u1" and used == 4  # the urgent one first, and the second free place filled too
