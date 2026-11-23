"""rigtune.reasoning.methods: what each way of answering sends, counts and decides, against a scripted endpoint."""

from __future__ import annotations

import asyncio

import pytest

from rigtune.reasoning import methods as rm
from rigtune.reasoning import problems as rp
from rigtune.reasoning.methods import Settings, majority, split_thinking
from rigtune.reasoning.problems import Problem

PROBLEM = Problem(id="0" * 16, subject="algebra", level=5, problem="What is $6/8$ in lowest terms?",
                  answer="\\frac{3}{4}")
SETTINGS = Settings(model="m/m", max_tokens=512, thinking=False, seed=7)


@pytest.fixture(autouse=True)
def _no_math_verify(monkeypatch):
    rp._parsed.cache_clear()  # readings cached under another stub
    monkeypatch.setattr(rp, "_math_verify", lambda: None)


class Endpoint:
    """Answers each request (always one answer) with the next scripted message, and records the bodies."""

    def __init__(self, *messages, cached=0, fail_on=None):
        self.messages, self.bodies, self.cached, self.fail_on = list(messages), [], cached, fail_on

    async def __call__(self, body):
        self.bodies.append(body)
        assert body["n"] == 1, "every request asks for one answer"
        if self.fail_on is not None and len(self.bodies) == self.fail_on:
            raise RuntimeError("400: prompt too long")
        m = self.messages.pop(0)
        message = m if isinstance(m, dict) else {"content": m}
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 25,
                          "prompt_tokens_details": {"cached_tokens": self.cached}}}


def run(coro):
    return asyncio.run(coro)


def boxed(x):
    return "Working... so the answer is $\\boxed{" + x + "}$."


def test_majority_votes_by_equality_and_keeps_the_first_writing():
    assert majority(["\\dfrac{3}{4}", "1", "\\frac34"]) == "\\dfrac{3}{4}"
    assert majority(["1", "2"]) == "1"  # a tie goes to the first seen
    assert majority([None, None, "5"]) == "5"  # no answer does not vote
    assert majority([None, ""]) is None


def test_votes_group_answers_equal_as_maths(monkeypatch):
    """0.75 and \\frac34 are one answer to math-verify; by the authors' normal form they would split the vote."""
    def verify(x, y):
        return x == y or {x[0], y[0]} == {"$0.75$", "$\\frac34$"}

    monkeypatch.setattr(rp, "_math_verify", lambda: (lambda t: [t], verify))
    assert majority(["1", "0.75", "1", "\\frac34", "\\frac34"]) == "0.75"


@pytest.mark.parametrize("message, thinking, text", [
    ({"content": "x"}, "", "x"),
    ({"reasoning_content": "hmm", "content": "\\boxed{2}"}, "hmm", "\\boxed{2}"),  # the server split it
    ({"content": "<think>try \\boxed{1}</think>So \\boxed{2}"}, "try \\boxed{1}", "So \\boxed{2}"),  # it did not
    ({"content": "<think>try \\boxed{1} and then"}, "try \\boxed{1} and then", ""),  # cut off while thinking
])
def test_thinking_is_kept_apart_from_the_answer(message, thinking, text):
    assert split_thinking(message) == (thinking, text)


def test_a_reply_cut_off_with_no_tag_was_cut_off_thinking():
    """Templates that write <think> into the prompt leave no tag in a cut-off reply."""
    msg = {"content": "Let me try x = 3 first, so \\boxed{3}? But wait, the constraint says"}
    assert split_thinking(msg, "length") == (msg["content"], "")
    assert split_thinking(msg, "length", thinking=False) == ("", msg["content"])  # thinking off: it was answering
    assert split_thinking(msg, "stop") == ("", msg["content"])


def test_a_provisional_answer_inside_cut_off_thinking_is_no_answer():
    a = run(rm.direct(Endpoint({"content": "<think>maybe \\boxed{3}"}), SETTINGS, PROBLEM))
    assert a.final is None and a.calls[0].thinking_chars > 0


def test_direct_asks_once_with_the_settings_and_does_not_grade():
    ep = Endpoint(boxed("\\dfrac{3}{4}"), cached=8)
    a = run(rm.direct(ep, SETTINGS, PROBLEM))
    body = ep.bodies[0]
    assert body["max_tokens"] == 512 and body["seed"] == 7 and body[rm.PRIORITY] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "\\boxed{}" in body["messages"][0]["content"] and PROBLEM.problem in body["messages"][0]["content"]
    assert (a.final, a.answers, a.grader, a.correct) == ("\\dfrac{3}{4}", ["\\dfrac{3}{4}"], "ungraded", False)
    assert (a.calls[0].prompt_tokens, a.calls[0].completion_tokens, a.calls[0].cached_tokens) == (40, 25, 8)


def test_self_consistency_sends_k_requests_of_one_with_k_different_seeds():
    ep = Endpoint(boxed("1"), boxed("\\frac{3}{4}"), "no box", boxed("0.75"), boxed("\\frac34"))
    a = run(rm.self_consistency(ep, SETTINGS, PROBLEM, k=5))
    assert len(ep.bodies) == 5 and sorted(b["seed"] for b in ep.bodies) == [7, 8, 9, 10, 11]
    assert all(b["messages"] == ep.bodies[0]["messages"] for b in ep.bodies)  # one prompt: the prefix cache shares it
    assert a.answers == ["1", "\\frac{3}{4}", None, "0.75", "\\frac34"]
    assert a.final is None  # its majority is taken after the batch's clock (run.grade_attempt)
    assert a.method == "self-consistency-5" and a.completion_tokens == 125


def test_self_correction_critiques_its_answer_text_without_a_hint():
    first = {"reasoning_content": "long thinking", "content": boxed("\\frac{6}{8}")}
    ep = Endpoint(first, boxed("\\frac{3}{4}"))
    a = run(rm.self_correction(ep, SETTINGS, PROBLEM))
    one, two = ep.bodies
    assert [m["role"] for m in two["messages"]] == ["user", "assistant", "user"]
    assert two["messages"][1]["content"] == first["content"]  # the answer text, not the thinking
    assert two["messages"][2]["content"] == rm.CRITIQUE
    # Nothing but the conversation, its seed and its priority differs from the first request.
    assert {k: v for k, v in one.items() if k not in ("messages", "seed", rm.PRIORITY)} == {
        k: v for k, v in two.items() if k not in ("messages", "seed", rm.PRIORITY)}
    assert (two["seed"], two[rm.PRIORITY]) == (8, 1)
    assert (a.first, a.final, a.answers) == ("\\frac{6}{8}", "\\frac{3}{4}", ["\\frac{6}{8}", "\\frac{3}{4}"])


def test_a_failed_critique_turn_keeps_the_first_answer_and_gives_no_final_one():
    a = run(rm.self_correction(Endpoint(boxed("\\frac{3}{4}"), fail_on=2), SETTINGS, PROBLEM))
    assert (a.first, a.final, len(a.calls)) == ("\\frac{3}{4}", None, 1) and a.error.startswith("critique turn")


def test_adaptive_stops_at_two_when_they_agree():
    ep = Endpoint(boxed("\\frac{3}{4}"), boxed("\\dfrac{3}{4}"))
    a = run(rm.adaptive(ep, SETTINGS, PROBLEM, k=8))
    assert len(ep.bodies) == 2 and len(a.answers) == 2 and a.final == "\\frac{3}{4}"


def test_adaptive_samples_up_to_k_when_they_disagree_never_reusing_a_seed():
    ep = Endpoint(boxed("1"), boxed("\\frac{3}{4}"),
                  boxed("\\frac34"), boxed("2"), boxed("\\frac{3}{4}"), boxed("1"), boxed("3"), "none")
    a = run(rm.adaptive(ep, SETTINGS, PROBLEM, k=8))
    seeds = [b["seed"] for b in ep.bodies]
    assert sorted(seeds) == list(range(7, 15))  # eight answers, eight seeds: none repeated
    assert [b[rm.PRIORITY] for b in ep.bodies] == [0, 0] + [1] * 6  # the extra answers are follow-ups
    assert len(a.answers) == 8 and a.final is None  # its majority is taken after the clock
    ep = Endpoint(boxed("1"), "none", boxed("1"))
    assert run(rm.adaptive(ep, SETTINGS, PROBLEM, k=3)).answers == ["1", None, "1"]


def test_adaptive_s_failed_second_round_gives_no_final_answer():
    a = run(rm.adaptive(Endpoint(boxed("1"), boxed("2"), fail_on=3), SETTINGS, PROBLEM, k=4))
    assert (a.answers, a.final) == (["1", "2"], None) and a.error.startswith("more answers")


def test_the_template_default_is_left_alone_when_thinking_is_unset():
    ep = Endpoint(boxed("1"))
    run(rm.direct(ep, Settings(model="m/m"), PROBLEM))
    assert "chat_template_kwargs" not in ep.bodies[0] and "seed" not in ep.bodies[0]


def test_the_http_client_has_no_connection_cap_and_sends_only_the_engine_s_fields(monkeypatch):
    import httpx

    made, sent = {}, []

    class Client:
        def __init__(self, **kw):
            made.update(kw)

        async def post(self, path, json):
            sent.append(json)
            return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "http://x" + path))

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    post, close = rm.http_post("http://127.0.0.1:8000/")
    assert made["limits"].max_connections is None  # httpx's default of 100 would cap every n = 1 method
    run(post({"model": "m", rm.PRIORITY: 1, rm.SERVICE_SECONDS: 3.0}))
    assert sent == [{"model": "m"}]
    run(close())


def test_cached_tokens_the_server_does_not_report_are_none():
    class Bare(Endpoint):
        async def __call__(self, body):
            reply = await super().__call__(body)
            del reply["usage"]["prompt_tokens_details"]
            return reply

    assert run(rm.direct(Bare(boxed("1")), SETTINGS, PROBLEM)).calls[0].cached_tokens is None
