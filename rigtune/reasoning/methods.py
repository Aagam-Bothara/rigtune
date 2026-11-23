"""The ways the same model may answer a problem, each against an OpenAI-compatible endpoint (vLLM, SGLang).

- direct: one answer.
- self-consistency: k answers and the most common of them. Each answer is its own request of one, sent together, so
  an answer that finishes frees its place at once (one request of n holds all n places until the longest is done,
  which left half of them idle at k = 8); the engine's prefix cache computes the shared prompt once.
- self-correction: one answer, then, in the same conversation, a turn asking the model to name a specific error in it
  or say why none needs revising, then its final answer. It is never told whether the first answer was right: the
  checker stays outside the loop, so this measures correction without feedback.
- adaptive: two answers; if they agree that is the answer, else more answers up to k and the most common. Easy
  problems cost about two answers, hard ones about k.

Answers are compared with each other (votes, agreement) by math-verify when it works (rigtune.reasoning.problems), so
0.75 and \\frac34 vote together, never against the reference. A reply's thinking (vLLM's `reasoning_content`, or the
text before `</think>`) is kept apart from its answer: only the text after the thinking is searched for an answer and
sent back to the critique turn, the same whether or not the server splits it. A reply cut off while thinking has no
answer. Methods do not grade: rigtune.reasoning.run grades after the batch's clock stops.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from rigtune.reasoning.problems import Problem, equivalent, last_boxed

SOLVE = "Solve the following problem. Reason step by step, and put your final answer in \\boxed{{}}.\n\n{problem}"
# Asks for a specific error or a reason to keep the answer; says nothing of whether the answer was right.
CRITIQUE = ("Review your solution above. Identify one specific error in it, or explain why no revision is justified. "
            "Then give your final answer in \\boxed{}.")
SERVICE_SECONDS = "_rigtune_service_seconds"  # set by run.limited: the time a request ran, its wait for a place left out
PRIORITY = "_rigtune_priority"  # read and removed by run.limited: a follow-up call goes before new problems

Post = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class Call:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: Optional[int]  # None when the server does not report it (vLLM: --enable-prompt-tokens-details)
    seconds: float
    finish_reason: str
    thinking_chars: int = 0
    text: Optional[str] = None  # the answer text and thinking, kept only with Settings.keep_texts: training data for
    thinking: Optional[str] = None  # a verifier or a STaR adapter (.git/wip/plan-reasoning.md)


@dataclass
class Attempt:
    method: str
    problem_id: str
    answers: List[Optional[str]]  # every answer the method saw, in order (None: no \boxed{} answer)
    final: Optional[str]
    first: Optional[str] = None  # self-correction: its first answer
    calls: List[Call] = field(default_factory=list)
    error: Optional[str] = None  # a request failed: graded wrong, never dropped
    correct: bool = False  # filled by rigtune.reasoning.run.grade_attempt
    grader: str = "ungraded"
    first_correct: Optional[bool] = None

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def truncated(self) -> int:
        return sum(1 for c in self.calls if c.finish_reason == "length")

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Settings:
    model: str
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.95
    thinking: Optional[bool] = None  # chat_template_kwargs enable_thinking; None leaves the template's default
    # chat_template_kwargs reasoning_effort, for templates that take one instead of (or beside) a thinking switch:
    # Inkling 2's ("none", "minimal", "low", "medium", "high", "xhigh", "max"; 0.9 when unset). None leaves it out.
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None
    top_k: Optional[int] = None  # sent when set; vLLM otherwise applies the model's generation_config (Qwen3: 20)
    keep_texts: bool = False  # keep every reply's text and thinking in its Call (not a model setting)


def groups(answers: Sequence[Optional[str]]) -> List[List[str]]:
    """The answers grouped by equality as maths (problems.equivalent), each group in the order first seen; None does
    not count."""
    out: List[List[str]] = []
    for a in answers:
        if a is None or not a.strip():
            continue
        for g in out:
            if equivalent(g[0], a):
                g.append(a)
                break
        else:
            out.append([a])
    return out


def majority(answers: Sequence[Optional[str]]) -> Optional[str]:
    """The first-seen writing of the largest group of equal answers; ties go to the group seen first; None when no
    answer can vote."""
    found = groups(answers)
    return max(found, key=len)[0] if found else None


def split_thinking(message: Dict[str, Any], finish_reason: str = "stop",
                   thinking: Optional[bool] = None) -> Tuple[str, str]:
    """(thinking, answer text) of a reply's message: vLLM's `reasoning_content` when the server splits it, else the text
    around `</think>`. A reply cut off before its thinking closed has no answer text: one that opened <think> and never
    closed it, and one cut off at max_tokens with no tag and nothing split out while thinking was not turned off, since
    templates that write <think> into the prompt themselves (Qwen3.5, the Qwen3 Thinking models, R1 distills) leave
    no tag in the reply, and a provisional \\boxed{} in that thinking is no answer."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if "</think>" in content:
        before, _, after = content.rpartition("</think>")
        return reasoning + before.replace("<think>", ""), after.strip()
    if "<think>" in content:
        return reasoning + content.replace("<think>", ""), ""
    if finish_reason == "length" and not reasoning and thinking is not False:
        return content, ""
    return reasoning, content.strip()


def _body(settings: Settings, messages: List[Dict[str, str]], salt: int, priority: int) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": settings.model, "messages": messages, "n": 1, "max_tokens": settings.max_tokens,
                            "temperature": settings.temperature, "top_p": settings.top_p, PRIORITY: priority}
    kwargs: Dict[str, Any] = {}
    if settings.thinking is not None:
        kwargs["enable_thinking"] = settings.thinking
    if settings.reasoning_effort is not None:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    if settings.seed is not None:
        body["seed"] = settings.seed + salt
    if settings.top_k is not None:
        body["top_k"] = settings.top_k
    return body


async def _ask(post: Post, settings: Settings, messages: List[Dict[str, str]], salt: int = 0,
               priority: int = 0) -> Tuple[str, Call]:
    """(answer text, call): one answer to `messages`, its seed the run's plus `salt`."""
    t0 = time.perf_counter()
    reply = await post(_body(settings, messages, salt, priority))
    seconds = float(reply.get(SERVICE_SECONDS, time.perf_counter() - t0))
    usage = reply.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = (reply.get("choices") or [{}])[0]
    thinking, text = split_thinking(choice.get("message") or {}, str(choice.get("finish_reason")), settings.thinking)
    return text, Call(prompt_tokens=int(usage.get("prompt_tokens") or 0),
                      completion_tokens=int(usage.get("completion_tokens") or 0),
                      cached_tokens=(int(details["cached_tokens"]) if details.get("cached_tokens") is not None
                                     else None), seconds=seconds,
                      finish_reason=str(choice.get("finish_reason")), thinking_chars=len(thinking),
                      text=text if settings.keep_texts else None,
                      thinking=thinking if settings.keep_texts else None)


def _solve_messages(problem: Problem) -> List[Dict[str, str]]:
    return [{"role": "user", "content": SOLVE.format(problem=problem.problem)}]


async def _sample(post: Post, settings: Settings, problem: Problem, salts: Sequence[int],
                  priority: int = 0) -> Tuple[List[Optional[str]], List[Call]]:
    replies = await asyncio.gather(*(_ask(post, settings, _solve_messages(problem), s, priority) for s in salts))
    return [last_boxed(text) for text, _ in replies], [call for _, call in replies]


async def direct(post: Post, settings: Settings, problem: Problem) -> Attempt:
    text, call = await _ask(post, settings, _solve_messages(problem))
    answer = last_boxed(text)
    return Attempt("direct", problem.id, [answer], answer, calls=[call])


async def self_consistency(post: Post, settings: Settings, problem: Problem, k: int) -> Attempt:
    """Its majority is taken after the batch's clock stops (rigtune.reasoning.run.grade_attempt): comparing answers as
    maths is CPU work that would otherwise hold up the loop that refills the GPU."""
    answers, calls = await _sample(post, settings, problem, range(k))  # seeds: the run's + 0 .. k-1
    return Attempt(f"self-consistency-{k}", problem.id, answers, None, calls=calls)


async def self_correction(post: Post, settings: Settings, problem: Problem) -> Attempt:
    messages = _solve_messages(problem)
    first_text, first_call = await _ask(post, settings, messages)
    first = last_boxed(first_text)
    try:
        final_text, second_call = await _ask(post, settings, messages + [
            {"role": "assistant", "content": first_text}, {"role": "user", "content": CRITIQUE}], salt=1, priority=1)
    except Exception as exc:  # the critique turn failed (a long first answer can overflow the context): the method
        return Attempt("self-correction", problem.id, [first], None, first=first, calls=[first_call],  # gave no
                       error=f"critique turn: {type(exc).__name__}: {exc}"[:300])  # final answer: wrong, first kept
    final = last_boxed(final_text)
    return Attempt("self-correction", problem.id, [first, final], final, first=first, calls=[first_call, second_call])


async def adaptive(post: Post, settings: Settings, problem: Problem, k: int) -> Attempt:
    answers, calls = await _sample(post, settings, problem, (0, 1))
    agreed = len(groups(answers)) == 1 and all(a is not None for a in answers)
    if not agreed and k > 2:
        try:
            more, more_calls = await _sample(post, settings, problem, range(2, k), priority=1)  # seeds never reused
        except Exception as exc:  # it needed more answers and got none: no final answer, its first two kept
            return Attempt(f"adaptive-{k}", problem.id, answers, None, calls=calls,
                           error=f"more answers: {type(exc).__name__}: {exc}"[:300])
        answers, calls = answers + more, calls + more_calls
    # Only the two-answer check above decides what runs; the majority is taken after the clock (grade_attempt).
    return Attempt(f"adaptive-{k}", problem.id, answers, answers[0] if agreed else None, calls=calls)


def http_post(base_url: str, timeout: float = 3600.0) -> tuple:
    """(post, close): a Post that sends to `base_url`/v1/chat/completions with no cap of its own on connections (the
    run's sequence limit is the only one: httpx's default of 100 held every n = 1 method below it), and the coroutine
    that closes its client."""
    import httpx

    client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                               limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))

    async def post(body: Dict[str, Any]) -> Dict[str, Any]:
        body = {k: v for k, v in body.items() if not k.startswith("_rigtune")}  # rigtune's own, not the engine's
        r = await client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()

    return post, client.aclose
