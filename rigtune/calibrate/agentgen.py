"""Synthetic agent traces: explicit-request sessions shaped like a MemGPT agent (Packer et al. 2023, arXiv 2310.08560).

A model of MemGPT's published design, not a recording of any framework. Its prompt is fixed instructions, then a
writable working context (core memory), then a FIFO queue of messages whose head is a recursive summary of what was
evicted. The agent edits its core memory in place, searches an archive whose results come back as function results,
and chains a further call after a function returns (`request_heartbeat`) instead of waiting for the user. An edit
and an eviction rewrite the prompt, which turn-by-turn session replay cannot express; a search result and a
heartbeat only append to it (a search brings the next eviction sooner).

    lines = generate(AgentShape(), sessions=200, seed=0)      # then to_jsonl(lines), or benchmarks/make_agent_trace.py

Every session draws its whole event stream (user messages, replies, edits, searches, heartbeats) from
random.Random(seed * 100_003 + i) before anything is rendered, so `layout="memgpt"` and `layout="append"` render the
same events: the second is the paired counterfactual, with every edit appended to the queue as a function result
and the first message never changed, so the two differ only in where the edits land. Both evict at the same hard
limit, the oldest `evict_frac` of the queue's tokens replaced by a summary; a summary is a fixed selection of the
words it replaces, so rendering draws nothing at random.

Placement follows the paper's order: the working context sits in the first (system) message after the instructions,
since several chat templates refuse a second system message; the summary is a user message at the head of the queue,
and a function result is a user message, so roles alternate for templates that require it. `retrieval_at="head"`
puts the latest search result in the first message instead, as frameworks that retrieve into the system prompt do,
and `meta_refresh` adds a status line to the first message that changes every request (a hypothesis about agent
frameworks, off by default). Both apply to the memgpt layout only; the append layout keeps its first message fixed.

Token counts are estimates (TOKENS_PER_WORD per word plus MESSAGE_TOKENS per message), the unit of every *_tokens
parameter and of `window_tokens`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

from rigtune.calibrate.workload import _WORDS

TOKENS_PER_WORD = 1.25  # short English words on Llama and Qwen tokenizers (as workload.py sizes synthetic prompts)
MESSAGE_TOKENS = 4  # what a chat template adds around each message
CORE_LINES = 10  # the working context is this many lines; an edit replaces one
CHAIN_CAP = 4  # heartbeats chained after one user message, at most
EVICT_FRAC_SOURCE = ("the default evict_frac 0.3 is Letta's documented sliding_window_percentage default, not a "
                     "figure from the MemGPT paper")
LAYOUTS = ("memgpt", "append")
RETRIEVAL_AT = ("tail", "head")


@dataclass(frozen=True)
class AgentShape:
    """What one agent session looks like. Lengths are in estimated tokens (see the module docstring)."""

    system_tokens: int = 1000  # fixed instructions, the same in every session
    core_tokens: int = 500  # the working context, CORE_LINES lines of it
    window_tokens: int = 6000  # no request is longer: past it, the queue's oldest messages are evicted
    user_messages: int = 12
    user_tokens: int = 60
    reply_tokens: int = 80  # each assistant message: its reply, before any function call it makes
    heartbeat_p: float = 0.5  # after each step, the chance it chains another (at most CHAIN_CAP per user message)
    core_edit_p: float = 0.15  # each step's chance of replacing a line of the working context
    retrieval_p: float = 0.2  # each step's chance of searching the archive
    retrieval_tokens: int = 600
    retrieval_at: str = "tail"  # tail: the result is a function result (MemGPT); head: it goes in the first message
    # Share of the queue's tokens evicted at once. 0.3 is Letta's documented default (sliding_window_percentage), not
    # a figure from the MemGPT paper, which this model does not take one from (EVICT_FRAC_SOURCE, in the "gen" key).
    evict_frac: float = 0.3
    summary_tokens: int = 200
    meta_refresh: bool = False
    layout: str = "memgpt"  # memgpt | append
    max_tokens: int = 160  # each request's answer cap (0: the workload's)

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
        if self.retrieval_at not in RETRIEVAL_AT:
            raise ValueError(f"retrieval_at must be one of {', '.join(RETRIEVAL_AT)}")
        for name in ("heartbeat_p", "core_edit_p", "retrieval_p", "evict_frac"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if min(self.system_tokens, self.core_tokens, self.user_tokens, self.reply_tokens, self.retrieval_tokens,
               self.summary_tokens, self.user_messages, self.window_tokens) < 1 or self.max_tokens < 0:
            raise ValueError("token counts and user_messages must be positive")

    def changed(self) -> Dict[str, object]:
        """The parameters that differ from the defaults, in field order: what a file name needs to say."""
        default = AgentShape()
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(default, f.name)}


def approx_tokens(text: str) -> int:
    return math.ceil(TOKENS_PER_WORD * len(text.split()))


def request_tokens(messages: List[Dict[str, str]]) -> int:
    """A request's estimated length, as window_tokens limits it."""
    return sum(approx_tokens(m["content"]) + MESSAGE_TOKENS for m in messages)


def _text(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(max(1, int(tokens / TOKENS_PER_WORD))))


@dataclass(frozen=True)
class _Step:
    """One assistant step: its reply, and what it does next."""

    reply: str
    edit: Optional[Tuple[int, str]]  # (line of the working context, its new text)
    retrieval: Optional[str]  # what the archive search returned
    heartbeat: bool


def _events(shape: AgentShape, rng: random.Random) -> Tuple[List[str], List[Tuple[str, List[_Step]]]]:
    """The session's working context, and per user message its text and the steps that answer it. Every draw happens
    here, in a fixed order whatever the outcomes, so a layout cannot shift a later event."""
    line_tokens = max(1, shape.core_tokens // CORE_LINES)
    core = [_text(rng, line_tokens) for _ in range(CORE_LINES)]
    turns = []
    for _ in range(shape.user_messages):
        user = _text(rng, shape.user_tokens)
        steps = []
        for k in range(CHAIN_CAP + 1):
            reply = _text(rng, shape.reply_tokens)
            edit_line, edit_text = rng.randrange(CORE_LINES), _text(rng, line_tokens)
            found = _text(rng, shape.retrieval_tokens)
            draws = (rng.random(), rng.random(), rng.random())
            step = _Step(reply=reply, edit=(edit_line, edit_text) if draws[0] < shape.core_edit_p else None,
                         retrieval=found if draws[1] < shape.retrieval_p else None,
                         heartbeat=k < CHAIN_CAP and draws[2] < shape.heartbeat_p)
            steps.append(step)
            if not step.heartbeat:
                break
        turns.append((user, steps))
    return core, turns


SUMMARY = "Summary of earlier messages: "


def _summary(old: str, evicted: List[Tuple[str, str]], tokens: int) -> str:
    """A recursive summary's text: a fixed, evenly spaced selection of the words of the last one and of what was
    evicted."""
    words = (old + " " + " ".join(c for _, c in evicted)).split()
    n = max(1, int(tokens / TOKENS_PER_WORD))
    if len(words) > n:
        words = [words[k * len(words) // n] for k in range(n)]
    return " ".join(words)


class _Agent:
    """Renders one session's events as its requests, in one layout."""

    def __init__(self, shape: AgentShape, instructions: str, core: List[str]):
        self.shape = shape
        self.instructions = instructions
        self.core = list(core)
        self.retrieved = ""  # retrieval_at="head": the latest result, in the first message
        self.queue: List[Tuple[str, str]] = []
        self.summary = ""
        self.evicted = 0
        self.sent = 0
        self.rewrites = shape.layout == "memgpt"

    def first_message(self) -> str:
        text = self.instructions + "\n\nCore memory:\n" + "\n".join(self.core)
        if self.rewrites and self.retrieved:
            text += "\n\nRetrieved from archival memory:\n" + self.retrieved
        if self.rewrites and self.shape.meta_refresh:
            text += (f"\n\nMemory status: {self.evicted} earlier messages in recall storage; "
                     f"request {self.sent + 1} of this session.")
        return text

    def messages(self) -> List[Dict[str, str]]:
        head = [("system", self.first_message())] + ([("user", SUMMARY + self.summary)] if self.summary else [])
        return [{"role": r, "content": c} for r, c in head + self.queue]

    def fit(self) -> bool:
        """Evict from the queue's head until the request fits the window; whether anything was evicted. The first
        kept message is an assistant's, after the summary (a user message), so roles still alternate."""
        evicted = False
        while request_tokens(self.messages()) > self.shape.window_tokens:
            cuts = [i for i in range(1, len(self.queue)) if self.queue[i][0] == "assistant"]
            if not cuts:
                raise ValueError(f"window_tokens {self.shape.window_tokens} cannot hold the instructions, the "
                                 "working context and the latest message")
            target = self.shape.evict_frac * sum(approx_tokens(c) for _, c in self.queue)
            cut = next((i for i in cuts if sum(approx_tokens(c) for _, c in self.queue[:i]) >= target), cuts[-1])
            gone, self.queue = self.queue[:cut], self.queue[cut:]
            self.summary = _summary(self.summary, gone, self.shape.summary_tokens)
            self.evicted += len(gone)
            evicted = True
        return evicted

    def send(self, out: List[Dict[str, object]], tag: str, chain: bool) -> None:
        tag = "evict" if self.fit() else tag
        req: Dict[str, object] = {"messages": self.messages(), "tag": tag}
        if chain:
            req["chain"] = True
        if self.shape.max_tokens:
            req["max_tokens"] = self.shape.max_tokens
        out.append(req)
        self.sent += 1

    def act(self, step: _Step) -> Tuple[str, str]:
        """Take in one step's reply and what it did: the function result it produces ("" when it called nothing) and
        the tag the next request carries for it."""
        calls, parts, tag = [], [], "heartbeat"
        if step.edit is not None:
            line, text = step.edit
            calls.append(f"core_memory_replace(line={line}, new={text!r})")
            if self.rewrites:
                self.core[line] = text
                parts.append("core memory updated")
            else:
                parts.append(f"core memory line {line} is now: {text}")
            tag = "core_edit"
        if step.retrieval is not None:
            calls.append("archival_memory_search(query=...)")
            if self.rewrites and self.shape.retrieval_at == "head":
                self.retrieved = step.retrieval
                parts.append("search results added to the system prompt")
            else:
                parts.append("archival memory search returned: " + step.retrieval)
            tag = "retrieval" if tag == "heartbeat" else tag
        if step.heartbeat:
            calls.append("request_heartbeat=true")
        self.queue.append(("assistant", "\n".join([step.reply] + calls)))
        if not (calls or parts):
            return "", "user"
        return "[function result] " + ("; ".join(parts) or "ok"), tag


def generate(shape: AgentShape, sessions: int, seed: int) -> List[Dict[str, object]]:
    """`sessions` explicit-request session lines (datasets.read_sessions), each with its parameters in a "gen" key:
    replay ignores all of it but `chars_per_token`, the session's characters per estimated token, which sizes the
    context the planner keeps (the generator's words run about 6 characters per token, not the 4 a file is assumed
    to run at)."""
    instructions = "You are a stateful agent with tools. " + _text(random.Random(f"agentgen-instructions-{seed}"),
                                                                    shape.system_tokens)
    gen = asdict(shape)
    lines = []
    for i in range(sessions):
        rng = random.Random(seed * 100_003 + i)
        core, turns = _events(shape, rng)
        agent = _Agent(shape, instructions, core)
        out: List[Dict[str, object]] = []
        pending, pending_tag = "", "first"
        for user, steps in turns:
            agent.queue.append(("user", pending + user))
            tag, chain = pending_tag, False
            pending, pending_tag = "", "user"
            for step in steps:
                agent.send(out, tag, chain)
                result, next_tag = agent.act(step)
                if step.heartbeat:
                    agent.queue.append(("user", result))
                    tag, chain = next_tag, True
                elif result:  # delivered with the next user message, which then carries its tag
                    pending, pending_tag = result + "\n\n", next_tag
        contents = [m["content"] for req in out for m in req["messages"]]
        cpt = round(sum(map(len, contents)) / max(1, sum(map(approx_tokens, contents))), 3)
        lines.append({"session": f"agent{seed}-{i}", "requests": out,
                      "gen": {**gen, "seed": seed, "index": i, "chars_per_token": cpt,
                              "evict_frac_source": EVICT_FRAC_SOURCE}})
    return lines


def to_jsonl(lines: List[Dict[str, object]]) -> str:
    """The lines as the file's text: sorted keys and ASCII only, so the bytes are the same on every platform (write
    them with newline="\\n")."""
    return "".join(json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n" for line in lines)


def default_name(shape: AgentShape, sessions: int, seed: int) -> str:
    """A file name that says what generated it: the layout, the count, the seed and every non-default parameter."""
    extra = "".join(f"-{k.replace('_', '')}{v}" for k, v in shape.changed().items() if k != "layout")
    return f"agent-{shape.layout}-n{sessions}-s{seed}{extra}.jsonl"
