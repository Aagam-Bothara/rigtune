"""Real prompts for the dataset-backed workloads.

Random-word prompts measure throughput well, but not strategies that depend on what the text says:
n-gram speculative decoding only pays when an answer repeats its input, and how long an answer runs
depends on the question. These sources supply real text:

  sharegpt        first user turns of real ChatGPT conversations (ShareGPT, as vLLM's benchmarks use)
  cnn-extract     news articles, with an instruction to quote sentences word for word (answers copy input)
  humaneval-edit  Python functions, with an instruction to add type hints (answers copy the code)

Each source is downloaded from the Hugging Face Hub on first use and cached there.
"""

from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

EXTRACT_INSTRUCTION = ("\n\nCopy, word for word, every sentence in the article above that contains a number. "
                       "Output only those sentences, one per line.")
EDIT_INSTRUCTION = ("\n\nRewrite the Python function above with type hints on every parameter and on the return "
                    "value, and a one-line docstring. Keep every other line exactly as it is. Output only the code.")
CHARS_PER_TOKEN = 4  # rough English average, for a first cut before a tokenizer trims precisely


def _sharegpt() -> List[Tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for conv in data:
        turns = conv.get("conversations") or []
        if turns and turns[0].get("from") == "human" and len((turns[0].get("value") or "").strip()) >= 20:
            out.append((turns[0]["value"].strip(), ""))
    return out


def _cnn_extract() -> List[Tuple[str, str]]:
    from datasets import load_dataset

    return [(row["article"].strip(), EXTRACT_INSTRUCTION)
            for row in load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")]


def _humaneval_edit() -> List[Tuple[str, str]]:
    from datasets import load_dataset

    return [((row["prompt"] + row["canonical_solution"]).rstrip(), EDIT_INSTRUCTION)
            for row in load_dataset("openai/openai_humaneval", split="test")]


# source -> loader of (body, instruction) pairs; the instruction survives any trimming of the body.
LOADERS: Dict[str, Callable[[], List[Tuple[str, str]]]] = {
    "sharegpt": _sharegpt, "cnn-extract": _cnn_extract, "humaneval-edit": _humaneval_edit,
}


FILE_PREFIX = "file:"  # a source naming your own prompt file: "file:/path/to/prompts.jsonl"


def read_prompt_file(path) -> List[str]:
    """Your own prompts, from a JSONL file with one per line: a JSON string, {"prompt": ...}, {"text": ...}
    or {"messages": [{"role": ..., "content": ...}, ...]}. A conversation's messages are joined in order with
    blank lines; no chat template is applied, as with the built-in sources, which send raw text too. A file
    whose lines name a session is read by read_sessions instead."""
    out: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
            text = _prompt_text(item)
            if not text:
                raise ValueError(f'{path}:{n}: expected a string, or an object with "prompt", "text" or "messages"')
            out.append(text)
    if not out:
        raise ValueError(f"{path}: no prompts")
    return out


def _prompt_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("prompt", "text"):
            if isinstance(item.get(key), str):
                return item[key].strip()
        messages = item.get("messages")
        if isinstance(messages, list):
            return "\n\n".join(str(m["content"]).strip() for m in messages
                               if isinstance(m, dict) and m.get("content")).strip()
    return ""


# --------------------------------------------------------------------------- sessions

DEFAULT_TURNS_CAP = 8  # requests replayed per session; later ones are dropped


@dataclass(frozen=True)
class Step:
    """One request of a session. By default it appends a user message to the conversation so far; a step with
    `messages` sends exactly those messages instead, for traces whose requests are not one growing conversation
    (an agent's, read from "requests" in a file: see read_sessions). `answer` is the reply the file recorded, when it
    has one.

    For agent traces: `chain` makes the next step follow this one at once, ahead of every other session, as a tool
    call's follow-up would; `tag` labels the step in its outcome; `max_tokens` caps its answer in place of the
    workload's."""

    user: str = ""
    messages: Optional[Tuple[Tuple[str, str], ...]] = None  # (role, content) pairs
    answer: Optional[str] = None
    chain: bool = False
    tag: str = ""
    max_tokens: Optional[int] = None

    @property
    def explicit(self) -> bool:
        return self.messages is not None


@dataclass(frozen=True)
class Session:
    """One conversation to replay request by request: its system prompt and its steps in order."""

    id: str
    system: str
    steps: Tuple[Step, ...]

    @property
    def turns(self) -> int:
        return len(self.steps)

    @property
    def users(self) -> Tuple[str, ...]:
        return tuple(s.user for s in self.steps)

    @property
    def answers(self) -> Tuple[Optional[str], ...]:
        """The recorded replies between the steps (one fewer than the steps: nothing follows the last)."""
        return tuple(s.answer for s in self.steps[:-1])

    @property
    def first_turn(self) -> str:
        """The opening request as plain text, for whatever needs one prompt per session (the quality probe). An
        explicit request already holds its system prompt."""
        first = self.steps[0]
        if first.explicit:
            return "\n\n".join(c for _, c in first.messages if c)
        return "\n\n".join(x for x in (self.system, first.user) if x)

    def capped(self, turns: int) -> "Session":
        if self.turns <= turns:
            return self
        steps = self.steps[:turns]
        return replace(self, steps=steps[:-1] + (replace(steps[-1], answer=None),))


def _is_session_line(item: object) -> bool:
    """A line that names a session and holds a conversation (messages) or one turn of one (role). A "session" key
    beside "prompt" or "text" was ignored before session files existed, and still is (see ignored_session_keys)."""
    return isinstance(item, dict) and "session" in item and (isinstance(item.get("messages"), list) or "role" in item)


def _ignores_session_key(item: object) -> bool:
    return isinstance(item, dict) and "session" in item and not _is_session_line(item)


def _is_explicit_line(item: object) -> bool:
    """A line giving requests whole: a non-empty "requests" list of objects with "messages" (a session's every
    request), or a whole-number "request" beside "session" and "messages" (one request). Nothing else makes a file
    one of explicit requests: a "request" or "requests" key of any other shape is metadata, ignored as it always was
    (a {"session", "request": "req-7", "messages"} line is a conversation, as before), and so is a "requests" key
    beside "prompt" or "text"."""
    return _is_request_line(item) or _is_requests_line(item)


def _is_request_line(item: object) -> bool:
    number = item.get("request") if isinstance(item, dict) else None
    return (isinstance(number, int) and not isinstance(number, bool) and "session" in item
            and isinstance(item.get("messages"), list))


def _is_requests_line(item: object) -> bool:
    requests = item.get("requests") if isinstance(item, dict) else None
    return (isinstance(requests, list) and bool(requests)
            and all(isinstance(r, dict) and isinstance(r.get("messages"), list) for r in requests)
            and not any(isinstance(item.get(k), str) for k in ("prompt", "text")))


EXPLICIT_ROLES = ("system", "user", "assistant")


def _explicit_message(m: object, where: str) -> Tuple[str, str]:
    """One message of an explicit request, kept byte for byte: no stripping, since an agent's next request is compared
    with this one character by character."""
    if not isinstance(m, dict) or set(m) - {"role", "content"} or m.get("role") not in EXPLICIT_ROLES \
            or not isinstance(m.get("content"), str):
        raise ValueError(f"{where}: each message needs a \"role\" ({', '.join(EXPLICIT_ROLES)}) and a string "
                         "\"content\", and nothing else; write a tool call as assistant text and its result as a user "
                         f"message (got {json.dumps(m)[:80]})")
    return m["role"], m["content"]


def _explicit_session(sid: str, requests: List[Tuple[object, str]]) -> Session:
    """A session from its requests in order, each with the place an error is reported against. `"chain": true` on a
    request means it was sent the moment the one before it finished, so it is that earlier step that chains."""
    steps: List[Step] = []
    for k, (req, where) in enumerate(requests):
        if not isinstance(req, dict) or not isinstance(req.get("messages"), list) or not req["messages"]:
            raise ValueError(f"{where}: session {sid!r}: a request needs a non-empty \"messages\" list")
        chain, tag, cap = req.get("chain", False), req.get("tag", ""), req.get("max_tokens")
        if not isinstance(chain, bool) or not isinstance(tag, str):
            raise ValueError(f"{where}: session {sid!r}: \"chain\" must be true or false and \"tag\" a string")
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{where}: session {sid!r}: \"max_tokens\" must be a positive whole number")
        if chain and not steps:
            raise ValueError(f"{where}: session {sid!r} opens with a chained request, but nothing comes before it")
        if chain:
            steps[-1] = replace(steps[-1], chain=True)
        steps.append(Step(messages=tuple(_explicit_message(m, where) for m in req["messages"]), tag=tag,
                          max_tokens=cap))
    role, text = steps[0].messages[0]
    return Session(id=sid, system=text if role == "system" else "", steps=tuple(steps))


def _explicit_sessions(path, items: List[Tuple[int, object]]) -> List[Session]:
    """The sessions of a file of explicit requests (see read_sessions), in the order each first appears."""
    first = next(n for n, item in items if _is_explicit_line(item))
    order: List[str] = []
    whole: Dict[str, Session] = {}
    lines: Dict[str, List[Tuple[int, dict]]] = {}
    for n, item in items:
        where = f"{path}:{n}"
        if not _is_explicit_line(item):
            raise ValueError(f"{where}: line {first} gives requests whole (\"requests\" or \"request\"), so every line "
                             "must; a file cannot mix them with turns")
        if "system" in item:
            raise ValueError(f"{where}: a \"system\" key does not apply to explicit requests; each request's "
                             "messages hold its system prompt")
        if _is_request_line(item) and _is_requests_line(item):
            raise ValueError(f"{where}: give \"request\" with its \"messages\", or \"requests\", not both")
        if _is_request_line(item):
            if item["session"] is None:
                raise ValueError(f"{where}: a line holding one request needs a \"session\"")
            sid = str(item["session"])
            if sid in whole:
                raise ValueError(f"{where}: session {sid!r} was already given whole on an earlier line")
            if sid not in lines:
                order.append(sid)
            lines.setdefault(sid, []).append((n, item))
            continue
        sid = str(item["session"]) if item.get("session") is not None else f"line{n}"
        if sid in whole or sid in lines:
            raise ValueError(f"{where}: session {sid!r} appears twice")
        if "messages" in item:
            raise ValueError(f"{where}: give \"requests\" or \"messages\", not both")
        whole[sid] = _explicit_session(sid, [(r, f"{where}: request {i}") for i, r in enumerate(item["requests"], 1)])
        order.append(sid)
    out: List[Session] = []
    for sid in order:
        if sid in whole:
            out.append(whole[sid])
            continue
        numbered = lines[sid]
        for (_, a), (n, b) in zip(numbered, numbered[1:]):
            x, y = a["request"], b["request"]  # whole numbers, or the line would not be one request
            if y <= x:
                raise ValueError(f"{path}:{n}: session {sid!r} request {y!r} does not follow request {x!r}")
        out.append(_explicit_session(sid, [(it, f"{path}:{n}") for n, it in numbered]))
    return out


def is_multi_turn(item: object) -> bool:
    """A {"messages": [...]} line holding an answer or a second user turn: a conversation a flat file flattens."""
    messages = item.get("messages") if isinstance(item, dict) else None
    if not isinstance(messages, list):
        return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return "assistant" in roles or roles.count("user") > 1


def _content(m: dict) -> str:
    return str(m.get("content") or "").strip()


def _session(sid: str, system: str, messages: List[Tuple[str, str, str]]) -> Session:
    """A session from (role, content, where) triples; `where` names the line an error is reported against."""
    systems = [system] if system else []
    rest = []
    for role, text, where in messages:
        if role == "system" and not rest:
            if text:
                systems.append(text)
        else:
            rest.append((role, text, where))
    while rest and rest[-1][0] == "assistant":  # nothing follows a trailing answer, so it is never sent
        rest.pop()
    if not rest:
        raise ValueError(f"{messages[0][2] if messages else sid}: session {sid!r} has no user turn")
    steps: List[Step] = []
    for i, (role, text, where) in enumerate(rest):
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"{where}: session {sid!r} has {role!r} where {expected!r} was expected; "
                             "turns must alternate user, assistant, user, ...")
        if role == "user":
            if not text:
                raise ValueError(f"{where}: session {sid!r} has an empty user turn")
            steps.append(Step(user=text))
        else:
            steps[-1] = replace(steps[-1], answer=text)
    return Session(id=sid, system="\n\n".join(systems), steps=tuple(steps))


def read_sessions(path, force: bool = False) -> List[Session]:
    """The sessions in a JSONL prompt file, or [] when it is a flat file (see read_prompt_file).

    A file is a session file when some line names a "session" and holds messages or one turn (a "role"), or when
    `force` (--sessions) says so. Multi-turn {"messages": [...]} lines alone do not make one, nor does a "session"
    key beside "prompt" or "text": such files were read as flat before sessions existed, and their cached profiles
    must keep meaning what they measured. The file may start with a byte-order mark. Two forms, which may be mixed:

      {"session": "s1", "system": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}
      {"session": "s2", "turn": 1, "role": "user", "content": "..."}     one turn per line, grouped by session

    In a session file a line without "session" is a session of its own, and a plain prompt a one-turn session. A
    system prompt comes from a "system" key or leading system messages; turns must alternate user and assistant,
    and a trailing answer is dropped. Sessions keep the order they first appear in, and the turns of a
    one-turn-per-line session keep file order ("turn", when given, must increase).

    A file of explicit requests gives each request's whole message list instead, for an agent that rewrites its
    prompt rather than appending to it; every line must then be one of these two forms:

      {"session": "a1", "requests": [{"messages": [...]}, {"messages": [...], "chain": true, "tag": "heartbeat"}]}
      {"session": "a2", "request": 1, "messages": [...]}               one request per line ("request" must increase)

    "requests" must be a non-empty list of objects with "messages", and "request" a whole number; a key of either name
    in any other shape is metadata and leaves the line as it was read before. Messages are kept byte for byte. `chain` marks a request sent as soon as the one before it finished; `tag` and
    `max_tokens` are optional."""
    items = _json_lines(path)
    if any(_is_explicit_line(item) for _, item in items):
        return _explicit_sessions(path, items)
    if not force and not any(_is_session_line(item) for _, item in items):
        return []
    order: List[str] = []
    whole: Dict[str, Session] = {}
    turns: Dict[str, List[Tuple[int, dict]]] = {}
    for n, item in items:
        where = f"{path}:{n}"
        if isinstance(item, dict) and "role" in item and "messages" not in item:
            sid = item.get("session")
            if sid is None:
                raise ValueError(f'{where}: a line holding one turn needs a "session" key')
            sid = str(sid)
            if sid in whole:
                raise ValueError(f"{where}: session {sid!r} was already given whole on an earlier line")
            if sid not in turns:
                order.append(sid)
            turns.setdefault(sid, []).append((n, item))
            continue
        sid = str(item.get("session")) if isinstance(item, dict) and item.get("session") is not None else f"line{n}"
        if sid in whole or sid in turns:
            raise ValueError(f"{where}: session {sid!r} appears twice")
        if isinstance(item, dict) and isinstance(item.get("messages"), list):
            messages = [(str(m.get("role")), _content(m), where) for m in item["messages"] if isinstance(m, dict)]
            whole[sid] = _session(sid, str(item.get("system") or "").strip(), messages)
        else:
            text = _prompt_text(item)
            if not text:
                raise ValueError(f'{where}: expected a string, or an object with "prompt", "text" or "messages"')
            whole[sid] = Session(id=sid, system="", steps=(Step(user=text),))
        order.append(sid)
    out: List[Session] = []
    for sid in order:
        if sid in whole:
            out.append(whole[sid])
            continue
        lines = turns[sid]
        numbered = [(n, it["turn"]) for n, it in lines if "turn" in it]
        for (_, a), (n, b) in zip(numbered, numbered[1:]):
            if not (isinstance(a, int) and isinstance(b, int) and b > a):
                raise ValueError(f"{path}:{n}: session {sid!r} turn {b!r} does not follow turn {a!r}")
        system = next((str(it["system"]).strip() for _, it in lines if it.get("system")), "")
        out.append(_session(sid, system, [(str(it.get("role")), _content(it), f"{path}:{n}") for n, it in lines]))
    return out


def _json_lines(path) -> List[Tuple[int, object]]:
    items: List[Tuple[int, object]] = []
    with open(path, encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                items.append((n, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
    return items


def flat_notes(path) -> Tuple[int, int]:
    """For a flat file: how many lines are conversations that --sessions would replay rather than flatten, and how
    many carry a "session" key that is ignored because they hold a prompt rather than messages."""
    items = [item for _, item in _json_lines(path)]
    return sum(is_multi_turn(i) for i in items), sum(_ignores_session_key(i) for i in items)


@functools.lru_cache(maxsize=None)
def session_pool(path: str, force: bool = False) -> Tuple[Session, ...]:
    """Every session of a file, in one fixed shuffled order, as `pool` orders prompts."""
    items = read_sessions(path, force)
    if not items:
        raise ValueError(f"{path}: no sessions")
    random.Random(0).shuffle(items)
    return tuple(items)


def sample_sessions(source: str, n: int, offset: int, turns_cap: int = DEFAULT_TURNS_CAP,
                    system: Optional[str] = None, force: bool = False) -> List[Session]:
    """n sessions from position `offset` of the shuffled pool (wrapping around a small one), each cut to its first
    `turns_cap` steps. `system`, when given, is every session's system prompt in place of its own."""
    if not source.startswith(FILE_PREFIX):
        raise ValueError(f"sessions come from a file, not {source!r}")
    items = session_pool(source[len(FILE_PREFIX):], force)
    out = []
    for j in range(n):
        s = items[(offset + j) % len(items)].capped(turns_cap)
        out.append(replace(s, system=system) if system else s)
    return out


@functools.lru_cache(maxsize=None)
def pool(source: str) -> Tuple[Tuple[str, str], ...]:
    """Every usable prompt of a source, in one fixed shuffled order."""
    if source.startswith(FILE_PREFIX):
        items = [(p, "") for p in read_prompt_file(source[len(FILE_PREFIX):])]
    elif source not in LOADERS:
        raise ValueError(f"unknown prompt source {source!r}; choose from {', '.join(LOADERS)} or file:<path>")
    else:
        items = list(LOADERS[source]())
    if not items:
        raise RuntimeError(f"prompt source {source!r} returned nothing")
    random.Random(0).shuffle(items)
    return tuple(items)


def sample(source: str, n: int, offset: int, max_chars: int) -> List[str]:
    """n prompts from position `offset` of the shuffled pool (wrapping around a small one), each body
    cut to fit max_chars with its instruction, at a line break when there is one nearby."""
    items = pool(source)
    out = []
    for j in range(n):
        body, instruction = items[(offset + j) % len(items)]
        room = max(64, max_chars - len(instruction))
        if len(body) > room:
            cut = body.rfind("\n", 0, room)
            body = body[:cut] if cut > room // 2 else body[:room]
        out.append(body + instruction)
    return out
