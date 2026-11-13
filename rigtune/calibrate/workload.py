"""Synthetic workloads. Every trial in a calibration uses the same one so results are comparable.

Presets (`rigtune serve <model> --workload NAME`):

| name             | prefill | decode | concurrency   | TTFT ceiling | shaped like                     |
|------------------|---------|--------|---------------|--------------|---------------------------------|
| default          |   256   |  128   | 1 / 4 / 8     |   500 ms     | the spec's calibration workload |
| chat             |   512   |  128   | 1 / 4 / 8     |   500 ms     | assistant turns                 |
| long-context     |  8192   |  256   | 1 / 2 / 4     |  2000 ms     | document Q&A, summarisation     |
| generation       |   128   | 1024   | 1 / 4 / 8     |   500 ms     | code / story generation         |
| high-concurrency |   256   |   64   | 32 / 64 / 128 |  1000 ms     | many short requests             |
| rag              |  6144   |   64   | 1 / 4 / 8     |  1500 ms     | retrieval-augmented answers     |
| sharegpt         | ≤1024   | ≤512   | 1 / 8 / 32    |  1000 ms     | real ChatGPT user turns         |
| extract          | ≤1536   | ≤384   | 1 / 4 / 8     |  1500 ms     | quote sentences from an article |
| code-edit        |  ≤768   | ≤512   | 1 / 4 / 8     |   500 ms     | add type hints to a function    |

The last three use real text (rigtune.calibrate.datasets) and let answers stop when the model
does; their prefill and decode figures are caps. `--workload-file` builds the same kind of workload
from your own prompts (workload_from_file), shaped like a preset.

When a tokenizer is available, `fit_prompts` resizes each prompt to the target token count
so "512-token prefill" means 512 tokens for that model, not ~512.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rigtune.calibrate.datasets import (CHARS_PER_TOKEN, DEFAULT_TURNS_CAP, EXPLICIT_WINDOW, FILE_PREFIX, Session,
                                        window)
from rigtune.calibrate.tokens import TokenCounter

logger = logging.getLogger(__name__)

_WORDS = (
    "system latency throughput memory kernel batch token decode prefill schedule cache page block "
    "tensor matrix vector attention layer head hidden context window request response server client "
    "measure sample power energy joule watt clock frequency thermal budget margin planner search stage "
    "quantize precision weight bias gradient optimizer inference runtime backend adapter profile hash "
    "network socket stream chunk parse encode dictionary index offset stride buffer queue worker thread"
).split()

_PREFIX = "Prompt {i}: "
_SUFFIX = "\n\nContinue the text:"
CTX_HEADROOM = 64  # tokens of slack for chat templates / special tokens
MIN_LEVEL_REQUESTS = 8  # fewest requests a level sends when requests_per_slot caps it
MAX_FILE_PREFILL = 32768  # a longer prompt in a --workload-file is cut to this many tokens, keeping its start
# Session files. A level replays a budget of turns rather than a number of sessions: TURNS_PER_SLOT per concurrency
# slot, from at least twice as many sessions' worth of turns, so the level stays closed-loop at its concurrency
# until the budget is spent instead of thinning out as its last sessions end.
TURNS_PER_SLOT = 6
TURNS_PER_POOLED_SESSION = 3  # a pool's live sessions each get about this many turns before the budget ends
POOL_CAP = 8  # explicit requests: sessions live per slot at most, whatever --session-pool asks (Workload.pool_at)
MIN_LEVEL_SESSIONS = 2
TURN_OVERHEAD = 16  # tokens a chat template adds per message, budgeted when sizing a session's longest prompt
HISTORIES = ("generated", "recorded")


def _pct(values: Sequence[int], pct: float) -> int:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return int(ordered[idx])


@dataclass
class Workload:
    n_prompts: int = 16
    prefill_tokens: int = 256
    decode_tokens: int = 128
    concurrencies: Tuple[int, ...] = (1, 4, 8)
    seed: int = 0
    temperature: float = 0.0
    name: str = "default"
    ttft_ceiling_ms: float = 500.0  # default constraint for the `balanced` objective
    # Per-token decode latency ceiling, applied under every objective. Prefill latency is what
    # TTFT constrains; this constrains decode, so neither phase can be sacrificed for the other.
    tpot_ceiling_ms: Optional[float] = 100.0
    # Tokens every prompt starts with (a system prompt, a retrieved document). This is what prefix
    # caching exploits: only the first request should pay to prefill them.
    shared_prefix_tokens: int = 0
    prefix_text: str = ""  # the shared prefix itself; generated from `seed` unless given
    prefix_fixed: bool = False  # the prefix was given (e.g. a warmup reusing it) and must not change
    prompts: List[str] = field(default_factory=list)
    fitted: bool = False  # prompts were sized with a real tokenizer
    # Token count of each entry in `prompts`, once a tokenizer has counted them. Held beside the prompts
    # rather than summarised, so a copy whose prompts were replaced or sliced is caught by the length
    # check in `prompt_token_stats` instead of reporting its parent's spread.
    token_counts: List[int] = field(default_factory=list)
    # Real text instead of random words (see rigtune.calibrate.datasets). prefill_tokens is then a
    # cap, and with natural_stop answers end when the model stops rather than at decode_tokens.
    source: str = "synthetic"
    natural_stop: bool = False
    sample_offset: int = 0  # where this workload's prompts start in the source's shuffled pool
    # Requests per concurrency slot at each level (0 = every level sends all n_prompts). Real answers
    # run to hundreds of tokens, so sending all 64 ShareGPT prompts one at a time took 5 minutes of
    # an 8-minute trial on an A40; a level at concurrency c needs a few rounds of c, not all of them.
    requests_per_slot: int = 0
    # Session replay, for a --workload-file of conversations (datasets.read_sessions). n_prompts then counts the
    # sessions a level may draw on, `session_list` holds them, and `prompts` holds each one's first turn as plain
    # text for whatever needs one prompt per session (the quality probe, prompt-length counts).
    sessions: bool = False
    force_sessions: bool = False  # --sessions: every line is a session, whether or not it names one
    turns_cap: int = DEFAULT_TURNS_CAP
    # What a later turn carries as the earlier answers: "generated" | "recorded", or "explicit" for a file that gives
    # every request's messages whole.
    history: str = "generated"
    session_pool: int = 0  # sessions live at once per level (0: one per concurrency slot)
    turns_per_slot: int = TURNS_PER_SLOT
    level_turns: int = 0  # a fixed turn budget in place of turn_budget's: one load-generator process's share
    session_system: str = ""  # every session's system prompt in place of its own, when given
    session_list: List[Session] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.shared_prefix_tokens and not self.prefix_text:
            self.prefix_text = self._prefix(max(4, int(self.shared_prefix_tokens * 0.8)))
        if not self.prompts and self.source == "synthetic":  # real text downloads: only when needed
            self.prompts = self._generate()
            self.token_counts = []  # a copy's regenerated prompts are not the ones that were counted

    def level_requests(self, concurrency: int) -> int:
        """How many of the prompts a level at this concurrency sends (for a session file: how many sessions it may
        draw on, see sessions_for)."""
        if self.sessions:
            turns = (sum(s.turns for s in self.session_list) / len(self.session_list) if self.session_list
                     else float(self.turns_cap))
            return min(len(self.prompts), sessions_for(self.turn_budget(concurrency), turns, self.pool_at(concurrency)))
        if not self.requests_per_slot:
            return len(self.prompts)
        return min(len(self.prompts), max(MIN_LEVEL_REQUESTS, concurrency * self.requests_per_slot))

    @property
    def reuses_prefix(self) -> bool:
        """Whether requests repeat a prefix an engine can cache: a shared one, or a session's own history."""
        return self.shared_prefix_tokens > 0 or self.sessions

    def head(self, n: int) -> "Workload":
        """A copy sending only the first n prompts, or for a session file drawing on only the first n sessions."""
        return replace(self, prompts=self.prompts[:n], n_prompts=n, session_list=self.session_list[:n])

    def turn_budget(self, concurrency: int) -> int:
        """Turns a level of a session file sends: it stops issuing new ones once they are spent (see turn_budget)."""
        per_session = self.turns_per_slot if self.history == "explicit" else TURNS_PER_POOLED_SESSION
        return self.level_turns or turn_budget(concurrency, self.turns_per_slot, self.pool_at(concurrency), per_session)

    def pool_at(self, concurrency: int) -> int:
        """Sessions live at a level of this concurrency: --session-pool, except that a file of explicit requests caps
        it at POOL_CAP per slot. A pool asks for a window per live session, so uncapped, a pool of 64 asked a level of
        one slot for 448 requests in a row; capped, the busiest level still gets the pool a validation asks for."""
        if self.history == "explicit" and self.session_pool:
            return min(self.session_pool, POOL_CAP * concurrency)
        return self.session_pool

    def ensure_prompts(self) -> "Workload":
        """Generate or download the prompts if they are not there yet."""
        if not self.prompts:
            self.prompts = self._generate()
            self.token_counts = []
        return self

    @property
    def _unique_tokens(self) -> int:
        return max(8, self.prefill_tokens - self.shared_prefix_tokens)

    def _prefix(self, n_words: int) -> str:
        rng = random.Random(7_919 * (self.seed + 1))
        return "Context: " + " ".join(rng.choice(_WORDS) for _ in range(n_words)) + "\n\n"

    @property
    def min_ctx(self) -> int:
        """Smallest per-request context a config must offer to run this workload."""
        return self.prefill_tokens + self.decode_tokens + CTX_HEADROOM

    def _words(self, i: int, n_words: int) -> List[str]:
        rng = random.Random(self.seed * 100_003 + i)
        return [rng.choice(_WORDS) for _ in range(n_words)]

    def _generate(self) -> List[str]:
        if self.sessions:
            from rigtune.calibrate import datasets

            self.session_list = datasets.sample_sessions(self.source, self.n_prompts, self.sample_offset,
                                                         self.turns_cap, self.session_system or None,
                                                         self.force_sessions)
            return [s.first_turn for s in self.session_list]
        if self.source != "synthetic":
            from rigtune.calibrate import datasets

            return datasets.sample(self.source, self.n_prompts, self.sample_offset,
                                   self.prefill_tokens * datasets.CHARS_PER_TOKEN)
        # ~1.25 tokens per short English word on Llama/Qwen tokenizers, so start at 0.8 x tokens words.
        n_words = max(8, int(self._unique_tokens * 0.8))
        return [self.prefix_text + _PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX
                for i in range(self.n_prompts)]

    def fit_prompts(self, counter: TokenCounter, tolerance: int = 2, max_iter: int = 8) -> "Workload":
        """Resize each prompt so the tokenizer counts within `tolerance` of prefill_tokens.

        Real-text prompts keep their own length; only those over the cap are cut, from the body so
        their instruction survives."""
        if not counter.available:
            return self
        if self.sessions:
            # A session's turns are never cut: one whose next prompt would not fit is stopped during replay.
            self.ensure_prompts()
            self.fitted = True
            return self
        if self.source != "synthetic":
            from rigtune.calibrate import datasets

            self.ensure_prompts()
            fitted_real: List[str] = []
            for i, prompt in enumerate(self.prompts):
                n, chars = counter.count(prompt), len(prompt)
                for _ in range(max_iter):
                    if n is None or n <= self.prefill_tokens:
                        break
                    chars = int(chars * self.prefill_tokens / n * 0.95)
                    prompt = datasets.sample(self.source, 1, self.sample_offset + i, chars)[0]
                    n = counter.count(prompt)
                fitted_real.append(prompt)
            self.prompts = fitted_real
            self.token_counts = []
            self.fitted = True
            return self
        if self.shared_prefix_tokens and not self.prefix_fixed:
            n_words = max(4, int(self.shared_prefix_tokens * 0.8))
            text = self._prefix(n_words)
            for _ in range(max_iter):
                text = self._prefix(n_words)
                n = counter.count(text)
                if n is None:
                    return self
                if abs(n - self.shared_prefix_tokens) <= tolerance:
                    break
                delta = self.shared_prefix_tokens - n
                step = int(round(delta * n_words / max(n, 1)))
                n_words = max(4, n_words + (step if step != 0 else (1 if delta > 0 else -1)))
            self.prefix_text = text
        fitted: List[str] = []
        for i in range(self.n_prompts):
            n_words = max(4, int(self._unique_tokens * 0.8))
            prompt = ""
            for _ in range(max_iter):
                prompt = self.prefix_text + _PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX
                n = counter.count(prompt)
                if n is None:
                    return self
                if abs(n - self.prefill_tokens) <= tolerance:
                    break
                delta = self.prefill_tokens - n
                step = int(round(delta * n_words / max(n, 1)))
                n_words = max(4, n_words + (step if step != 0 else (1 if delta > 0 else -1)))
            fitted.append(prompt)
        self.prompts = fitted
        self.token_counts = []
        self.fitted = True
        return self

    def measured_prompt_tokens(self, counter: TokenCounter) -> Optional[int]:
        if not counter.available or not self.prompts:
            return None
        counts = [counter.count(p) or 0 for p in self.prompts]
        self.token_counts = counts
        return int(round(sum(counts) / len(counts)))

    def prompt_token_stats(self) -> Optional[Dict[str, int]]:
        """How long the calibration prompts actually were, or None if no tokenizer measured these ones.

        A mean describes no real request when the set is bimodal (short chats beside long documents),
        and it leaves live traffic's p90 with nothing to be compared against. The tail is what breaches
        a TTFT ceiling, so it is recorded too.
        """
        if not self.token_counts or len(self.token_counts) != len(self.prompts):
            return None
        counts = self.token_counts
        return {"p50": _pct(counts, 50), "p90": _pct(counts, 90), "max": max(counts),
                "mean": int(round(sum(counts) / len(counts)))}

    @property
    def _source_label(self) -> str:
        return Path(self.source[len(FILE_PREFIX):]).name if self.source.startswith(FILE_PREFIX) else self.source

    def describe(self) -> str:
        if self.sessions:
            return (f"{self.name}: {self.turns_per_slot} turns per slot from {self._source_label} sessions of up to "
                    f"{self.turns_cap} "
                    + ("requests given whole" if self.history == "explicit" else f"turns with {self.history} history")
                    + f", prompts up to {self.prefill_tokens} "
                    f"tokens, answers up to {self.decode_tokens} tokens (natural stop), "
                    f"{f'{self.session_pool} sessions live' if self.session_pool else 'one session per slot'}, "
                    f"concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
                    f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
                    + (f", TPOT ceiling {self.tpot_ceiling_ms:.0f} ms" if self.tpot_ceiling_ms else ""))
        if self.source != "synthetic":
            return (f"{self.name}: {self.n_prompts} {self._source_label} prompts of up to {self.prefill_tokens} tokens, "
                    f"answers up to {self.decode_tokens} tokens{' (natural stop)' if self.natural_stop else ''}, "
                    f"concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
                    f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
                    + (f", TPOT ceiling {self.tpot_ceiling_ms:.0f} ms" if self.tpot_ceiling_ms else ""))
        return (
            f"{self.name}: {self.n_prompts} prompts x {'' if self.fitted else '~'}{self.prefill_tokens} prefill"
            f"{f' ({self.shared_prefix_tokens} shared)' if self.shared_prefix_tokens else ''} x "
            f"{self.decode_tokens} decode, concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
            f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
            + (f", TPOT ceiling {self.tpot_ceiling_ms:.0f} ms" if self.tpot_ceiling_ms else "")
        )

    def spec(self, served: Optional[object] = None) -> Dict[str, object]:
        """Serialisable description stored in profiles and benchmark results. For a session file, `served` (the
        winning trial's TrialMetrics) supplies what replay served: prompt lengths from the engine's own counts,
        turns reached, and the reuse achieved. They come from that one trial, not from every trial, so neither a
        losing configuration nor a prefix-cache-off variant colours what live traffic is compared with."""
        stats = self.prompt_token_stats()
        return {
            "name": self.name,
            "n_prompts": self.n_prompts,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "concurrencies": list(self.concurrencies),
            "ttft_ceiling_ms": self.ttft_ceiling_ms,
            "tpot_ceiling_ms": self.tpot_ceiling_ms,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "seed": self.seed,
            "source": self.source,
            "natural_stop": self.natural_stop,
            **({"requests_per_slot": self.requests_per_slot} if self.requests_per_slot else {}),
            **({"prompt_tokens_seen": stats} if stats else {}),
            **(self._session_spec(served) if self.sessions else {}),
        }

    def _session_spec(self, served: Optional[object] = None) -> Dict[str, object]:
        out: Dict[str, object] = {"sessions": True, "turns_cap": self.turns_cap, "history": self.history,
                                  "session_pool": self.session_pool, "turns_per_slot": self.turns_per_slot}
        if served is not None:
            if getattr(served, "prompt_tokens_seen", None):  # replaces the first-turn counts
                out["prompt_tokens_seen"] = dict(served.prompt_tokens_seen)
            for key, name in (("turns_seen", "turns_seen"), ("prefix_reuse", "reuse_achieved"),
                              ("effective_pool", "effective_pool"), ("reuse_distance", "reuse_distance")):
                value = getattr(served, key, None)
                if value not in (None, {}):
                    out[name] = value
        return out


_PRESETS: Dict[str, Workload] = {
    "default": Workload(name="default"),
    "chat": Workload(name="chat", prefill_tokens=512, decode_tokens=128, tpot_ceiling_ms=50.0),
    "long-context": Workload(
        name="long-context", n_prompts=8, prefill_tokens=8192, decode_tokens=256,
        concurrencies=(1, 2, 4), ttft_ceiling_ms=2000.0,
    ),
    "generation": Workload(name="generation", prefill_tokens=128, decode_tokens=1024, tpot_ceiling_ms=50.0),
    "high-concurrency": Workload(
        name="high-concurrency", n_prompts=256, prefill_tokens=256, decode_tokens=64,
        concurrencies=(32, 64, 128), ttft_ceiling_ms=1000.0, tpot_ceiling_ms=150.0,
    ),
    "rag": Workload(name="rag", prefill_tokens=6144, decode_tokens=64, ttft_ceiling_ms=1500.0),
    # Shared-prefix shapes, where prefix caching decides time to first token.
    "chat-system": Workload(name="chat-system", prefill_tokens=2048, shared_prefix_tokens=1536, decode_tokens=128,
                            tpot_ceiling_ms=50.0),
    "rag-shared": Workload(name="rag-shared", prefill_tokens=6144, shared_prefix_tokens=5632, decode_tokens=64,
                           ttft_ceiling_ms=1500.0),
    # Real text, for strategies that depend on content. Prompt and answer lengths are the data's and
    # the model's own; the figures here are caps.
    "sharegpt": Workload(name="sharegpt", source="sharegpt", natural_stop=True, n_prompts=64, prefill_tokens=1024,
                         decode_tokens=512, concurrencies=(1, 8, 32), ttft_ceiling_ms=1000.0, requests_per_slot=2),
    "extract": Workload(name="extract", source="cnn-extract", natural_stop=True, n_prompts=32, prefill_tokens=1536,
                        decode_tokens=384, ttft_ceiling_ms=1500.0, tpot_ceiling_ms=50.0, requests_per_slot=4),
    "code-edit": Workload(name="code-edit", source="humaneval-edit", natural_stop=True, n_prompts=32,
                          prefill_tokens=768, decode_tokens=512, tpot_ceiling_ms=50.0, requests_per_slot=4),
}

WORKLOAD_NAMES: Tuple[str, ...] = tuple(_PRESETS)


def get_workload(name: str = "default") -> Workload:
    """A fresh copy of a preset (prompts are regenerated, so fitting one never affects another)."""
    if name not in _PRESETS:
        raise ValueError(f"unknown workload {name!r}; choose from {', '.join(WORKLOAD_NAMES)}")
    base = _PRESETS[name]
    return replace(base, prompts=[], fitted=False)


def workload_from_file(path, template: Optional[Workload] = None, session_pool: Optional[int] = None,
                       turns_cap: Optional[int] = None, history: Optional[str] = None,
                       session_system: Optional[str] = None, sessions: bool = False) -> Workload:
    """Your own prompts as a workload, shaped like `template` (default `default`): its concurrency levels
    and latency ceilings are kept, and the prompts come from the file (see datasets.read_prompt_file).

    Answers stop when the model does, capped at 512 tokens or the template's decode length if longer.
    The name carries a hash of the file, so a profile calibrated on one version of it is never served for
    another. Each concurrency level gets prompts of its own: sending the same ones again would let the
    engine's prefix cache make later levels look faster, so a small file sends fewer requests instead.

    A file whose lines name a session, or any file with `sessions` set, becomes a session workload instead
    (_session_workload); the session options apply only to one, and a flat file given any of them is refused."""
    from rigtune.calibrate import datasets

    p = Path(path).expanduser().resolve()
    found = datasets.read_sessions(p, force=sessions)
    if found:
        return _session_workload(p, found, template or get_workload("default"), session_pool, turns_cap,
                                 history, session_system, sessions)
    given = [name for name, v in (("--session-pool", session_pool), ("--session-turns", turns_cap),
                                  ("--history", history), ("--session-system", session_system)) if v is not None]
    if given:
        raise ValueError(f"{p.name} holds no sessions (no line has a \"session\" key), so {', '.join(given)} "
                         "cannot apply to it")
    conversations, ignored = datasets.flat_notes(p)
    if conversations:
        logger.info("%s: %d conversations are flattened into one prompt each, as before; pass --sessions (or give "
                    "each line a \"session\" key) to replay them turn by turn", p.name, conversations)
    if ignored:
        logger.info("%s: %d lines have a \"session\" key beside a prompt rather than messages; it is ignored, as "
                    "before", p.name, ignored)
    prompts = datasets.read_prompt_file(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", p.stem)[:40] or "prompts"
    t = template or get_workload("default")
    per_slot = t.requests_per_slot or 4
    levels = len(t.concurrencies)
    want = max(max(MIN_LEVEL_REQUESTS, c * per_slot) for c in t.concurrencies)
    per_level = max(1, min(want, len(prompts) // levels))
    if per_level < want:
        logger.warning("%s has %d prompts: each of the %d concurrency levels gets %d of its own (it could use %d)%s",
                       p.name, len(prompts), levels, per_level, want,
                       f"; with fewer than {max(t.concurrencies)} a level cannot reach its concurrency"
                       if per_level < max(t.concurrencies) else "")
    if len(prompts) < levels:
        logger.warning("%s has fewer prompts than concurrency levels, so levels repeat prompts and the prefix cache "
                       "can make later levels look faster", p.name)
    longest = max(len(x) for x in prompts) // CHARS_PER_TOKEN + 1
    return replace(t, name=f"file-{stem}-{digest}", source=FILE_PREFIX + str(p), natural_stop=True,
                   n_prompts=per_level, prefill_tokens=min(MAX_FILE_PREFILL, -(-longest // 256) * 256),
                   decode_tokens=max(t.decode_tokens, 512), shared_prefix_tokens=0, prefix_text="", prefix_fixed=False,
                   prompts=[], fitted=False, requests_per_slot=per_slot, sample_offset=0)


def turn_budget(concurrency: int, turns_per_slot: int = TURNS_PER_SLOT, pool: int = 0,
                per_session: int = TURNS_PER_POOLED_SESSION) -> int:
    """Turns a level of a session file sends: TURNS_PER_SLOT per concurrency slot, and with a pool enough for each
    live session to get TURNS_PER_POOLED_SESSION (`per_session`). A budget of slots alone (6c) left a pool above it
    with no second turn at all: at c = 8 with a pool of 64, all 48 turns went to first turns and nothing was
    reused."""
    return max(MIN_LEVEL_REQUESTS, concurrency * turns_per_slot, per_session * pool)


def sessions_for(budget: int, mean_turns: float, pool: int = 0) -> int:
    """Sessions a level needs: twice its turn budget's worth, so it is still admitting new sessions when the budget
    runs out, and at least twice the pool, so a pool is refilled at least once and reaches its steady spacing."""
    return max(MIN_LEVEL_SESSIONS, -(-2 * budget // max(1, int(mean_turns))), 2 * pool)


def _session_workload(p: Path, sessions: List[Session], t: Workload, session_pool: Optional[int],
                      turns_cap: Optional[int], history: Optional[str], session_system: Optional[str],
                      forced: bool) -> Workload:
    """A file of conversations as a workload, replayed turn by turn.

    prefill_tokens is the longest prompt a session can reach: its system prompt, every user turn, and the answers
    between them, each budgeted at the length the file recorded (capped at the answer cap; a generated answer
    that runs longer is caught during replay, which stops the session before a prompt that would not fit and
    counts it as cut). Budgeting every answer at the full cap instead put an 8-turn session past 4096 tokens and
    silently dropped every 4k context. An answer the file left empty is budgeted at the median of those it did
    record (at the cap when it recorded none). Explicit steps are sized by their own messages, counted the way
    replay estimates a prompt (measure._LiveSession.next_prompt_tokens). The name says it is a session workload
    and carries every option that changes what is sent, so no profile is ever served across them.

    A file of explicit requests (an agent's: each request's messages given whole) has no history to choose and no
    system prompt to replace, so --history and --session-system are refused for it, its history is recorded as
    "explicit", and its name says so ("-sessions-explicit"), so it can never share a name with a file read before
    explicit requests existed. Each of its requests is self-contained, so a level replays a window of each session
    rather than its start (datasets.window): one primer request, unmeasured, then --session-turns measured ones
    (EXPLICIT_WINDOW by default), from an offset spread over the session's length, so a level meets the requests an
    agent sends once its context has filled as often as the file holds them. A level's budget is a window per slot,
    and with a pool a window per live session, the pool capped at POOL_CAP per slot (Workload.pool_at)."""
    explicit = any(step.explicit for s in sessions for step in s.steps)
    if explicit:
        given = [name for name, v in (("--history", history), ("--session-system", session_system)) if v is not None]
        if given:
            raise ValueError(f"{p.name} gives each request's messages whole, so {', '.join(given)} cannot apply to it: "
                             "the file says what every request sends")
    cap = (EXPLICIT_WINDOW if explicit else DEFAULT_TURNS_CAP) if turns_cap is None else int(turns_cap)
    hist = "explicit" if explicit else history or "generated"
    pool = int(session_pool or 0)
    system = (session_system or "").lstrip("\ufeff").strip()
    if cap < 1:
        raise ValueError("--session-turns must be at least 1")
    if hist not in HISTORIES and not explicit:
        raise ValueError(f"--history must be one of {', '.join(HISTORIES)}")
    if pool < 0:
        raise ValueError("--session-pool cannot be negative")
    if pool and pool < max(t.concurrencies):
        logger.warning("--session-pool %d is below the busiest level's concurrency %d; that level keeps one session "
                       "per slot", pool, max(t.concurrencies))
    digest = hashlib.sha256(p.read_bytes() + (b"\0system\0" + system.encode() if system else b"")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", p.stem)[:40] or "prompts"
    cut = turns_cap is not None if explicit else cap != DEFAULT_TURNS_CAP
    name = (f"file-{stem}-{digest}-sessions" + ("-explicit" if explicit else "") + (f"-t{cap}" if cut else "")
            + (f"-pool{pool}" if pool else "") + ("-recorded" if hist == "recorded" else ""))
    decode = max(t.decode_tokens, 512)

    def tokens(text: str) -> int:
        return len(text) // CHARS_PER_TOKEN + 1

    # Any request of an explicit session can fall in a window, so all of them size the context; a level's sessions are
    # windows, here one per session at evenly spread offsets, for the mean a level's sessions hold.
    capped = sessions if explicit else [s.capped(cap) for s in sessions]
    replayed = ([window(s, cap, (i + 0.5) / len(sessions)) for i, s in enumerate(sessions)] if explicit
                else capped)
    recorded =sorted(tokens(st.answer) for s in capped for st in s.steps if st.answer)
    unrecorded = min(decode, recorded[len(recorded) // 2]) if recorded else decode

    def answer(step) -> int:
        n = tokens(step.answer) if step.answer else unrecorded
        return n if hist == "recorded" and step.answer else min(decode, n)

    def longest(s: Session) -> int:
        sys_text = system or s.system
        convo = tokens(sys_text) + TURN_OVERHEAD if sys_text else 0  # what the next append step builds on
        best = 0
        for step in s.steps:
            if step.explicit:
                prompt = sum(tokens(c) + TURN_OVERHEAD for _, c in step.messages)
            else:
                prompt = convo + tokens(step.user) + TURN_OVERHEAD
            best = max(best, prompt)
            convo = prompt + answer(step) + TURN_OVERHEAD
        return best

    mean_turns = sum(s.turns for s in replayed) / len(replayed)
    per_slot = cap + 1 if explicit else TURNS_PER_SLOT  # explicit: the primer and the measured requests
    per_session = per_slot if explicit else TURNS_PER_POOLED_SESSION
    levels = len(t.concurrencies)
    pools = [min(pool, POOL_CAP * c) if explicit else pool for c in t.concurrencies]
    budgets = [turn_budget(c, per_slot, pc, per_session) for c, pc in zip(t.concurrencies, pools)]
    want = max(sessions_for(b, mean_turns, pc) for b, pc in zip(budgets, pools))
    # Every level, and every level's close-call re-measure, gets sessions of its own (measure._level_workload).
    per_level = max(1, min(want, len(sessions) // (2 * levels)))
    if per_level < want:
        logger.warning("%s has %d sessions: each of the %d concurrency levels, and its re-measure, gets %d of its own "
                       "(it could use %d), so the busiest levels may run out of sessions before their turn budget and "
                       "end below their concurrency", p.name, len(sessions), levels, per_level, want)
    if per_level * mean_turns < max(budgets):
        logger.warning("%s: the busiest level's %d sessions hold about %.0f turns, short of its turn budget of %d%s; "
                       "it will send fewer turns than asked", p.name, per_level, per_level * mean_turns, max(budgets),
                       f" (a pool of {pool} asks for {per_session} turns per live session)" if pool else "")
    if len(sessions) < 2 * levels:
        logger.warning("%s has fewer sessions than concurrency levels and re-measures, so levels repeat sessions and "
                       "the prefix cache can make later levels look faster", p.name)
    worst = max(longest(s) for s in capped)
    return replace(t, name=name, source=FILE_PREFIX + str(p), natural_stop=True, n_prompts=per_level,
                   prefill_tokens=min(MAX_FILE_PREFILL, -(-worst // 256) * 256), decode_tokens=decode,
                   shared_prefix_tokens=0, prefix_text="", prefix_fixed=False, prompts=[], fitted=False,
                   requests_per_slot=0, sample_offset=0, sessions=True, force_sessions=forced, turns_cap=cap,
                   history=hist, session_pool=pool, turns_per_slot=per_slot, session_system=system,
                   session_list=[])


def workload_table() -> List[Dict[str, object]]:
    return [_PRESETS[n].spec() for n in WORKLOAD_NAMES]
