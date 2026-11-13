"""Does live traffic still look like the traffic this profile was tuned on?

A profile is calibrated for one shape of work — prompt length, answer length, concurrency — and nothing re-tunes
itself when real traffic drifts away from that shape. The proxy feeds every request through here, and the watch
says in one report whether what is being served still resembles what was measured, so `rigtune recalibrate`
is a decision rather than a guess.

Only what the backend already reports is counted: the `usage` block, which non-streamed replies carry anyway and
streamed ones carry because the proxy asks for it (`stream_options: {"include_usage": true}`). A caller who set
`include_usage` themselves keeps their choice, and a backend that ignores the request answers without usage; those
replies still count as requests and toward concurrency, but contribute no lengths. Nothing here reads prompt text.

Served latency is watched the same way, for streamed replies only, and is reported rather than warned about,
because a load spike breaches a ceiling through queueing and re-tuning would not fix that: see `latency_findings`.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

WINDOW = 512  # requests kept for the rolling picture
MIN_REQUESTS = 50  # below this, say "not enough traffic yet" rather than cry drift
WIDER_THAN = 1.5  # a median this many times the calibrated value counts as drift
NARROWER_THAN = 1 / WIDER_THAN
MIN_LATENCY_SAMPLES = 20  # streamed replies needed before a ceiling breach is worth saying
# The proxy sees queueing and its own hop on top of the engine's own time to first token, so a breach is
# called only when the ceiling is clearly passed rather than grazed.
BREACH_MARGIN = 1.2
RATE_OFF_BY = 2.0  # live arrivals outside [1/2, 2]x the rate a profile was scored for count as drift


def _median(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return float(ordered[idx])


class TrafficWatch:
    """Rolling picture of served traffic, against the workload the profile was calibrated on."""

    def __init__(self, workload_spec: Optional[Dict[str, Any]] = None, window: int = WINDOW,
                 min_requests: int = MIN_REQUESTS, clock: Callable[[], float] = time.monotonic) -> None:
        spec = workload_spec or {}
        # The rate an `efficiency --arrival-rate` profile was scored for; None for every other profile.
        self.calibrated_rate = spec.get("arrival_rate_rps")
        self._clock = clock
        self.arrivals: Deque[float] = deque(maxlen=window)
        self.workload_name = spec.get("name")
        # Prompts a tokenizer actually measured describe the calibration set better than the nominal
        # prefill, which is only a cap for a --workload-file and a mean at best. Older profiles carry none.
        seen = spec.get("prompt_tokens_seen") or {}
        self.calibrated_prompt = seen.get("p50") or spec.get("prefill_tokens")
        self.calibrated_prompt_p90 = seen.get("p90")
        self.calibrated_completion = spec.get("decode_tokens")
        self.calibrated_concurrency: List[int] = [int(c) for c in (spec.get("concurrencies") or [])]
        self.ttft_ceiling_ms = spec.get("ttft_ceiling_ms")
        self.ttft_ms: Deque[float] = deque(maxlen=window)
        self.min_requests = min_requests
        self.requests = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.prompt_tokens: Deque[int] = deque(maxlen=window)
        self.completion_tokens: Deque[int] = deque(maxlen=window)
        self.concurrency: Deque[int] = deque(maxlen=window)
        # Prefix reuse, (prompt tokens, cached tokens) per reply, watched for session profiles only: they launch the
        # engine to report a count and record what calibration achieved. SGLang leaves the count out when nothing
        # was cached, so a reply without one counts as zero, as it did in calibration. Other profiles' reports keep
        # the shape they had.
        self.calibrated_reuse = spec.get("reuse_achieved")
        self.sessions = bool(spec.get("sessions"))
        self.reuse: Deque[tuple] = deque(maxlen=window)
        self._warned = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ recording

    def began(self) -> None:
        with self._lock:
            self.arrivals.append(self._clock())
            self.requests += 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.concurrency.append(self.in_flight)

    def ended(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def record_response(self, content: bytes, media_type: str = "") -> None:
        """Take the token counts out of a non-streaming response body, if it carries a usage block."""
        if "json" not in media_type.lower() or not content:
            return
        try:
            body = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            return
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return
        with self._lock:
            for key, into in (("prompt_tokens", self.prompt_tokens), ("completion_tokens", self.completion_tokens)):
                value = usage.get(key)
                if isinstance(value, int) and value > 0:
                    into.append(value)
        self._record_reuse(usage)

    def _record_reuse(self, usage: Dict[str, Any]) -> None:
        details = usage.get("prompt_tokens_details")
        prompt = usage.get("prompt_tokens")
        if not self.sessions or not isinstance(prompt, int) or prompt <= 0:
            return
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if not isinstance(cached, int):
            cached = 0
        with self._lock:
            self.reuse.append((prompt, cached))

    def live_reuse(self) -> Optional[float]:
        """Share of prompt tokens served from the prefix cache, over the replies counted."""
        with self._lock:
            prompts = sum(p for p, _ in self.reuse)
            return round(sum(c for _, c in self.reuse) / prompts, 4) if prompts else None

    def record_stream_chunk(self, chunk: bytes) -> None:
        """Take token counts out of a streamed reply's usage chunk.

        Streamed replies carry no counts unless the request asked for them, and most OpenAI clients stream, so
        the proxy adds `stream_options: {"include_usage": true}` on the way past and reads the result here.
        """
        if not chunk:
            return
        for line in chunk.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                body = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                continue
            usage = body.get("usage") if isinstance(body, dict) else None
            if not isinstance(usage, dict):
                continue
            with self._lock:
                for key, into in (("prompt_tokens", self.prompt_tokens),
                                  ("completion_tokens", self.completion_tokens)):
                    value = usage.get(key)
                    if isinstance(value, int) and value > 0:
                        into.append(value)
            self._record_reuse(usage)

    def record_ttft(self, ms: float) -> None:
        """Time from the request reaching the proxy to the first streamed chunk leaving it.

        Only a streamed reply exposes this: a non-streamed body arrives whole, so its total time is not a
        time to first token and is not recorded as one. This measurement includes queueing and the proxy's
        own hop, so it reads a little above what calibration measured against the engine directly.
        """
        if ms >= 0:
            with self._lock:
                self.ttft_ms.append(float(ms))

    # ------------------------------------------------------------------ reading

    @property
    def enough_data(self) -> bool:
        return self.requests >= self.min_requests

    def live_rate(self) -> Optional[float]:
        """Requests per second over the arrivals in the window: the gaps between them, not the time since."""
        if len(self.arrivals) < 2:
            return None
        span = self.arrivals[-1] - self.arrivals[0]
        return (len(self.arrivals) - 1) / span if span > 0 else None

    def findings(self) -> List[str]:
        """Plain sentences about how live traffic differs from the calibrated workload; empty when it matches."""
        out: List[str] = []
        if not self.enough_data:
            return out
        flagged = set()
        for label, seen, calibrated in (("prompts", _median(self.prompt_tokens), self.calibrated_prompt),
                                        ("answers", _median(self.completion_tokens), self.calibrated_completion)):
            if seen is None or not calibrated:
                continue
            ratio = seen / calibrated
            if ratio >= WIDER_THAN or ratio <= NARROWER_THAN:
                longer = "longer" if ratio > 1 else "shorter"
                flagged.add(label)
                out.append(f"{label} are {max(ratio, 1 / ratio):.1f}x {longer} than the profile was tuned for "
                           f"(median {seen:.0f} tokens against {calibrated})")
        # A median that still matches can hide a tail that no longer does, and the long requests are the
        # ones that breach a latency ceiling. Only the heavy side is worth saying, and only when the
        # median said nothing, so one shift is never reported twice.
        live_p90 = _percentile(self.prompt_tokens, 90)
        if "prompts" not in flagged and live_p90 and self.calibrated_prompt_p90:
            ratio = live_p90 / self.calibrated_prompt_p90
            if ratio >= WIDER_THAN:
                out.append(f"the longest prompts are {ratio:.1f}x longer than the profile was tuned for "
                           f"(p90 {live_p90:.0f} tokens against {self.calibrated_prompt_p90})")
        typical = _median(self.concurrency)
        if typical is not None and self.calibrated_concurrency:
            top = max(self.calibrated_concurrency)
            if typical > top:
                levels = "/".join(str(c) for c in self.calibrated_concurrency)
                out.append(f"traffic runs at {typical:.0f} concurrent requests; the profile was measured at {levels}")
        # A profile scored for an arrival rate chose by energy per request at that rate, and the idle share of each
        # request scales with the gap between them: past a factor of two the choice may no longer hold.
        live = self.live_rate()
        if self.calibrated_rate and live:
            ratio = live / self.calibrated_rate
            if ratio > RATE_OFF_BY or ratio < 1 / RATE_OFF_BY:
                more = "more" if ratio > 1 else "fewer"
                out.append(f"requests arrive at {live:.2g}/s, {max(ratio, 1 / ratio):.1f}x {more} than the "
                           f"{self.calibrated_rate:g}/s this profile's energy was scored for")
        return out

    def latency_findings(self) -> List[str]:
        """Whether served latency still respects the ceiling the profile was tuned against.

        Deliberately kept out of `findings`: a load spike breaches the ceiling through queueing, which
        re-tuning cannot fix, so this reports and never triggers the drift warning. The load it was seen
        at is named alongside, because that is what separates "the wrong configuration" from "more
        traffic than this was ever measured at".
        """
        out: List[str] = []
        if len(self.ttft_ms) < MIN_LATENCY_SAMPLES or not self.ttft_ceiling_ms:
            return out
        p95 = _percentile(self.ttft_ms, 95)
        if p95 is not None and p95 > self.ttft_ceiling_ms * BREACH_MARGIN:
            busy = _median(self.concurrency)
            at = f", at {busy:.0f} concurrent requests" if busy is not None else ""
            out.append(f"time to first token is {p95:.0f} ms at the 95th percentile{at}, past the "
                       f"{self.ttft_ceiling_ms:.0f} ms ceiling this profile was tuned against")
        return out

    def latency_verdict(self) -> str:
        if not self.ttft_ceiling_ms:
            return "no ceiling recorded in this profile"
        if len(self.ttft_ms) < MIN_LATENCY_SAMPLES:
            return "not enough streamed traffic yet"
        return "past the calibrated ceiling" if self.latency_findings() else "inside the calibrated ceiling"

    def should_warn(self) -> bool:
        """True once, the first time drift is worth telling the operator about."""
        with self._lock:
            if self._warned or not self.findings():
                return False
            self._warned = True
            return True

    def report(self) -> Dict[str, Any]:
        findings = self.findings()
        reuse = {"prefix_reuse": {"live": self.live_reuse(), "calibrated": self.calibrated_reuse,
                                  "replies_counted": len(self.reuse)}} if self.sessions else {}
        return {
            "calibrated_workload": self.workload_name,
            "requests_seen": self.requests,
            "requests_with_token_counts": len(self.completion_tokens),
            "enough_data": self.enough_data,
            "prompt_tokens": {"median": _median(self.prompt_tokens), "p90": _percentile(self.prompt_tokens, 90),
                              "calibrated": self.calibrated_prompt,
                              "calibrated_p90": self.calibrated_prompt_p90},
            "completion_tokens": {"median": _median(self.completion_tokens),
                                  "p90": _percentile(self.completion_tokens, 90),
                                  "calibrated": self.calibrated_completion},
            "concurrency": {"median": _median(self.concurrency), "peak": self.peak_in_flight,
                            "calibrated": self.calibrated_concurrency},
            "arrival_rate": {"live_rps": self.live_rate(),
                             **({"calibrated_rps": self.calibrated_rate} if self.calibrated_rate else {})},
            "latency": {"ttft_ms": {"median": _percentile(self.ttft_ms, 50),
                                    "p95": _percentile(self.ttft_ms, 95)},
                        "ceiling_ms": self.ttft_ceiling_ms,
                        "streamed_requests_timed": len(self.ttft_ms),
                        "findings": self.latency_findings(),
                        "verdict": self.latency_verdict()},
            # Reported, never warned about: reuse falls with a load spike or a cold cache as well as with a change
            # in traffic, and only the last is something re-tuning would fix.
            **reuse,
            "drift": findings,
            "verdict": ("not enough traffic yet" if not self.enough_data else
                        "drifted from the calibrated workload" if findings else
                        "matches the calibrated workload"),
        }
