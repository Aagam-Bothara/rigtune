"""A conformance run: does an installed engine release do what rigtune plans for? Needs a GPU.

rigtune's engine rules were read from particular releases' source and measured on particular releases, and the
engine probe (rigtune.backends.facts) only says what exists: a flag can be there and mean something else. A
conformance run checks behaviour, one launch per check with a small model, through the backend's own launch command,
so what it certifies is what a calibration would launch:

  C01 the engine probe answers every question, and agrees with the source checks wherever both answer
  C02 a stock launch starts, serves, and its log yields the memory figures rigtune parses (weights, KV, vLLM's non-KV)
  C03 every launch flag rigtune emits starts and serves: each KV-cache type, the prefix cache on and off, a prefill
      budget (one launch each)
  C04 a launch that reports reuse puts cached tokens in the usage of a repeated request
  C05 the /metrics counters rigtune reads are exposed (vLLM; rigtune reads no SGLang metrics, so it does not apply)
  C06 each speculative method rigtune would try starts and serves (one result per method; vLLM's accepted-token
      counter must move); MTP with the hybrid model
  C07 an oversized configuration that fails to start is classified as out of memory at start-up, which is what the
      search's retry acts on (skipped when the card starts it anyway, or when vLLM's free-memory check refuses it)
  C08 vLLM's text-only launch of a multimodal checkpoint starts and serves (with a multimodal model)
  C09 a host-RAM KV-cache tier starts and a repeated request reports reuse
  C10 the hybrid model starts at the planner's configuration; the pools its log reports are recorded beside the plan
  C11 fp8 weights start and serve on this card
  C12 the load generator measures a fake engine faithfully at 64 and 128 users

Each check is independent: one that fails, or raises, is recorded and the run goes on, and so is each launch within
a check. A release is certified when C01 to C04 pass, C05 passes or does not apply to the engine, and C07 did not
fail; the rest are recorded per check. The result is one record in the format of rigtune/data/conformance/
(rigtune.engines.Tested), with each check's details, the probe's answers and the libraries, so committing it there
adds the release to the checked list. The record names no folder: every path in it is cut to its last part.

Never run on a GPU yet: every check has run only against a fake engine in the tests.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import httpx

from rigtune.backends.base import BaseBackend, Process, free_port
from rigtune.models import Config, HardwareDescriptor, PreparedModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CERTIFYING = ("C01", "C02", "C03", "C04", "C05")  # each passed, or skipped as not applying; and C07 must not fail
REQUESTS = 8
MAX_TOKENS = 16
# Each request asks the model to copy a passage the prompt holds, so its answer repeats the prompt and n-gram lookup,
# which proposes continuations found in the prompt, has something to match: "count from 1 to N" gave it nothing, and
# C06's accepted-token check could fail on a working engine.
PASSAGE = "The quick brown fox jumps over the lazy dog, and the lazy dog sleeps under the old oak tree."
# A prompt long enough to fill several KV blocks (vLLM caches whole blocks of 16 tokens), so a repeat can hit.
REUSE_PROMPT = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(60))
# C07: a batch and a prefill budget meant to run out of the memory outside the reservation (CUDA graphs and activations
# for them), at 90% reserved. Not more: vLLM refuses at start-up when free memory is under the fraction asked of the
# whole card (v1/worker/utils.py request_memory, line 761 at v0.33.0 and 525 at v0.29.0: "Free memory on device ... is
# less than desired GPU memory utilization"), which at 99% is every card, since the driver and the CUDA context
# already hold some. That ValueError says nothing of the configuration, so it is told apart (FREE_MEMORY_CHECK).
OVERSIZED = {"gpu_memory_utilization": 0.90, "batch": 1024, "prefill_budget": 32768}
FREE_MEMORY_CHECK = "is less than desired GPU memory utilization"
NOT_APPLICABLE = {"applies": False}  # a check's data when it does not apply to this engine
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"  # v1/spec_decode/metrics.py, line 231 at v0.29.0 and v0.33.0


@dataclass
class CheckResult:
    id: str
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Launch:
    """One engine launched for a check: whether it became ready, and its log."""

    ready: bool
    base_url: str
    log: Callable[[], str]
    error: Optional[str] = None


def startup_error(cfg: Config, log: str, returncode: Optional[int]) -> str:
    """The error a calibration trial records for this start-up failure (SubprocessTrialRunner.run), so C07 checks the
    classification the search acts on."""
    from rigtune.calibrate.search import mentions_oom, startup_finders

    cause = "out of memory at start-up; " if mentions_oom(log) else ""
    for kind, find in startup_finders(cfg, log):
        line = None if cause else find(log)
        if line:
            cause = f"{kind} ({line}); "
    return f"{cause}failed to start (rc={returncode})"


def serve(client: httpx.Client, base_url: str, model_name: str, n: int = REQUESTS) -> List[str]:
    """Send `n` short completion requests one after another; returns the errors (empty when all were answered)."""
    errors: List[str] = []
    for i in range(n):
        try:
            r = client.post(base_url + "/v1/completions", json={
                "model": model_name, "prompt": f"Copy this line exactly.\nLine {i}: {PASSAGE}\nCopy: Line {i}:",
                "max_tokens": MAX_TOKENS,
                "temperature": 0})
            text = r.json()["choices"][0]["text"] if r.status_code == 200 else None
            if not isinstance(text, str):
                errors.append(f"request {i}: HTTP {r.status_code}")
        except Exception as exc:
            errors.append(f"request {i}: {type(exc).__name__}: {exc}"[:200])
    return errors


def cached_tokens(client: httpx.Client, base_url: str, model_name: str) -> List[Optional[int]]:
    """The same chat request twice; each reply's usage.prompt_tokens_details.cached_tokens (None when absent)."""
    out: List[Optional[int]] = []
    for _ in range(2):
        r = client.post(base_url + "/v1/chat/completions", json={
            "model": model_name, "messages": [{"role": "user", "content": REUSE_PROMPT}],
            "max_tokens": MAX_TOKENS, "temperature": 0})
        details = ((r.json().get("usage") or {}).get("prompt_tokens_details") or {}) if r.status_code == 200 else {}
        out.append(details.get("cached_tokens"))
    return out


def metric_names(client: httpx.Client, base_url: str, path: str = "/metrics") -> Dict[str, float]:
    """Every sample on the engine's /metrics, by name without `_total`, summed over its labels."""
    text = client.get(base_url + path).text
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.partition("{")[0].strip().removesuffix("_total")
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


class Conformance:
    """One run against one engine. `models` maps a role ("small", and optionally "hybrid" and "multimodal") to its
    PreparedModel; tests give fakes for the launches and the client self-check."""

    def __init__(self, engine: str, backend: BaseBackend, hw: HardwareDescriptor, models: Dict[str, PreparedModel],
                 out_dir: Path, startup_timeout: float = 900.0,
                 selfcheck: Optional[Callable[[Sequence[int]], List[Any]]] = None,
                 reporting: Optional[BaseBackend] = None, only: Optional[Sequence[str]] = None,
                 poll: float = 1.0, drafts: str = "table", draft: Optional[str] = None, hub: str = "on"):
        self.engine, self.backend, self.hw, self.models = engine, backend, hw, models
        self.reporting = reporting or backend  # the same backend launched with per-reply cached tokens
        self.out_dir, self.startup_timeout = Path(out_dir), startup_timeout
        self.selfcheck = selfcheck
        self.only = set(only) if only else None
        self.poll = poll  # seconds between readiness checks of a launch
        self.launches = 0
        self.drafts, self.draft, self.hub = drafts, draft, hub
        if drafts == "verified" or draft:
            from rigtune.pairing import pin_target
            self.models = dict(models)
            for role, model in models.items():
                pinned = pin_target(model.spec, draft=draft, hub=hub)
                if pinned.revision != model.spec.revision:
                    self.models[role] = backend.prepare(pinned, hw, quants=list(model.weights_bytes) or None)
        self.http = httpx.Client(timeout=120.0)  # one client for the run: each new one loads the CA store

    # ---- launching

    @contextlib.contextmanager
    def launched(self, cfg: Config, model: PreparedModel, label: str,
                 backend: Optional[BaseBackend] = None) -> Iterator[Launch]:
        """Launch `cfg` through the backend's own launch command, wait for it, and always stop it."""
        be = backend or self.backend
        port = free_port()
        log_path = self.out_dir / "logs" / f"{self.launches:02d}-{label}.log"
        self.launches += 1
        proc: Optional[Process] = None
        try:
            proc = be.launch(cfg, model, port, log_path=log_path)
            ready = proc.wait_ready(timeout=self.startup_timeout, poll=self.poll)
            log = proc.tail_log
            error = None if ready else startup_error(cfg, log(400), proc.returncode())
            yield Launch(ready=ready, base_url=f"http://127.0.0.1:{port}", log=lambda: log(400), error=error)
        finally:
            if proc is not None:
                proc.stop()

    def _served(self, cfg: Config, model: PreparedModel, label: str) -> Dict[str, Any]:
        """Launch, send REQUESTS, stop: {"ready", "errors", "error"} for a sub-result."""
        with self.launched(cfg, model, label) as up:
            if not up.ready:
                return {"key": cfg.key(), "ready": False, "error": up.error}
            errors = serve(self.http, up.base_url, model.spec.hf_id)
            return {"key": cfg.key(), "ready": True, "errors": errors}

    @staticmethod
    def _isolated(launch: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """A sub-result whose launch raised (a refusal, a full disk) as a failed one, so the next one still runs."""
        try:
            return launch()
        except Exception as exc:
            return {"ready": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    @staticmethod
    def _ok(sub: Dict[str, Any]) -> bool:
        return bool(sub.get("ready")) and not sub.get("errors")

    def base(self, role: str = "small") -> Config:
        return self.backend.default_config(self.hw, self.models[role])

    # ---- the checks

    def c01_probe(self) -> CheckResult:
        name = "the engine probe answers every question and agrees with the source checks"
        facts = self.backend.facts(refresh=True)
        if facts is None:
            return CheckResult("C01", name, "fail", "the probe was not run (RIGTUNE_ENGINE_PROBE=off, or the engine was "
                               "not found)")
        module = sys.modules[type(self.backend).__module__]
        report = module.feature_report() if hasattr(module, "feature_report") else {}
        plain = {k: (sorted(p) if isinstance(p, (set, frozenset)) else p,
                     sorted(s) if isinstance(s, (set, frozenset)) else s) for k, (p, s) in report.items()}
        disagree = sorted(k for k, (p, s) in plain.items() if p is not None and p != s)
        missing = facts.unanswered()
        data = {"unanswered": {k: facts.errors.get(k, "") for k in missing}, "disagreements": disagree,
                "checks": {k: {"probe": p, "source": s} for k, (p, s) in plain.items()}, "seconds": facts.seconds}
        # A newly optional experimental capability does not revoke a legacy baseline
        # certification. Its absence stays in the record and withholds head quantization.
        required_missing = [k for k in missing if k != "spec_fields"]
        problems = ([f"not answered: {', '.join(required_missing)}"] if required_missing else []) + \
            ([f"the probe and the source disagree on {', '.join(disagree)}"] if disagree else [])
        return CheckResult("C01", name, "fail" if problems else "pass", "; ".join(problems), data)

    def c02_c05_stock(self) -> List[CheckResult]:
        """C02 and C05 share one stock launch."""
        from rigtune.memlog import parse_log

        n2 = "a stock launch starts, serves, and its log yields weights and KV memory"
        n5 = "the /metrics counters rigtune reads are exposed"
        model = self.models["small"]
        hooks = self.backend.workload_hooks(self.hw, model)
        not_applicable = CheckResult("C05", n5, "skip", f"rigtune reads no {self.engine} metrics",
                                    dict(NOT_APPLICABLE))
        with self.launched(self.base(), model, "stock") as up:
            if not up.ready:
                fail = f"did not start: {up.error}"
                return [CheckResult("C02", n2, "fail", fail),
                        CheckResult("C05", n5, "fail", fail) if hooks.metrics_path else not_applicable]
            errors = serve(self.http, up.base_url, model.spec.hf_id)
            measured = parse_log(self.engine, up.log())
            need = ["weights_mb", "kv_mb"] + (["non_kv_mb"] if self.engine == "vllm" else [])
            absent = [k for k in need if getattr(measured, k) is None]
            problems = errors[:3] + ([f"the log gave no {', '.join(absent)}"] if absent else [])
            c2 = CheckResult("C02", n2, "fail" if problems else "pass", "; ".join(problems),
                             {"measured": measured.model_dump(), "errors": errors})
            if not hooks.metrics_path:
                return [c2, not_applicable]
            from rigtune.calibrate.measure import PREFIX_HITS, PREFIX_QUERIES, PROMPT_BY_SOURCE

            try:
                names = metric_names(self.http, up.base_url, hooks.metrics_path)
            except Exception as exc:
                return [c2, CheckResult("C05", n5, "fail", f"{hooks.metrics_path} unreadable: {exc}")]
            wanted = (PREFIX_QUERIES, PREFIX_HITS, PROMPT_BY_SOURCE)
            gone = [w for w in wanted if w not in names]
            c5 = CheckResult("C05", n5, "fail" if gone else "pass", f"missing: {', '.join(gone)}" if gone else "",
                             {"present": sorted(w for w in wanted if w in names)})
            return [c2, c5]

    def c03_flags(self) -> CheckResult:
        name = "every launch flag rigtune emits starts and serves"
        model, base = self.models["small"], self.base()
        variants = [("kv-" + d, base.model_copy(update={"kv_dtype": d})) for d in self.backend.kv_dtypes(self.hw)]
        variants += [("prefix-on", base.model_copy(update={"prefix_cache": True})),
                     ("prefix-off", base.model_copy(update={"prefix_cache": False}))]
        budgets = self.backend.prefill_variants(base)
        if budgets:
            variants.append((f"prefill-{budgets[0].prefill_budget}", budgets[0]))
        subs = {label: self._isolated(lambda c=cfg, la=label: self._served(c, model, la))
                for label, cfg in variants}
        failed = sorted(k for k, s in subs.items() if not self._ok(s))
        return CheckResult("C03", name, "fail" if failed else "pass", f"failed: {', '.join(failed)}" if failed else "",
                           {"launches": subs})

    def c04_reuse(self) -> CheckResult:
        name = "a launch that reports reuse puts cached tokens in a repeated request's usage"
        model = self.models["small"]
        with self.launched(self.base(), model, "reuse", backend=self.reporting) as up:
            if not up.ready:
                return CheckResult("C04", name, "fail", f"did not start: {up.error}")
            counts = cached_tokens(self.http, up.base_url, model.spec.hf_id)
        ok = counts[-1] is not None and counts[-1] > 0
        return CheckResult("C04", name, "pass" if ok else "fail",
                           "" if ok else f"cached tokens on the repeat: {counts[-1]}", {"cached_tokens": counts})

    def c06_speculative(self) -> List[CheckResult]:
        from rigtune import speculative

        out: List[CheckResult] = []
        for role in ("small", "hybrid"):
            if role not in self.models:
                continue
            model = self.models[role]
            model.drafts_mode = "verified" if self.drafts == "verified" or self.draft else "table"
            from rigtune.pairing import hub_off, immutable_revision, is_local
            notes = ([] if model.drafts_mode == "verified" and hub_off(self.hub)
                     else self.backend.find_second_models(model))
            base = self.base(role)
            if model.drafts_mode == "verified":
                if is_local(model.spec.hf_id) or immutable_revision(model.spec.revision):
                    notes += self.backend.find_pairs(model, base.ctx, draft=self.draft, hub=self.hub, hw=self.hw)
                else:
                    notes.append("second models not checked: target revision was not pinned before preparation")
            # min_batch=1: an MTP setting that does not fit at the default batch is tried at the largest that does.
            variants = self.backend.spec_variants(base, model, self.hw, all_methods=True, min_batch=1)
            if role == "hybrid":  # the hybrid model is here for its own head; the small one covers the rest
                variants = [v for v in variants if speculative.own_head(v.spec_decode)]
            for v in variants:
                kind = speculative.parse(v.spec_decode)[0]
                cid = f"C06-{kind}" if role == "small" else f"C06-{kind}-hybrid"
                if v.spec_head_quant:
                    cid += "-" + v.spec_head_quant
                if any(r.id == cid for r in out):
                    continue  # one setting per method (an MTP head is offered at more than one k)
                try:
                    out.append(self._spec_check(cid, v, model, notes))
                except Exception as exc:  # one method's launch never stops the others
                    out.append(CheckResult(cid, f"speculative decoding {v.spec_decode} starts and serves", "fail",
                                           f"{type(exc).__name__}: {exc}"[:300]))
        if not out:
            out.append(CheckResult("C06", "speculative methods start and serve", "skip",
                                   "rigtune offers no speculative method here"))
        return out

    def _spec_check(self, cid: str, cfg: Config, model: PreparedModel, notes: List[str]) -> CheckResult:
        name = f"speculative decoding {cfg.spec_decode} starts and serves"
        with self.launched(cfg, model, cid) as up:
            if not up.ready:
                return CheckResult(cid, name, "fail", f"did not start: {up.error}", {"notes": notes})
            errors = serve(self.http, up.base_url, model.spec.hf_id)
            accepted = None
            if self.engine == "vllm":
                try:
                    accepted = metric_names(self.http, up.base_url).get(SPEC_ACCEPTED)
                except Exception:
                    accepted = None
        problems = errors[:3] + (["no accepted draft tokens counted"] if self.engine == "vllm" and not accepted else [])
        return CheckResult(cid, name, "fail" if problems else "pass", "; ".join(problems),
                           {"accepted_tokens": accepted, "errors": errors, "notes": notes})

    def c07_oom(self) -> CheckResult:
        name = "an oversized configuration that fails to start is classified as out of memory at start-up"
        model = self.models["small"]
        cfg = self.base().model_copy(update={**OVERSIZED, "ctx": model.arch.max_position_embeddings})
        with self.launched(cfg, model, "oversized") as up:
            if up.ready:
                return CheckResult("C07", name, "skip", f"{cfg.key()} started on this card: nothing to classify")
            error = up.error or ""
            if FREE_MEMORY_CHECK in up.log():
                return CheckResult("C07", name, "skip", "vLLM's free-memory check refused it before loading: the "
                                   "card was not idle enough for the fraction asked", {"error": error, "key": cfg.key()})
        ok = error.startswith("out of memory at start-up")
        return CheckResult("C07", name, "pass" if ok else "fail", "" if ok else f"classified as: {error[:200]}",
                           {"error": error, "key": cfg.key()})

    def c08_text_only(self) -> CheckResult:
        name = "a multimodal checkpoint's text-only launch starts and serves"
        if "multimodal" not in self.models:
            return CheckResult("C08", name, "skip", "no --multimodal-model given")
        refusal = self.backend.text_only_refusal()
        if refusal:
            return CheckResult("C08", name, "skip", refusal)
        model = self.models["multimodal"]
        sub = self._served(self.base("multimodal").model_copy(update={"text_only": True}), model, "text-only")
        return CheckResult("C08", name, "pass" if self._ok(sub) else "fail", "" if self._ok(sub) else str(
            sub.get("error") or sub.get("errors")), sub)

    def c09_tier(self) -> CheckResult:
        name = "a host-RAM KV-cache tier starts and a repeated request reports reuse"
        model = self.models["small"]
        cfg = self.base().model_copy(update={"kv_tier": "host:4"})
        refusal = self.backend.tier_refusal(cfg, model)
        if refusal:
            return CheckResult("C09", name, "skip", refusal)
        with self.launched(cfg, model, "tier", backend=self.reporting) as up:
            if not up.ready:
                return CheckResult("C09", name, "fail", f"did not start: {up.error}")
            counts = cached_tokens(self.http, up.base_url, model.spec.hf_id)
        ok = counts[-1] is not None and counts[-1] > 0
        return CheckResult("C09", name, "pass" if ok else "fail", "" if ok else f"cached tokens: {counts[-1]}",
                           {"cached_tokens": counts})

    def c10_hybrid(self) -> CheckResult:
        from rigtune.memlog import parse_log
        from rigtune.memory import estimate

        name = "the hybrid model starts at the planner's configuration; its pools are recorded beside the plan"
        if "hybrid" not in self.models:
            return CheckResult("C10", name, "skip", "no --hybrid-model given")
        model = self.models["hybrid"]
        cfg = self.base("hybrid")
        mm = self.backend.memory_model(self.hw)
        plan = estimate(self.hw, model, cfg, mm)
        hybrid = mm.hybrid_for(cfg, model.arch)
        data: Dict[str, Any] = {"key": cfg.key(), "planned": plan.model_dump(),
                                "planned_hybrid": asdict(hybrid) if hybrid is not None else None}
        with self.launched(cfg, model, "hybrid") as up:
            if not up.ready:
                return CheckResult("C10", name, "fail", f"did not start: {up.error}", data)
            errors = serve(self.http, up.base_url, model.spec.hf_id)
            data["measured"] = parse_log(self.engine, up.log()).model_dump()
        return CheckResult("C10", name, "fail" if errors else "pass", "; ".join(errors[:3]), data)

    def c11_fp8(self) -> CheckResult:
        name = "fp8 weights start and serve on this card"
        sub = self._served(self.base().model_copy(update={"quant": "fp8"}), self.models["small"], "fp8")
        return CheckResult("C11", name, "pass" if self._ok(sub) else "fail", "" if self._ok(sub) else str(
            sub.get("error") or sub.get("errors")), sub)

    def c12_client(self) -> CheckResult:
        name = "the load generator measures a fake engine faithfully at 64 and 128 users"
        if self.selfcheck is None:
            from rigtune import selfcheck

            run = selfcheck.run
        else:
            run = self.selfcheck
        reports = run([64, 128])
        missed = [r for r in reports if not r.passed]
        return CheckResult("C12", name, "fail" if missed else "pass",
                           "; ".join(f"{r.users} users: {', '.join(r.problems())}" for r in missed),
                           {"levels": [r.to_dict() for r in reports]})

    # ---- the run

    def checks(self) -> List[tuple]:
        return [("C01", self.c01_probe), ("C02", self.c02_c05_stock), ("C03", self.c03_flags),
                ("C04", self.c04_reuse), ("C06", self.c06_speculative), ("C07", self.c07_oom),
                ("C08", self.c08_text_only), ("C09", self.c09_tier), ("C10", self.c10_hybrid),
                ("C11", self.c11_fp8), ("C12", self.c12_client)]

    def run(self, progress: Optional[Callable[[CheckResult], None]] = None) -> List[CheckResult]:
        results: List[CheckResult] = []
        try:
            self._run(results, progress)
        finally:
            self.http.close()
        return results

    def _run(self, results: List[CheckResult], progress: Optional[Callable[[CheckResult], None]]) -> None:
        for cid, check in self.checks():
            if self.only is not None and cid not in self.only and not (cid == "C02" and "C05" in self.only):
                continue
            try:
                got = check()
            except Exception as exc:  # a check that raises is a failed check, never the end of the run
                logger.exception("%s raised", cid)
                got = CheckResult(cid, "(raised)", "fail", f"{type(exc).__name__}: {exc}"[:300])
            for r in got if isinstance(got, list) else [got]:
                results.append(r)
                if progress:
                    progress(r)


def certified(results: Sequence[CheckResult]) -> bool:
    """Each of C01 to C05 passed, or was skipped because it does not apply to the engine (C05 on SGLang, whose metrics
    rigtune does not read), and C07 did not fail (it is skipped when the card starts the oversized configuration)."""
    by = {r.id: r for r in results}

    def counts(r: Optional[CheckResult]) -> bool:
        return r is not None and (r.status == "pass" or (r.status == "skip" and r.data.get("applies") is False))

    return all(counts(by.get(c)) for c in CERTIFYING) and "C07" in by and by["C07"].status in ("pass", "skip")


def _scrubbed(value: Any) -> Any:
    """`value` with every absolute path in its strings cut to its last part (rigtune.bench.compare.scrub_text): an
    engine's start-up error quotes log lines, which can hold the folders it ran from."""
    from rigtune.bench.compare import scrub_text

    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrubbed(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrubbed(v) for k, v in value.items()}
    return value


def record(engine: str, version: str, hw: HardwareDescriptor, results: Sequence[CheckResult], facts: Any,
           models: Dict[str, str], now: Optional[datetime] = None) -> Dict[str, Any]:
    """The run as one record of rigtune/data/conformance/<engine>-<version>.json (rigtune.engines.Tested, whose
    extra fields it ignores), with every check's details. It names no folder: a model given as a local path is
    named by its last part, and every path in an error, a log line or a default is cut the same way."""
    from rigtune import __version__
    from rigtune.bench.compare import name_only

    libs = dict(getattr(facts, "libs", None) or (hw.backends.get(engine).libs if hw.backends.get(engine) else {}))
    when = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    named = {k: name_only(v) for k, v in models.items()}
    return {
        "engine": engine, "version": version, "card": hw.gpu.name if hw.gpu else "none", "date": when,
        "torch": libs.get("torch"), "transformers": libs.get("transformers"),
        "checks": {f"{r.id} {r.name}": r.status for r in results},
        "source": f"rigtune conformance (rigtune {__version__}) with {', '.join(f'{k} {v}' for k, v in named.items())}",
        "certified": certified(results),
        "details": _scrubbed({r.id: asdict(r) for r in results}),
        "libs": libs,
        "probe": _scrubbed(facts.model_dump(mode="json")) if facts is not None else None,
    }


def write(out_dir: Path, rec: Dict[str, Any]) -> Path:
    """<out_dir>/<engine>-<version>.json, as a list of one record; an earlier run's file is never written over."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{rec['engine']}-{rec['version']}"
    path, n = out_dir / f"{stem}.json", 2
    while path.exists():
        path, n = out_dir / f"{stem}-{n}.json", n + 1
    path.write_text(json.dumps([rec], indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
