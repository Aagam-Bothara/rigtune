"""`rigtune compare`: rigtune's calibrated pick vs stock defaults vs Ollama, one workload, one file.

Result files are the raw material for the benchmark matrix and `rigtune report`:

    benchmarks/results/<hardware_hash>__<model>__<workload>__<objective>.json
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import AfterValidator, AliasChoices, BaseModel, Field, model_serializer

from rigtune import __version__, paths
from rigtune.backends.base import BaseBackend
from rigtune.bench.references import OllamaReference, ollama_tag_for, reference_configs
from rigtune.bench.rows import ExtraRow
from rigtune.calibrate.objectives import Constraints
from rigtune.calibrate.search import SubprocessTrialRunner, TrialRunner
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.hardware import hardware_hash
from rigtune.models import (
    AnswerStats,
    Config,
    HardwareDescriptor,
    MemoryObservation,
    ModelSpec,
    PreparedModel,
    Profile,
    TrialMetrics,
)

logger = logging.getLogger(__name__)


SELF = "rigtune"


def self_label(label: str) -> str:
    """A row's label, reading this tool's former name as its current one.

    Results recorded before the rename label their own row "polyserve". They are the evidence behind the published
    tables, so they are read as recorded rather than rewritten.
    """
    return SELF if label == paths.FORMER else label


class RunRecord(BaseModel):
    """One run of one row, as scored: whether it met the limits, at which level, with what tail. A row stands
    for its median run, so these are what the verdict needs to count runs that met the limits, pair rounds
    and see a run that missed them. Kept under `compare --verdict` only."""

    round: int  # 1-based repeat
    position: int  # 1-based place in its round's measuring order
    ok: bool
    meets_slo: Optional[bool] = None
    scored_concurrency: Optional[int] = None
    scored_tok_s: Optional[float] = None
    scored_ttft_p95_ms: Optional[float] = None
    scored_tpot_ms: Optional[float] = None
    error: Optional[str] = None  # the first line, for a run that failed
    started_at: Optional[float] = None  # wall clock, seconds since the epoch
    ended_at: Optional[float] = None
    answers: Optional[AnswerStats] = None  # the scored level's answers, when compare recorded them
    level_answers: Dict[str, AnswerStats] = Field(default_factory=dict)  # every level's, by number of users


class ComparisonRow(BaseModel):
    label: Annotated[str, AfterValidator(self_label)]  # "rigtune" | "vllm-default" | "sglang-default" | "llamacpp-cuda-default" | "ollama-default"
    runtime: str  # backend name
    runtime_version: Optional[str] = None
    config: Config
    config_key: str
    launch_args: List[str] = Field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None
    metrics: TrialMetrics = Field(default_factory=TrialMetrics)
    memory: Optional[MemoryObservation] = None
    # Objective-relevant view: the concurrency level the objective would run this row at.
    scored_concurrency: Optional[int] = None
    scored_tok_s: Optional[float] = None
    scored_ttft_ms: Optional[float] = None
    scored_ttft_p95_ms: Optional[float] = None
    scored_tpot_ms: Optional[float] = None
    scored_joules_per_token: Optional[float] = None
    meets_slo: Optional[bool] = None
    calibration_seconds: float = 0.0
    calibration_trials: int = 0
    # --repeats: the row stands for its median run; every run's scored tok/s is kept (None for a failed run).
    repeats: int = 1
    runs_tok_s: List[Optional[float]] = Field(default_factory=list)
    tok_s_spread_pct: Optional[float] = None  # (max - min) / median over the runs that succeeded
    # Rows added with --expert or --rows (rigtune/bench/rows.py): "expert" or "tuner", and where the
    # settings came from. Unset for rigtune's own row and the stock rows, and then left out of the file, so a
    # comparison without added rows is written exactly as before they existed.
    kind: Optional[str] = None
    source: Optional[str] = None
    run_records: List[RunRecord] = Field(default_factory=list)  # every run, under --verdict; else not written

    @model_serializer(mode="wrap")
    def _omit_unset_provenance(self, handler) -> Dict[str, Any]:
        return _drop(handler(self), ("kind", "source", "run_records"))


class SkippedRow(BaseModel):
    """An added row that was not measured, and why (its engine is not compared here, the card's default
    already does what it sets, rigtune knows it fails, or the tool behind it produced no pick). The verdict
    reports it as not applicable."""

    label: str
    kind: str
    source: str = ""
    reason: str


def _drop(data: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    for k in keys:
        if data.get(k) in (None, [], {}):
            data.pop(k, None)
    return data


class ComparisonResult(BaseModel):
    rigtune_version: str = Field(validation_alias=AliasChoices("rigtune_version", "polyserve_version"))
    hardware_hash: str
    gpu: Optional[str]
    cpu: str
    model_id: str
    objective: str
    workload: str
    workload_spec: Dict[str, Any]
    ttft_ceiling_ms: float
    tpot_ceiling_ms: Optional[float] = None
    rows: List[ComparisonRow] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    notes: List[str] = Field(default_factory=list)
    # The workload rigtune's pick was calibrated on, when the rows were measured on another (held out).
    calibrated_on: Optional[str] = None
    skipped: List[SkippedRow] = Field(default_factory=list)  # left out of the file when empty
    # Set only by read_runs, for a comparison that stopped before every row had all its repeats.
    incomplete: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_empty_skipped(self, handler) -> Dict[str, Any]:
        return _drop(handler(self), ("skipped", "incomplete"))

    @property
    def rigtune_row(self) -> Optional[ComparisonRow]:
        return next((r for r in self.rows if r.label == "rigtune"), None)

    def default_rows(self) -> List[ComparisonRow]:
        """The rows that stand for not tuning: every row but rigtune's and the expert and tuner rows."""
        return [r for r in self.rows if r.label != "rigtune" and r.kind is None]

    def other_rows(self) -> List[ComparisonRow]:
        return [r for r in self.rows if r.label != "rigtune"]


def results_dir() -> Path:
    return Path(paths.env("RESULTS") or "benchmarks/results")


def results_path(hw_hash: str, spec: ModelSpec, workload: str, objective: str, out: Optional[Path] = None) -> Path:
    return (out or results_dir()) / f"{hw_hash}__{spec.safe_id}__{workload}__{objective}.json"


def _score_row(row: ComparisonRow, objective: str, cons: Constraints) -> None:
    """Fill the scored_* fields from the concurrency level the objective would pick."""
    from rigtune.calibrate.objectives import rank
    from rigtune.models import TrialResult

    if not row.ok:
        return
    tr = TrialResult(config=row.config, stage="compare", metrics=row.metrics)
    ranked = rank([tr], objective, cons)
    if not ranked:
        return
    m = ranked[0].metrics
    row.scored_concurrency = m.concurrency
    row.scored_tok_s = m.tok_s
    row.scored_ttft_ms = m.ttft_ms
    row.scored_ttft_p95_ms = m.ttft_p95_ms
    row.scored_tpot_ms = m.tpot_ms
    row.scored_joules_per_token = m.joules_per_token
    row.meets_slo = ranked[0].feasible


def run_record(row: ComparisonRow, repeat: int, position: int, started: Optional[float] = None,
               ended: Optional[float] = None) -> RunRecord:
    """The RunRecord of one measured run."""
    error = row.error.splitlines()[0][:200] if not row.ok and row.error else None
    level = row.metrics.by_concurrency.get(str(row.scored_concurrency)) if row.scored_concurrency else None
    return RunRecord(round=repeat, position=position, ok=row.ok, meets_slo=row.meets_slo if row.ok else None,
                     scored_concurrency=row.scored_concurrency,
                     scored_tok_s=row.scored_tok_s if row.ok else None,
                     scored_ttft_p95_ms=row.scored_ttft_p95_ms, scored_tpot_ms=row.scored_tpot_ms,
                     error=error, started_at=started, ended_at=ended,
                     answers=level.answers if level is not None else row.metrics.answers,
                     level_answers={k: m.answers for k, m in row.metrics.by_concurrency.items()
                                    if m.answers is not None})


def combine_runs(rows: List[ComparisonRow]) -> ComparisonRow:
    """One row from repeated measurements of it. The run with the median scored tok/s stands for the row,
    so every metric shown comes from one real run rather than a blend; each run's tok/s is kept."""
    if len(rows) == 1:
        return rows[0]
    ok = sorted((r for r in rows if r.ok and r.scored_tok_s is not None), key=lambda r: r.scored_tok_s)
    row = (ok[(len(ok) - 1) // 2] if ok else rows[0]).model_copy(deep=True)
    row.repeats = len(rows)
    row.runs_tok_s = [r.scored_tok_s if r.ok else None for r in rows]
    if len(ok) >= 2 and row.scored_tok_s:
        row.tok_s_spread_pct = (ok[-1].scored_tok_s - ok[0].scored_tok_s) / row.scored_tok_s * 100
    return row


class RunsLog:
    """Every run of a comparison, appended to a JSON Lines file the moment it is measured, so that a crash
    keeps what was done. The file is only appended to, each record flushed and fsync'd: a header with the
    comparison's settings, the rows it plans to measure and how many times, one record per run, and an end
    record once every row is done. Every record carries the comparison's run id, so a reader takes only the
    last comparison's records, whatever else the file holds. fsync guards against a crash of the process or
    the machine, not against losing the machine: on 14 November 2026 a pod deleted itself with its disk, and
    only copying the files off it would have kept them."""

    def __init__(self, path: Path, result: "ComparisonResult", profile: Profile,
                 planned: List[Dict[str, str]], repeats: int):
        self.path, self.run_id = path, uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            with path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b"\n"
            if torn:  # an earlier comparison stopped mid-line: start on a line of our own
                self._append("\n")
        self._write({"type": "comparison", "planned": planned, "repeats": repeats,
                     "result": json.loads(result.model_dump_json()),
                     "calibration_seconds": profile.calibration_seconds,
                     "calibration_trials": profile.calibration_trials})

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    def _write(self, record: Dict[str, Any]) -> None:
        self._append(json.dumps({"run_id": self.run_id, **record}) + "\n")

    def run(self, label: str, record: RunRecord, row: ComparisonRow) -> None:
        self._write({"type": "run", "label": label, "record": json.loads(record.model_dump_json()),
                     "row": json.loads(row.model_dump_json())})

    def end(self, runs: int) -> None:
        self._write({"type": "end", "runs": runs})


def runs_path_for(results: Path) -> Path:
    """<name>.runs.jsonl beside the results file <name>.json."""
    return results.with_name(results.name.removesuffix(".json") + ".runs.jsonl")


def free_results_path(path: Path) -> Path:
    """`path`, or `<name>-2.json`, `-3`, ... when it or its runs file or verdict already exists: an evidence
    run never writes over another's files."""
    stem, n = path.name.removesuffix(".json"), 1
    while True:
        name = stem if n == 1 else f"{stem}-{n}"
        if not any((path.parent / f"{name}{suffix}").exists()
                   for suffix in (".json", ".runs.jsonl", ".verdict.json", ".verdict.md")):
            return path.parent / f"{name}.json"
        n += 1


def read_runs(path: Path) -> "ComparisonResult":
    """The last comparison a runs file holds, each row combined from its runs as `compare` combines them, with
    its RunRecords. A planned row with no run is listed in `skipped` as not measured, and `incomplete` says
    what is missing unless the end record is there and every row has all its repeats."""
    records, torn = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            torn += 1  # a line cut off as the process or machine went away
    headers = [r for r in records if r.get("type") == "comparison" and r.get("run_id")]
    if not headers:
        raise ValueError(f"{path}: no comparison header")
    header = headers[-1]
    mine = [r for r in records if r.get("run_id") == header["run_id"]]
    result = ComparisonResult.model_validate(header["result"])
    runs: Dict[str, List[Tuple[RunRecord, ComparisonRow]]] = {}
    for r in mine:
        if r.get("type") == "run":
            runs.setdefault(r["label"], []).append((RunRecord.model_validate(r["record"]),
                                                    ComparisonRow.model_validate(r["row"])))
    planned = header.get("planned") or [{"label": label, "kind": ""} for label in runs]
    repeats = int(header.get("repeats") or 1)
    for p in planned:
        got = runs.get(p["label"])
        if not got:
            result.skipped.append(SkippedRow(label=p["label"], kind=p["kind"] or "tuner",
                                             reason="the comparison stopped before it reached this row"))
            continue
        row = combine_runs([r for _, r in got])
        row.run_records = [rec for rec, _ in got]
        result.rows.append(row)
    if result.rigtune_row is not None:
        result.rigtune_row.calibration_seconds = header.get("calibration_seconds", 0.0)
        result.rigtune_row.calibration_trials = header.get("calibration_trials", 0)
    short = [f"{p['label']} {len(runs.get(p['label'], []))} of {repeats}" for p in planned
             if len(runs.get(p["label"], [])) < repeats]
    ended = any(r.get("type") == "end" for r in mine)
    if short or not ended:
        result.incomplete = ("the comparison stopped before it finished"
                             + (f": runs measured, {', '.join(short)}" if short else ", after its last run"))
    note = f"rebuilt from {path.name}: {sum(map(len, runs.values()))} runs of {len(runs)} rows"
    if torn:
        note += f"; {torn} unreadable line{'s' if torn > 1 else ''} skipped"
    result.notes.append(note)
    return result


def noise_notes(result: "ComparisonResult") -> List[str]:
    """Other rows whose runs overlap rigtune's: the difference between them is within run-to-run noise."""
    ps = result.rigtune_row
    mine = [v for v in (ps.runs_tok_s if ps else []) if v is not None]
    if len(mine) < 2:
        return []
    out = []
    for r in result.other_rows():
        theirs = [v for v in r.runs_tok_s if v is not None]
        if len(theirs) >= 2 and min(mine) <= max(theirs) and min(theirs) <= max(mine):
            out.append(f"{r.label}: its runs ({min(theirs):.0f}-{max(theirs):.0f} tok/s) overlap rigtune's "
                       f"({min(mine):.0f}-{max(mine):.0f}), so the difference is within run-to-run noise")
    return out


def _run_disagg_row(profile: Profile, backends: Dict[str, BaseBackend], models: Dict[str, PreparedModel],
                    hw: HardwareDescriptor, workload: Workload, log_dir: Optional[Path], power: Optional[object],
                    disagg_runner: Optional[object], objective: str, cons: Constraints,
                    progress: Optional[Callable[[str, Optional[ComparisonRow]], None]],
                    more: Optional[Dict[str, Any]] = None) -> ComparisonRow:
    """Measure a disaggregated profile end to end: both engines plus the router. `more` goes to the runner
    rigtune builds (answers=True records answer statistics)."""
    from rigtune.disagg import DisaggTrialRunner

    spec = profile.disagg
    backend = backends[profile.backend]
    power_for_gpu = (lambda i: power) if power is not None else None
    runner = disagg_runner or DisaggTrialRunner(backend, models[profile.backend], hw, workload, log_dir=log_dir,
                                                power_for_gpu=power_for_gpu, **(more or {}))
    row = ComparisonRow(label="rigtune", runtime=f"{profile.backend}-disaggregated",
                        runtime_version=backend.version(hw), config=spec.decode, config_key=spec.key())
    if progress:
        progress("rigtune", None)
    tr = runner.run(spec, "compare:rigtune")  # type: ignore[attr-defined]
    row.ok, row.error, row.metrics = tr.ok, tr.error, tr.metrics
    _score_row(row, objective, cons)
    if progress:
        progress("rigtune", row)
    return row


def _run_replica_row(profile: Profile, backends: Dict[str, BaseBackend], models: Dict[str, PreparedModel],
                     hw: HardwareDescriptor, workload: Workload, log_dir: Optional[Path],
                     replica_runner: Optional[object], objective: str, cons: Constraints,
                     progress: Optional[Callable[[str, Optional[ComparisonRow]], None]],
                     more: Optional[Dict[str, Any]] = None) -> ComparisonRow:
    """Measure a replicas profile end to end through the load balancer. `more` as for _run_disagg_row."""
    from rigtune.layout import LayoutTrialRunner

    backend = backends[profile.backend]
    gpus = [g.index for g in hw.gpus if g.vendor == "nvidia"][: profile.replicas]
    runner = replica_runner or LayoutTrialRunner(backend, models[profile.backend], hw, workload,
                                                 log_dir=log_dir, **(more or {}))
    row = ComparisonRow(label="rigtune", runtime=f"{profile.backend}-x{profile.replicas}",
                        runtime_version=backend.version(hw), config=profile.config,
                        config_key=f"{profile.config.key()} x{profile.replicas}")
    if progress:
        progress("rigtune", None)
    tr = runner.run_replicas(profile.config, gpus, "compare:rigtune")  # type: ignore[attr-defined]
    row.ok, row.error, row.metrics = tr.ok, tr.error, tr.metrics
    _score_row(row, objective, cons)
    if progress:
        progress("rigtune", row)
    return row


def compare(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    profile: Profile,
    prepared: Dict[str, PreparedModel],
    reg: Dict[str, BaseBackend],
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    runner: Optional[TrialRunner] = None,
    ollama_tag: Optional[str] = None,
    include: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    progress: Optional[Callable[[str, Optional[ComparisonRow]], None]] = None,
    power: Optional[object] = None,
    disagg_runner: Optional[object] = None,
    replica_runner: Optional[object] = None,
    repeats: int = 1,
    extra: Optional[Dict[str, Union[Config, ExtraRow]]] = None,
    record_runs: bool = False,
    runs_path: Optional[Path] = None,
    answers: bool = False,
) -> ComparisonResult:
    """Measure rigtune's winner and every reference config under the same workload. `extra` adds rows under
    their own labels (other searches' picks, say), measured like every other row and never filtered by `include`.
    An ExtraRow (rigtune/bench/rows.py: expert rows, other tools' picks) also records its kind and source in
    its row and the notes; one marked skipped, or whose engine is not compared here, is listed in `skipped`
    instead of measured.

    With repeats > 1 every row is measured that many times, round-robin (all rows once, then all again),
    so drift over the session (temperature, a noisy neighbour) falls on every row alike. `record_runs` keeps
    every run's RunRecord on its row (limit status, level, tail, place in its round, when) for the verdict.
    With `runs_path`, every run is also appended there as soon as it is measured (RunsLog). `answers`
    records each level's AnswerStats (answer lengths, finish reasons, repetition) in the metrics."""
    workload = workload or get_workload(profile.workload)
    cons = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                      tpot_ceiling_ms=workload.tpot_ceiling_ms)
    objective = profile.objective

    backends: Dict[str, BaseBackend] = dict(reg)
    models: Dict[str, PreparedModel] = dict(prepared)
    if profile.prepared is not None:
        models[profile.backend] = profile.prepared

    # References: stock defaults for each candidate backend, plus Ollama when available.
    refs: Dict[str, Config] = reference_configs(hw, prepared, reg)
    tag = ollama_tag or ollama_tag_for(spec.hf_id)
    if tag:
        ollama = OllamaReference(tag)
        if ollama.available(hw):
            backends["ollama"] = ollama
            try:
                models["ollama"] = ollama.prepare(spec, hw)
                refs["ollama-default"] = ollama.default_config(hw, models["ollama"])
            except Exception as exc:
                logger.warning("ollama reference skipped: %s", exc)
        else:
            logger.info("ollama binary not found; skipping the ollama-default row")
    if include:
        refs = {k: v for k, v in refs.items() if k in include}
    added: Dict[str, ExtraRow] = {}
    skipped: List[SkippedRow] = []
    for label, row in (extra or {}).items():
        if not isinstance(row, ExtraRow):
            refs[label] = row
            continue
        reason = row.skipped
        if reason is None and row.config.backend not in models:
            reason = f"{row.config.backend} is not one of this comparison's engines"
        if reason is not None:
            skipped.append(SkippedRow(label=label, kind=row.kind, source=row.source, reason=reason))
            continue
        refs[label] = row.config
        added[label] = row

    if runner is None:
        more = {"answers": True} if answers else {}  # the runner is built as before unless answers are kept
        runner = SubprocessTrialRunner(backends=backends, models=models, hw=hw, workload=workload,
                                       log_dir=log_dir, power=power, **more)

    result = ComparisonResult(
        rigtune_version=__version__,
        hardware_hash=hardware_hash(hw),
        gpu=hw.gpu.name if hw.gpu else None,
        cpu=hw.cpu.model_name,
        model_id=spec.hf_id,
        objective=objective,
        workload=workload.name,
        workload_spec=workload.spec(),
        ttft_ceiling_ms=cons.ttft_ceiling_ms,
        tpot_ceiling_ms=cons.tpot_ceiling_ms,
        skipped=skipped,
    )
    for label, row in added.items():
        article = "an" if row.kind == "expert" else "a"
        result.notes.append(f"{label} is {article} {row.kind} row: {row.source}")
        result.notes += row.notes
        if row.changes_numerics:
            result.notes.append(f"{label} changes what the model computes (a quantized KV cache): this "
                                "comparison measures its speed, not its answers")
        busiest = max(workload.concurrencies)
        if row.acts_up_to_users is not None and busiest > row.acts_up_to_users:
            result.notes.append(f"{label} is measured at up to {busiest} users, above "
                                f"{row.acts_up_to_users}: {row.acts_note}")
    for s in skipped:
        result.notes.append(f"{s.label} ({s.kind}) was not measured: {s.reason}")

    if profile.workload and profile.workload != workload.name:
        result.calibrated_on = profile.workload
        result.notes.append(f"rigtune's pick was calibrated on `{profile.workload}`; every row here was measured on "
                            f"`{workload.name}`, which calibration never saw")

    swapped: Dict[str, Config] = {}  # added rows measured with their fallback: the first run failed to start
    seen: set = set()

    def run_row(label: str, cfg: Config) -> ComparisonRow:
        cfg = swapped.get(label, cfg)
        first = label not in seen
        seen.add(label)
        backend = backends[cfg.backend]
        row = ComparisonRow(label=label, runtime=cfg.backend, runtime_version=backend.version(hw), config=cfg,
                            config_key=cfg.key())
        if label in added:
            row.kind, row.source = added[label].kind, added[label].source
        if progress:
            progress(label, None)
        try:
            row.launch_args = backend.launch_spec(cfg, models[cfg.backend], 0).args
        except Exception:
            pass
        tr = runner.run(cfg, f"compare:{label}")
        spare = added[label].fallback if label in added else None
        if spare is not None and first and not tr.launched:
            swapped[label] = spare
            error = (tr.error or "no error recorded").splitlines()[0][:200]
            result.notes.append(f"{added[label].fallback_note} (it failed with: {error})")
            return run_row(label, spare)
        row.ok, row.error, row.metrics, row.memory = tr.ok, tr.error, tr.metrics, tr.memory
        _score_row(row, objective, cons)
        if progress:
            progress(label, row)
        return row

    # rigtune's winner, re-measured now so it faces the same conditions as the references.
    measurers: List[Tuple[str, Callable[[], ComparisonRow]]] = []
    pick_more: Dict[str, Any] = {"answers": True} if answers else {}  # for a disaggregated or replicas pick
    if profile.disagg is not None:
        measurers.append(("rigtune", lambda: _run_disagg_row(profile, backends, models, hw, workload, log_dir, power,
                                                               disagg_runner, objective, cons, progress,
                                                               pick_more)))
    elif profile.replicas > 1:
        measurers.append(("rigtune", lambda: _run_replica_row(profile, backends, models, hw, workload, log_dir,
                                                                replica_runner, objective, cons, progress,
                                                                pick_more)))
        result.notes.append(f"rigtune serves {profile.replicas} replicas on {profile.replicas} GPUs; the stock rows "
                            "use one GPU, so this row is not a like-for-like throughput comparison")
    else:
        measurers.append(("rigtune", lambda: run_row("rigtune", profile.config)))
    for label, cfg in refs.items():
        if cfg.backend in models:
            measurers.append((label, lambda label=label, cfg=cfg: run_row(label, cfg)))

    runs: Dict[str, List[ComparisonRow]] = {label: [] for label, _ in measurers}
    records: Dict[str, List[RunRecord]] = {label: [] for label, _ in measurers}
    log = None
    if runs_path is not None:
        planned = [{"label": label, "kind": added[label].kind if label in added else
                    ("rigtune" if label == "rigtune" else "stock" if label.endswith("-default") else "tuner")}
                   for label, _ in measurers]
        log = RunsLog(runs_path, result, profile, planned, max(1, repeats))
    for repeat in range(1, max(1, repeats) + 1):
        for position, (label, measure) in enumerate(measurers, 1):
            started = time.time()
            runs[label].append(measure())
            record = run_record(runs[label][-1], repeat, position, started, time.time())
            if record_runs:
                records[label].append(record)
            if log is not None:
                log.run(label, record, runs[label][-1])
    if log is not None:
        log.end(sum(map(len, runs.values())))
    for label, _ in measurers:
        result.rows.append(combine_runs(runs[label]))
        result.rows[-1].run_records = records[label]
    ps = result.rigtune_row
    ps.calibration_seconds = profile.calibration_seconds
    ps.calibration_trials = profile.calibration_trials
    if repeats > 1:
        result.notes.append(f"every row was measured {repeats} times, interleaved; tok/s is the median run's, and the "
                            "spread is (max - min) / median")
        result.notes += noise_notes(result)

    ps_ok = ps.ok and ps.meets_slo
    if not ps_ok:
        result.notes.append("rigtune row failed or missed the SLO in this run; inspect before reporting")
    return result


def save(result: ComparisonResult, out: Optional[Path] = None, path: Optional[Path] = None) -> Path:
    """Write the results to `path`, or to results_path() in `out`, replacing what is there as always."""
    path = path or results_path(result.hardware_hash, ModelSpec(hf_id=result.model_id), result.workload,
                                result.objective, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _f(x: Optional[float], nd: int = 0) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(result: ComparisonResult) -> str:
    lines = [
        f"**{result.gpu or result.cpu}** · {result.model_id} · workload `{result.workload}` · objective "
        f"`{result.objective}` (TTFT ≤ {result.ttft_ceiling_ms:.0f} ms"
        f"{f', TPOT ≤ {result.tpot_ceiling_ms:.0f} ms' if result.tpot_ceiling_ms else ''})",
        "",
        "| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result.rows:
        name = f"**rigtune** → {r.config_key}" if r.label == "rigtune" else f"{r.label} ({r.config_key})"
        if not r.ok:
            lines.append(f"| {name} | failed: {(r.error or '').splitlines()[0][:60]} | | | | | | | | |")
            continue
        m = r.metrics
        calib = f"{r.calibration_seconds:.0f}s / {r.calibration_trials}" if r.label == "rigtune" else "-"
        mem = _f(m.peak_mem_mb / 1024, 1) if m.peak_mem_mb else "-"
        spread = (f" (median of {r.repeats}, spread {r.tok_s_spread_pct:.1f}%)"
                  if r.repeats > 1 and r.tok_s_spread_pct is not None else "")
        lines.append(
            f"| {name} | {_f(r.scored_tok_s)} @c{r.scored_concurrency}{spread} | {_f(r.scored_ttft_ms)} | "
            f"{_f(r.scored_ttft_p95_ms)} | {_f(r.scored_tpot_ms, 1)} | {mem} GB | {_f(m.power_w)} | "
            f"{_f(r.scored_joules_per_token, 3)} | {'yes' if r.meets_slo else 'no'} | {calib} |"
        )
    if result.notes:
        lines += [""] + [f"_{n}_" for n in result.notes]
    return "\n".join(lines)
