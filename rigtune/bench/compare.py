"""`rigtune compare`: rigtune's calibrated pick vs stock defaults vs Ollama, one workload, one file.

Result files are the raw material for the benchmark matrix and `rigtune report`:

    benchmarks/results/<hardware_hash>__<model>__<workload>__<objective>.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import AfterValidator, AliasChoices, BaseModel, Field, model_serializer

from rigtune import __version__, paths
from rigtune.backends.base import BaseBackend
from rigtune.bench.references import OllamaReference, ollama_tag_for, reference_configs
from rigtune.bench.rows import ExtraRow
from rigtune.calibrate.objectives import CUSTOM, Constraints, parse_goal, resolve_goal
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
    gpu_state: List[Dict[str, Any]] = Field(default_factory=list)  # each GPU's power limit, temperature
    engine_logs: List[str] = Field(default_factory=list)  # the engine log files it wrote, by name


class AnswerCheck(BaseModel):
    """A row's answers graded on a labelled set against its engine's stock row (rigtune/bench/answers.py)."""

    eval_set: str
    sha256: str
    items: int
    tolerance: float  # the largest loss in accuracy allowed, as a fraction
    outcomes: str = ""  # per item: '1' right, '0' wrong, '-' unanswered
    correct: int = 0
    answered: int = 0
    reference: Optional[str] = None  # the row it was compared with
    lost: Optional[int] = None
    gained: Optional[int] = None
    paired: Optional[int] = None
    lower_bound: Optional[float] = None  # of its accuracy minus the reference's, 95% score interval
    noise_flips: Optional[int] = None  # items on which its reference's two passes disagreed: the floor
    noise_paired: Optional[int] = None  # items both of those passes answered
    passed: Optional[bool] = None  # None: not checked
    reason: str = ""


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
    # A custom goal's optimised metric at the scored level (objectives.Goal.reader); unset under a preset, and then
    # left out of the file, so a preset comparison is written exactly as before goals.
    scored_objective: Optional[float] = None
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
    answer_check: Optional[AnswerCheck] = None  # --check-answers only
    launch_env: Dict[str, str] = Field(default_factory=dict)  # beside launch_args, under --verdict only

    @model_serializer(mode="wrap")
    def _omit_unset_provenance(self, handler) -> Dict[str, Any]:
        return _drop(handler(self), ("kind", "source", "run_records", "answer_check", "launch_env",
                                     "scored_objective"))


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
    # What the comparison ran on (provenance()), under --verdict only.
    provenance: Optional[Dict[str, Any]] = None
    goal: Optional[str] = None  # objective "custom": the goal's text (Constraints.goal); else left out of the file

    @model_serializer(mode="wrap")
    def _omit_empty_skipped(self, handler) -> Dict[str, Any]:
        return _drop(handler(self), ("skipped", "incomplete", "provenance", "goal"))

    @property
    def rigtune_row(self) -> Optional[ComparisonRow]:
        return next((r for r in self.rows if r.label == "rigtune"), None)

    def default_rows(self) -> List[ComparisonRow]:
        """The rows that stand for not tuning: every row but rigtune's and the expert and tuner rows."""
        return [r for r in self.rows if r.label != "rigtune" and r.kind in (None, "stock")]

    def other_rows(self) -> List[ComparisonRow]:
        return [r for r in self.rows if r.label != "rigtune"]


def results_dir() -> Path:
    return Path(paths.env("RESULTS") or "benchmarks/results")


def results_path(hw_hash: str, spec: ModelSpec, workload: str, objective: str, out: Optional[Path] = None,
                 goal: Optional[str] = None) -> Path:
    """Where a comparison is written. A custom goal is named by what it optimises and a hash of its text,
    `custom-joules_per_token-1a2b3c4d`, so two goals never share a file and no `<` or `>` reaches a file name."""
    if objective == CUSTOM and goal is not None:
        objective = f"{CUSTOM}-{parse_goal(goal).optimise}-{hashlib.sha256(goal.encode()).hexdigest()[:8]}"
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
    if objective == CUSTOM:
        goal = resolve_goal(objective, cons)
        row.scored_objective = goal.reader(goal.optimise)(m)
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

    def answers(self, checks: Dict[str, "AnswerCheck"]) -> None:
        dumped = {k: json.loads(v.model_dump_json()) for k, v in checks.items()}
        self._write({"type": "answers", "checks": dumped})

    def end(self, runs: int) -> None:
        self._write({"type": "end", "runs": runs})


def with_levels(workload: Workload, levels: List[int]) -> Tuple[Workload, str]:
    """The workload measured at these numbers of users instead of its own, named `<name>-c<levels>` so its
    results never stand in for the workload's own, and a note saying so. A synthetic workload with fewer
    prompts than twice its busiest level gets that many (as the high-concurrency preset sends 256 for 128
    users), so the busiest level can keep every slot busy; a file is what it holds, and the note says when it
    is short."""
    from dataclasses import replace

    wanted = tuple(sorted(set(levels)))
    if not wanted or wanted[0] < 1:
        raise ValueError(f"levels must be numbers of users of at least 1, not {levels}")
    name = f"{workload.name}-c{'-'.join(map(str, wanted))}"
    changed = replace(workload, concurrencies=wanted, name=name)
    note = (f"measured at {', '.join(map(str, wanted))} users instead of the workload's own "
            f"{', '.join(map(str, workload.concurrencies))}")
    busiest = wanted[-1]
    if workload.source == "synthetic" and workload.n_prompts < 2 * busiest:
        changed = replace(changed, n_prompts=2 * busiest, prompts=[])  # generated afresh, as many as asked
        note += f", with {2 * busiest} prompts per level instead of {workload.n_prompts}"
    elif workload.source != "synthetic" and len(workload.prompts) < 2 * busiest:
        note += (f"; the prompt file holds {len(workload.prompts)} prompts, fewer than twice the busiest "
                 "level, so that level may not stay full")
    return changed, note


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _own_checkout() -> Optional[Callable[..., str]]:
    """git, run in rigtune's own checkout, when rigtune runs from one; None otherwise. The checkout must be
    rigtune's own (its top level holds rigtune/ and a pyproject.toml naming rigtune): an installed copy can
    sit inside some other repository, such as a home directory kept in git, whose state would mean nothing
    here."""
    here = Path(__file__).resolve().parent.parent  # the rigtune package directory

    def git(*args: str, text: bool = True) -> Any:
        out = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=text, timeout=10,
                             check=True).stdout
        return out.strip() if text else out

    try:
        top = Path(git("rev-parse", "--show-toplevel"))
        project = (top / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if here.parent.resolve() != top.resolve() or 'name = "rigtune"' not in project:
        return None
    return git


def rigtune_commit() -> Optional[str]:
    """The commit rigtune runs from, with "+dirty" for local changes, untracked files included, when it runs
    from a git checkout of rigtune itself (_own_checkout); None otherwise. A dirty tree is recorded, not
    refused: rigtune_diff_sha256 says which changes."""
    git = _own_checkout()
    if git is None:
        return None
    try:
        dirty = git("status", "--porcelain", "--untracked-files=normal")
        return git("rev-parse", "HEAD") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def rigtune_diff_sha256() -> Optional[str]:
    """For a dirty checkout of rigtune, the sha256 of its changes against the commit: the binary diff of
    tracked files, then each untracked file's path and contents (ignored files left out). The same changes
    give the same hash, so a verdict can be matched to the tree that produced it. None when clean or not a
    checkout of rigtune."""
    git = _own_checkout()
    if git is None:
        return None
    try:
        digest = hashlib.sha256(git("diff", "HEAD", "--binary", text=False))
        untracked = sorted(git("ls-files", "--others", "--exclude-standard").splitlines())
        top = Path(git("rev-parse", "--show-toplevel"))
        for name in untracked:
            digest.update(b"\0" + name.encode("utf-8") + b"\0" + (top / name).read_bytes())
        if not untracked and not git("diff", "HEAD", "--stat"):
            return None
        return digest.hexdigest()
    except (OSError, subprocess.SubprocessError):
        return None


_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")  # C:\ or C:/
_SHORT_FLAG = re.compile(r"^(-[A-Za-z])(.+)$")  # -o/home/me/out
# An absolute path inside free text (an error, a traceback): a Windows drive or share path, a home-relative
# one, or a POSIX path of two or more parts. API routes such as /v1/chat/completions are not paths here.
_TEXT_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"<>|*?,;()\[\]]*"
                        r"|\\\\[^\s'\"<>|*?,;()\[\]]+"
                        r"|~[\\/][^\s'\"<>|*?,;()\[\]]*"
                        r"|(?:^|(?<=[\s'\"(=\[:,]))/(?!v1/)(?:[^\s'\"<>|:,;()\[\]/]+/)+"
                        r"[^\s'\"<>|:,;()\[\]/]*")


def is_path(value: str) -> bool:
    """Whether `value` is an absolute or home-relative path on either kind of system, or a Windows path,
    whatever system this is and whether or not it exists. A relative path such as at/rows.json, which names
    no folder above the one it was run in, is not; nor is a Hub id such as Qwen/Qwen2.5-7B."""
    return value.startswith(("/", "~", "\\")) or bool(_DRIVE.match(value)) or "\\" in value


def last_part(value: str) -> str:
    """The last part of a path, split on either separator: what Path(...).name gives on its own system."""
    parts = [p for p in re.split(r"[\\/]+", value) if p]
    return parts[-1] if parts else value


def name_only(arg: str) -> str:
    """A command-line argument with any path in it cut to its last part: a path can carry a user name.
    `/home/me/at/rows.json`, `C:\\Users\\me\\rows.json`, `--rows=/home/me/rows.json` and `-o/home/me/out`
    become `rows.json`, `rows.json`, `--rows=rows.json` and `-oout`, on any system; a Hub id such as
    Qwen/Qwen2.5-7B and a relative path are kept."""
    if arg.startswith("--") and "=" in arg:
        flag, value = arg.split("=", 1)
        return f"{flag}={name_only(value)}"
    short = _SHORT_FLAG.match(arg)
    if short and is_path(short.group(2)):
        return short.group(1) + last_part(short.group(2))
    return last_part(arg) if is_path(arg) else arg


def scrub_text(text: str) -> str:
    """Free text, an engine's error or traceback, with every absolute path in it cut to `<path>/` and its last
    part, so the message stays readable without naming the folders it ran from."""
    return _TEXT_PATH.sub(lambda m: f"<path>/{last_part(m.group(0))}", text)


def scrub_row(row: "ComparisonRow") -> None:
    """A measured row as a scored comparison writes it: launch arguments cut to names, and paths cut out of
    its launch environment, its error and its requests' errors."""
    row.launch_args = [name_only(a) for a in row.launch_args]
    row.launch_env = {k: scrub_text(v) for k, v in row.launch_env.items()}
    if row.error:
        row.error = scrub_text(row.error)
    for m in [row.metrics, *row.metrics.by_concurrency.values()]:
        m.errors = [scrub_text(e) for e in m.errors]
def model_provenance(spec: ModelSpec) -> Dict[str, Any]:
    """The weights: the Hub id, the revision asked for, and the snapshot (commit) the local Hugging Face cache
    holds for it, read from the cache without the network; None when it is not cached there."""
    snapshot = None
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(spec.hf_id, "config.json", revision=spec.revision)
        parts = Path(found).parts if isinstance(found, str) else ()
        if "snapshots" in parts[:-1]:
            snapshot = parts[parts.index("snapshots") + 1]
    except Exception:  # an id that is a local folder, or no huggingface_hub: there is no snapshot to name
        snapshot = None
    return {"hf_id": name_only(spec.hf_id), "revision": spec.revision, "snapshot": snapshot}


def workload_provenance(workload: Workload) -> Dict[str, Any]:
    """What the prompts were and how they were sampled: name, source (a file's name only), seed, temperature
    and whether answers stop naturally, and how to get the prompts again."""
    from rigtune.calibrate.datasets import FILE_PREFIX

    source = workload.source
    if source.startswith(FILE_PREFIX):
        source = FILE_PREFIX + last_part(source[len(FILE_PREFIX):])
        again = "a copy of the file is beside the results"
    else:
        again = (f"rigtune.calibrate.datasets rebuilds them from `{source}` with seed {workload.seed} at the "
                 "commit above: synthetic prompts are generated, a named dataset is downloaded from the "
                 "Hugging Face Hub and sampled")
    return {"name": workload.name, "source": source, "seed": workload.seed,
            "temperature": workload.temperature, "natural_stop": workload.natural_stop, "prompts": again}


def copy_inputs(target: Path, files: Dict[str, Optional[Path]]) -> Dict[str, str]:
    """Copy each prompt file into `<name>.inputs/` beside the results `<name>.json`, so the comparison can be
    rerun on exactly those prompts; {role: the copy's path relative to the results' folder}."""
    import shutil

    folder = target.with_name(target.name.removesuffix(".json") + ".inputs")
    out = {}
    for role, path in files.items():
        if path is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        name = path.name if not (folder / path.name).exists() else f"{role}-{path.name}"
        shutil.copyfile(path, folder / name)
        out[role] = f"{folder.name}/{name}"
    return out


def gpu_state() -> List[Dict[str, Any]]:
    """Each NVIDIA GPU's power limit, temperature and SM clock now, from nvidia-smi; [] when it cannot be
    read. Recorded at the start of every run under compare --verdict: a card that starts a run hot or capped
    measures differently."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.limit,temperature.gpu,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    def number(x: str) -> Optional[float]:
        try:
            return float(x)
        except ValueError:  # "[N/A]" on a card that does not report it
            return None

    state = []
    for line in out.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 4:
            state.append({"index": int(number(fields[0]) or 0), "power_limit_w": number(fields[1]),
                          "temperature_c": number(fields[2]), "sm_clock_mhz": number(fields[3])})
    return state


class RequestLog:
    """Every request of every run, as the load generator measured it, in `<folder>/<label>.r<round>.jsonl`:
    one line per request with its level, whether it succeeded, its time to first token and end to end, its
    prompt and completion tokens, its finish reason and the first line of any error. The runner calls it
    after each level (run_trial's on_level); compare sets `current` before each run."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.current: Optional[Tuple[str, int]] = None

    def __call__(self, level: int, outcomes: list) -> None:
        if self.current is None:
            return
        label, rnd = self.current
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / f"{label}.r{rnd}.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            for i, o in enumerate(outcomes):
                error = scrub_text(o.error.splitlines()[0])[:200] if o.error else None
                line = {"level": level, "request": i, "ok": o.ok, "ttft_ms": round(o.ttft_s * 1000, 3),
                        "e2e_ms": round(o.duration_s * 1000, 3), "prompt_tokens": o.prompt_tokens,
                        "tokens": o.tokens, "finish_reason": getattr(o, "finish_reason", None),
                        "error": error}
                fh.write(json.dumps(line) + "\n")


def provenance(hw: HardwareDescriptor, files: Dict[str, Optional[Path]], *, model: Optional[ModelSpec] = None,
               workload: Optional[Workload] = None, options: Optional[Dict[str, Any]] = None,
               command: Optional[List[str]] = None, rows: Optional[List[Path]] = None, expert: bool = False,
               inputs: Optional[Dict[str, str]] = None, logs: Optional[str] = None) -> Dict[str, Any]:
    """What a comparison ran on, for its results and verdict: rigtune's version and commit (with the hash of
    any local changes), the OS and Python, each GPU's driver and CUDA, each available engine's version with
    the torch, transformers and FlashInfer it found beside it, and each prompt file's name and full sha256
    (names only: a path can carry a user name). Under `compare --verdict` also the weights' revision and
    cached snapshot, the command line (paths cut to file names) and every option that shapes the result,
    the workload's source, seed and temperature, the content and sha256 of each --rows file and of the expert
    rows' data, where the copied prompt files and the engine logs are. Every run's start and end, its place
    in its round, the GPUs' power limit and temperature at its start, and its engine logs are in its
    RunRecord; its requests are in <name>.requests/."""
    out: Dict[str, Any] = {
        "rigtune_version": __version__,
        "rigtune_commit": rigtune_commit(),
        "os": hw.os,
        "python": hw.python,
        "gpus": [{"name": g.name, "driver_version": g.driver_version, "cuda_version": g.cuda_version,
                  "vram_total_bytes": g.vram_total_bytes} for g in hw.gpus],
        "engines": {name: {"version": b.version, "libs": dict(sorted(b.libs.items()))}
                    for name, b in sorted(hw.backends.items()) if b.available},
        "files": {role: {"name": p.name, "sha256": file_sha256(p)}
                  for role, p in files.items() if p is not None},
        "launch_args_port": "launch_args show port 0; each run's engine got a free port in its place",
    }
    if out["rigtune_commit"] and out["rigtune_commit"].endswith("+dirty"):
        out["rigtune_diff_sha256"] = rigtune_diff_sha256()
    if model is not None:
        out["model"] = model_provenance(model)
    if workload is not None:
        out["workload"] = workload_provenance(workload)
    if command is not None:
        out["command"] = ["rigtune", *map(name_only, command[1:])]
    if options is not None:
        out["options"] = options
    if rows:
        out["rows_files"] = [{"name": p.name, "sha256": file_sha256(p),
                              "content": json.loads(p.read_text(encoding="utf-8"))} for p in rows]
    if expert:
        from importlib import resources

        text = resources.files("rigtune").joinpath("data", "expert_rows.json").read_text(encoding="utf-8")
        out["expert_rows"] = {"name": "expert_rows.json",
                              "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                              "content": json.loads(text)}
    if inputs:
        out["inputs"] = inputs
    if logs:
        out["engine_logs"] = f"{logs}, under rigtune's logs folder (rigtune.cache.logs_dir())"
    return out
def _logs(folder: Optional[Path]) -> set:
    try:
        return {p.name for p in folder.iterdir()} if folder is not None and folder.is_dir() else set()
    except OSError:
        return set()


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
                   for suffix in (".json", ".runs.jsonl", ".verdict.json", ".verdict.md", ".inputs",
                                  ".requests", ".answers.jsonl")):
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
        found = [r["checks"][p["label"]] for r in mine
                 if r.get("type") == "answers" and p["label"] in r["checks"]]
        row.answer_check = AnswerCheck.model_validate(found[-1]) if found else None
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
    grader: Optional[object] = None,
    answer_tolerance: float = 0.03,
    origin: Optional[Dict[str, Any]] = None,
    gpu_probe: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    requests_dir: Optional[Path] = None,
    bare_stock: bool = False,
    rotate: bool = False,
    levels: Optional[List[int]] = None,
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
    records each level's AnswerStats (answer lengths, finish reasons, repetition) in the metrics. `grader`
    (rigtune/bench/answers.py RowGrader) grades each row's answers on a labelled set while its engine is up,
    and each row records its AnswerCheck against its engine's stock row at `answer_tolerance`. `origin`
    (provenance()) is recorded in the results, and each row's launch environment beside its launch args.
    With `record_runs`, `gpu_probe` (gpu_state) is read at the start of every run and kept in its RunRecord
    with the names of the engine logs the run wrote to `log_dir`; `requests_dir` keeps every request of
    every run there (RequestLog). `bare_stock` adds an `<engine>-bare` row per vLLM or SGLang candidate: the
    engine launched with the model and port alone (references.BareStock). `rotate` starts each round one row
    later (round 2 with the second row, and so on), so that no row is always the one measured right after an
    engine was torn down, or first on a cold card; each RunRecord keeps its place in its round.
    `levels` measures every row at these numbers of users instead of the workload's own (with_levels)."""
    workload = workload or get_workload(profile.workload)
    measured_on = workload.name  # the prompts' name, for the held-out note, whatever the levels
    level_note = None
    if levels:
        workload, level_note = with_levels(workload, levels)
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
    kept_note = None
    if include:
        need = f"{profile.backend}-default" if grader is not None else None  # rigtune's answers' reference
        if need in refs and need not in include:
            kept_note = (f"{need} is measured although --include leaves it out: --check-answers grades "
                         "rigtune's answers against it, and without it they could not be checked")
        refs = {k: v for k, v in refs.items() if k in include or k == need}
    bare: Dict[str, Config] = {}
    if bare_stock:  # asked for by name, so --include does not filter them
        from rigtune.bench.references import BareStock, bare_configs

        for label, cfg in bare_configs(prepared).items():
            engine = cfg.backend.removesuffix("-bare")
            backends[cfg.backend], models[cfg.backend] = BareStock(reg[engine]), models[engine]
            refs[label] = bare[label] = cfg
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

    sink = RequestLog(requests_dir) if requests_dir is not None else None
    if runner is None:
        more: Dict[str, Any] = {"answers": True} if answers else {}  # built as before unless asked for more
        if grader is not None:
            more["gate"] = grader
        if sink is not None:
            more["on_level"] = sink
        runner = SubprocessTrialRunner(backends=backends, models=models, hw=hw, workload=workload,
                                       log_dir=log_dir, power=power, **more)

    result = ComparisonResult(
        rigtune_version=__version__,
        hardware_hash=hardware_hash(hw),
        gpu=hw.gpu.name if hw.gpu else None,
        cpu=hw.cpu.model_name,
        model_id=spec.hf_id if origin is None else name_only(spec.hf_id),  # a local folder by its name
        objective=objective,
        workload=workload.name,
        workload_spec=workload.spec(),
        ttft_ceiling_ms=cons.ttft_ceiling_ms,
        tpot_ceiling_ms=cons.tpot_ceiling_ms,
        skipped=skipped,
        provenance=origin,
        goal=cons.goal,
    )
    if origin is not None:  # with provenance a prompt file is named, not located: a path can carry a user
        source = str(result.workload_spec.get("source", ""))
        from rigtune.calibrate.datasets import FILE_PREFIX

        if source.startswith(FILE_PREFIX):
            result.workload_spec["source"] = FILE_PREFIX + last_part(source[len(FILE_PREFIX):])
    for label, row in added.items():
        article = "an" if row.kind == "expert" else "a"
        result.notes.append(f"{label} is {article} {row.kind} row: {row.source}")
        result.notes += row.notes
        if row.changes_numerics and grader is None:
            result.notes.append(f"{label} changes what the model computes (a quantized KV cache): this "
                                "comparison measures its speed, not its answers")
        busiest = max(workload.concurrencies)
        if row.acts_up_to_users is not None and busiest > row.acts_up_to_users:
            result.notes.append(f"{label} is measured at up to {busiest} users, above "
                                f"{row.acts_up_to_users}: {row.acts_note}")
    for s in skipped:
        result.notes.append(f"{s.label} ({s.kind}) was not measured: {s.reason}")
    if kept_note:
        result.notes.append(kept_note)

    if profile.workload and profile.workload != measured_on:
        result.calibrated_on = profile.workload
        result.notes.append(f"rigtune's pick was calibrated on `{profile.workload}`; every row here was measured on "
                            f"`{measured_on}`, which calibration never saw")
    if level_note:
        result.notes.append(level_note)

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
        if label in bare:
            from rigtune.bench.references import BARE_KEY

            row.kind, row.config_key = "stock", BARE_KEY
        if progress:
            progress(label, None)
        try:
            launch = backend.launch_spec(cfg, models[cfg.backend], 0)
            from rigtune.pairing import redact_paths
            row.launch_args = [redact_paths(arg, models[cfg.backend].spec_paths) for arg in launch.args]
            if origin is not None:  # with provenance, the environment each engine was launched with too
                row.launch_env = dict(launch.env)
        except Exception:
            pass
        tr = runner.run(cfg, f"compare:{label}")
        from rigtune.pairing import redact_trial
        tr = redact_trial(tr, models[cfg.backend].spec_paths)
        spare = added[label].fallback if label in added else None
        if spare is not None and first and not tr.launched:
            swapped[label] = spare
            error = (tr.error or "no error recorded").splitlines()[0]
            error = (scrub_text(error) if origin is not None else error)[:200]
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
    if sink is not None:
        pick_more["on_level"] = sink
    pick_key = profile.config.key()  # the key rigtune's row is graded under
    if profile.disagg is not None:
        pick_key = profile.disagg.key()
    elif profile.replicas > 1:
        pick_key = f"{profile.config.key()} x{profile.replicas}"
    if grader is not None:
        from rigtune.bench.answers import KeyedGate, reference_label

        # The stock rows answer twice, to show their own noise; a disaggregated or replicas pick answers
        # through its router or balancer, graded under its row's key.
        grader.twice |= {cfg.key() for label, cfg in refs.items() if label == reference_label(cfg.backend)}
        if profile.disagg is not None or profile.replicas > 1:
            pick_more["gate"] = KeyedGate(grader, pick_key)
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
                    ("rigtune" if label == "rigtune" else
                     "stock" if label.endswith(("-default", "-bare")) else "tuner")}
                   for label, _ in measurers]
        log = RunsLog(runs_path, result, profile, planned, max(1, repeats))
    for repeat in range(1, max(1, repeats) + 1):
        shift = (repeat - 1) % len(measurers) if rotate else 0
        for position, (label, measure) in enumerate(measurers[shift:] + measurers[:shift], 1):
            if sink is not None:
                sink.current = (label, repeat)
            state = gpu_probe() if gpu_probe is not None and record_runs else []
            logs_before = _logs(log_dir) if record_runs else set()
            started = time.time()
            runs[label].append(measure())
            if origin is not None:  # a scored comparison names files, never the folders they are in
                scrub_row(runs[label][-1])
            record = run_record(runs[label][-1], repeat, position, started, time.time())
            record.gpu_state = state
            record.engine_logs = sorted(_logs(log_dir) - logs_before) if record_runs else []
            if record_runs:
                records[label].append(record)
            if log is not None:
                log.run(label, record, runs[label][-1])
    checks: Dict[str, AnswerCheck] = {}
    if grader is not None:
        from rigtune.bench.answers import check

        keys = {label: runs[label][0].config.key() for label, _ in measurers}
        keys["rigtune"] = pick_key
        checks = {label: check(label, runs[label][0].config.backend, keys[label], grader, keys,
                               answer_tolerance) for label, _ in measurers}
    if log is not None:
        if checks:
            log.answers(checks)
        log.end(sum(map(len, runs.values())))
    for label, _ in measurers:
        result.rows.append(combine_runs(runs[label]))
        result.rows[-1].run_records = records[label]
        result.rows[-1].answer_check = checks.get(label)
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
                                result.objective, out, goal=result.goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _f(x: Optional[float], nd: int = 0) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(result: ComparisonResult) -> str:
    limits = (f"(TTFT ≤ {result.ttft_ceiling_ms:.0f} ms"
              f"{f', TPOT ≤ {result.tpot_ceiling_ms:.0f} ms' if result.tpot_ceiling_ms else ''})")
    if result.goal is not None:  # a custom goal names its own limits; the TTFT field is only the workload's
        limits = f"`{result.goal}`" + (f" (TPOT ≤ {result.tpot_ceiling_ms:.0f} ms)" if result.tpot_ceiling_ms
                                       else "")
    lines = [
        f"**{result.gpu or result.cpu}** · {result.model_id} · workload `{result.workload}` · objective "
        f"`{result.objective}` {limits}",
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
