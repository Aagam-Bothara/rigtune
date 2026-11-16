"""What an installed engine offers, asked of the engine itself.

rigtune's feature checks read an engine's source text for a name (a flag's field, a method in a help string, a class),
in files at the paths one release keeps them. That breaks when a release moves a file: vLLM 0.33 moved its server
arguments to entrypoints/common/cli_args.py, and the check for --enable-prompt-tokens-details answered no until it was
told where to look; SGLang 0.5.23 moved its argument definitions to arg_groups/fields/, and its checks still answer
only because a tuple of field names elsewhere happens to list them. The engine's own Python can say what its parser
takes, which architectures its registry has and which speculative methods its config accepts, wherever it keeps them.

So rigtune runs a small script (rigtune/backends/probes/) in the Python the engine will be launched with, once per
install, and keeps the answers in <home>/engines/. A probe imports the engine, which takes seconds to tens of seconds
and could initialise CUDA, so it runs with no GPU visible (CUDA_VISIBLE_DEVICES=""), from an empty directory, and is
stopped after PROBE_TIMEOUT seconds; whatever it answered by then is kept. A probe that cannot run, or a question it
could not answer, leaves that answer to the checks rigtune made before it existed. What the answers are used for is
each backend's business; a probe says what exists, never what works (a measured failure stays a measured failure).

`RIGTUNE_ENGINE_PROBE=off` turns it off: every check then reads the engine's source as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from rigtune import paths

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 120.0
MARK = "RIGTUNE-PROBE "
PROBES = Path(__file__).parent / "probes"
SCRIPTS = {"vllm": "vllm_probe.py", "sglang": "sglang_probe.py"}
# What each probe is asked, in the order it answers. "done" closes a probe that ran to the end.
KEYS = {"vllm": ("version", "libs", "spec_methods", "spec_fields", "kv_dtypes", "kernel", "flags", "features",
                 "archs"),
        "sglang": ("version", "libs", "spec_methods", "flags", "request_fields", "env_vars", "arch_sets", "archs")}


class EngineFacts(BaseModel):
    """One engine install's answers. A field left None was not answered (`errors` says why); every check that would
    read it falls back to what rigtune did before the probe."""

    engine: str
    version: Optional[str] = None
    flags: Optional[Dict[str, Dict[str, Any]]] = None  # every option string -> {"choices": [...] | None, "default"}
    archs: Optional[List[str]] = None
    spec_methods: Optional[List[str]] = None  # the engine's own names: vLLM's methods, SGLang's algorithms
    spec_fields: Optional[List[str]] = None  # vLLM SpeculativeConfig fields; None = unanswered
    kv_dtypes: Optional[List[str]] = None  # vLLM
    # vLLM: the names its graph and compile settings take, one list per KERNEL_PARTS entry; None = unanswered
    kernel: Optional[Dict[str, List[str]]] = None
    request_fields: Optional[List[str]] = None  # SGLang: ChatCompletionRequest's fields
    env_vars: Optional[List[str]] = None  # SGLang: the variables its environ.py declares
    arch_sets: Optional[Dict[str, List[str]]] = None  # SGLang: architecture sets its memory rules test
    features: Optional[Dict[str, bool]] = None  # vLLM: the KV offloading tier's parts
    libs: Dict[str, str] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)  # key -> why it has no answer
    probed_at: Optional[str] = None
    seconds: Optional[float] = None
    # "probe" for a probe's own output; anything else says where hand-made answers came from (test fixtures).
    provenance: str = "probe"

    def answered(self, key: str) -> bool:
        if key == "spec_fields" and self.spec_fields == []:
            return self.provenance == "probe"
        return getattr(self, key, None) is not None

    def has_flag(self, option: str) -> Optional[bool]:
        """Whether the engine's server takes `option` (e.g. "--attention-backend"); None when the probe did not say."""
        return None if self.flags is None else option in self.flags

    def unanswered(self) -> List[str]:
        return [k for k in KEYS.get(self.engine, ()) if not self.answered(k) and k not in ("version", "libs")]


def script_path(engine: str) -> Path:
    return PROBES / SCRIPTS[engine]


def enabled() -> bool:
    return (paths.env("ENGINE_PROBE") or "").strip().lower() != "off"


# Answers that must not be empty: a release that renamed what the probe reads (a Literal turned into an Enum, a
# registry with another method) would otherwise report none, and an empty list would take every method or
# architecture away rather than leave the question to the source checks.
NON_EMPTY = ("flags", "archs", "spec_methods", "kv_dtypes", "request_fields", "env_vars", "arch_sets", "features",
             "kernel")
# The lists a `kernel` answer holds. One that is missing or empty makes the whole answer none: a release that renamed
# an enum must not read as "this engine has no CUDA-graph modes".
KERNEL_PARTS = ("cudagraph_modes", "compilation_modes", "compilation_fields", "pass_fields", "attention_backends",
                "opt_levels")
TIMED_OUT = "no answer within"


def _scrub(value: Any) -> Any:
    """`value` with every absolute path in its strings cut to its last part (rigtune.bench.compare.scrub_text): an
    ImportError names the module's file, and a flag's default can name a folder under the user's home."""
    from rigtune.bench.compare import scrub_text

    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def parse(engine: str, stdout: str, why_missing: str) -> EngineFacts:
    """EngineFacts from a probe's output: the marked lines, whatever else the engine printed around them. A key with
    no line gets `why_missing` as its error; an empty answer is no answer. Paths are cut out of every error and
    default, so the answers kept on disk, and the records made from them, name no folder."""
    answers: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for line in (stdout or "").splitlines():
        if not line.startswith(MARK):
            continue
        try:
            item = json.loads(line[len(MARK):])
        except ValueError:
            continue
        if not isinstance(item, dict) or "key" not in item:
            continue
        if "error" in item:
            errors[str(item["key"])] = str(item["error"])
        else:
            answers[str(item["key"])] = item.get("value")
    data: Dict[str, Any] = {"engine": engine}
    for key in KEYS[engine]:
        if key in errors:
            continue
        if key not in answers:
            errors[key] = why_missing
            continue
        try:
            EngineFacts.model_validate({"engine": engine, key: answers[key]})
        except Exception as exc:  # an answer of the wrong shape is no answer
            errors[key] = f"unreadable answer: {type(exc).__name__}"
            continue
        if key in NON_EMPTY and not answers[key]:
            errors[key] = "an empty answer, taken as none"
            continue
        if key == "kernel" and (lacking := [p for p in KERNEL_PARTS if not answers[key].get(p)]):
            errors[key] = f"an empty answer, taken as none: {', '.join(lacking)}"
            continue
        data[key] = answers[key] if key == "version" else _scrub(answers[key])
    return EngineFacts.model_validate({**data, "errors": _scrub(errors)})


def _text(output: Any) -> str:
    """A probe's output as text. Decoded here, never by subprocess: one byte the engine or a C extension prints that is
    not UTF-8 would otherwise lose every answer (on POSIX communicate() raises; on Windows its reader thread dies and
    the output comes back empty)."""
    if isinstance(output, bytes):
        return output.decode("utf-8", "replace")
    return output or ""


def run_probe(engine: str, python: str, timeout: float = PROBE_TIMEOUT,
              run: Callable[..., Any] = subprocess.run, clock: Callable[[], float] = time.monotonic) -> EngineFacts:
    """Ask `python` what `engine` offers. Never raises: a probe that cannot start, crashes or runs out of time returns
    facts with what it answered (possibly nothing) and an error for every other key.

    It runs in rigtune's environment, which the engine inherits, so it imports what a launch would (PYTHONPATH
    included), with CUDA_VISIBLE_DEVICES="" so that nothing it imports can take a GPU, and from an empty directory, as
    rigtune.speculative.run_suffix_probe does, so that a relative entry on PYTHONPATH ("." or "") finds nothing lying in
    the current one in the engine's place. (The script's own folder, not the current one, is first on its
    sys.path.)"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    started = clock()
    stdout, why = "", ""
    try:
        with tempfile.TemporaryDirectory(prefix="rigtune-probe-") as empty:
            out = run([python, str(script_path(engine))], capture_output=True, timeout=timeout, cwd=empty, env=env)
        stdout = _text(out.stdout)
        if out.returncode != 0:
            tail = (_text(out.stderr).strip().splitlines() or ["no output"])[-1]
            why = f"the probe exited {out.returncode}: {tail[:200]}"
        else:
            why = "not answered"
    except subprocess.TimeoutExpired as exc:
        stdout, why = _text(exc.stdout), f"{TIMED_OUT} {timeout:g} s"
    except Exception as exc:  # OSError: no such interpreter, not executable
        why = f"the probe could not run: {type(exc).__name__}: {exc}"[:300]
    facts = parse(engine, stdout, why)
    facts.probed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    facts.seconds = round(clock() - started, 1)
    return facts


# Files and folders under the engine's package that the probe's answers come from. Their modification times are part
# of the cache key, so an editable install updated in place (`git pull`), which changes neither the package folder's
# own time nor its dist-info, is asked again; a folder's time moves when a file is added to it or removed (a model).
SOURCES = {
    "vllm": ("config/speculative.py", "config/cache.py", "entrypoints/cli/serve.py", "engine/arg_utils.py",
             "entrypoints/common/cli_args.py", "entrypoints/launchers/cli_args.py", "model_executor/models",
             "model_executor/models/registry.py", "distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
             "v1/kv_offload/base.py", "config/compilation.py", "config/vllm.py",
             "v1/attention/backends/registry.py"),
    "sglang": ("srt/server_args.py", "srt/arg_groups", "srt/arg_groups/fields", "srt/speculative/spec_info.py",
               "srt/speculative/spec_registry.py", "srt/environ.py", "srt/entrypoints/openai/protocol.py",
               "srt/models", "srt/models/registry.py"),
}


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _dist_infos(engine: str, package: Path) -> List[Path]:
    return sorted((d for d in package.parent.glob("*.dist-info") if d.name.lower().startswith(engine + "-")),
                  key=lambda d: d.name)


def installed_version(engine: str, package: Path) -> Optional[str]:
    """The engine's version as the dist-info beside its package names it (`vllm-0.33.0.dist-info`), read without
    importing it; None when there is none there (an editable install keeps it elsewhere) or more than one."""
    found = {d.name[len(engine) + 1:-len(".dist-info")] for d in _dist_infos(engine, package)}
    return found.pop() if len(found) == 1 else None


def cache_key(engine: str, python: str, package: Path) -> str:
    """Names one install: the interpreter, the engine's package folder and when it last changed (a reinstall or an
    upgrade replaces it), its dist-info (the version, and when RECORD was written), the files the answers come from
    (SOURCES), and the probe script itself, so a rigtune whose probe asks more asks again."""
    script = script_path(engine).read_bytes()
    parts = [engine, str(python), str(package), str(_mtime(package)), hashlib.sha1(script).hexdigest()]
    parts += [f"{d.name}:{_mtime(d / 'RECORD')}" for d in _dist_infos(engine, package)]
    parts += [f"{rel}:{_mtime(package.joinpath(*rel.split('/')))}" for rel in SOURCES[engine]]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def cache_file(engine: str, python: str, package: Path) -> Path:
    return paths.home() / "engines" / f"{engine}-{cache_key(engine, python, package)}.json"


def forget(engine: str, python: Optional[str], package: Optional[Path]) -> None:
    """Drop the kept answers of one install, so the next cached_facts asks again (`rigtune probe --reprobe`)."""
    if python and package is not None:
        try:
            cache_file(engine, python, package).unlink()
        except OSError:
            pass


def same_release(probed: Optional[str], installed: Optional[str]) -> bool:
    """Whether the version the probe's Python reported is the one of the package that will be launched (a local build
    tag such as +cu129 aside). Unknown on either side: nothing to compare."""
    if not probed or not installed:
        return True
    return probed.split("+")[0] == installed.split("+")[0]


def cached_facts(engine: str, python: Optional[str], package: Optional[Path],
                 runner: Callable[[str, str], EngineFacts] = run_probe) -> Optional[EngineFacts]:
    """The answers of the install `python` runs `engine` from: read from <home>/engines/ when this install was asked
    before, else asked now and kept, a failed probe included (its errors say why; `rigtune probe --reprobe` asks
    again). A probe that ran out of time is not kept, so the next run asks again. None when probing is off or the
    engine is not found (nothing is run then), and when the Python asked reports another version than the package that
    will be launched: its answers are then about another install (a `vllm` command that is a shell wrapper into another
    environment), and the source checks answer instead."""
    if not enabled() or not python or package is None:
        return None
    installed = installed_version(engine, package)
    path = cache_file(engine, python, package)
    facts = None
    if path.exists():
        try:
            facts = EngineFacts.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("ignoring unreadable engine facts %s: %s", path, exc)
    if facts is None:
        logger.info("asking %s's own Python what it offers (once per install, up to %.0f s): %s", engine,
                    PROBE_TIMEOUT, python)
        facts = runner(engine, python)
        missing = facts.unanswered()
        if missing:
            first = facts.errors.get(missing[0], "")
            logger.warning("%s did not answer %s (%s); rigtune reads its source for those, as before", engine,
                           ", ".join(missing), first)
        timed_out = any(e.startswith(TIMED_OUT) for e in facts.errors.values())
        if same_release(facts.version, installed) and not timed_out:
            _keep(path, facts)
    if not same_release(facts.version, installed):
        logger.warning("%s reports %s %s, but the package that will be launched is %s; its answers are not used",
                       python, engine, facts.version, installed)
        return None
    return facts


def _keep(path: Path, facts: EngineFacts) -> None:
    """Write `facts` at `path` through a temporary file of this writer's own, so two rigtune processes asking at once
    never write into the same file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(facts.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("could not keep engine facts at %s: %s", path, exc)
