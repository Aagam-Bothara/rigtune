"""Whether an engine has an MTP draft for a model's architecture, asked of the engine.

Each engine keeps one rule per model family for turning a target's config into its MTP draft's, and every release
adds some: vLLM 0.33's SpeculativeConfig.hf_config_override has about twenty (Qwen3.5, Inkling, DeepSeek, GLM, Kimi,
MiMo, ...), SGLang 0.5.23's ModelConfig._config_draft_model as many. rigtune kept a list of the four Qwen3.5
architectures it had checked, so a head both engines load (Inkling-Small's, DeepSeek's, GLM's) was never offered, and
nothing said so. Now the engine is asked: probes/vllm_mtp_probe.py and probes/sglang_mtp_probe.py load the model's
config in the engine's Python and apply the engine's own rewrite to it, as a launch does, and say which architecture
the engine serves, which class the draft would be, and how many MTP layers it takes.

A yes says the engine has a rule for the architecture: SGLang's rules rename by architecture alone (and many set one
layer whatever the checkpoint holds), and vLLM's read a layer count from the config, not the weights. Whether this
checkpoint holds the head is rigtune.introspect's question, and a head is offered only when both say yes. A no needs a
positive finding: the draft would be the target's own class (no rule), vLLM's rule reads no MTP layers, or vLLM's
rewrite itself refused the checkpoint. Anything else is unknown, among it every failure to load the config (vLLM turns
those into ValueErrors too: a gated repository, a bad revision, an unreachable Hub); the caller then decides from the
checkpoint alone, and a start that fails costs one trial, after which the search blocks the head.

Answers are kept in <home>/engines/heads/, named by the install, the probe script and the model at a commit (a branch
or tag moves; a local directory is named by its path and when its config changed), and only when they settle the
question: a clean answer, or a refusal by vLLM's rewrite. A kept answer is used while the engine's Python still has the
transformers and huggingface_hub it was given with (they load the config, wherever they are installed). An answer with
any other error, one that ran out of time, and one for a model at no commit are asked again next time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Collection, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from rigtune import paths
from rigtune.backends import facts

logger = logging.getLogger(__name__)

HEAD_TIMEOUT = 180.0  # the probe imports the engine and reads the model's config, from the Hub when not cached
SCRIPTS = {"vllm": "vllm_mtp_probe.py", "sglang": "sglang_mtp_probe.py"}
KEYS = ("target_arch", "draft_arch", "layers")
CONFIG_LIBS = ("transformers", "huggingface_hub")  # they load the config in the engine's Python
COMMIT = re.compile(r"[0-9a-f]{7,40}")  # a revision that cannot move under a kept answer
LIBS_CODE = ("import importlib.metadata as m, json\n"
             f"print(json.dumps({{n: m.version(n) for n in {CONFIG_LIBS!r}}}))\n")


class HeadAnswer(BaseModel):
    """One engine install's answer for one model. A field left None was not answered (`errors` says why)."""

    engine: str
    target_arch: Optional[str] = None
    draft_arch: Optional[str] = None
    layers: Optional[int] = None
    libs: Dict[str, str] = Field(default_factory=dict)  # CONFIG_LIBS' versions in the engine's Python
    errors: Dict[str, str] = Field(default_factory=dict)
    probed_at: Optional[str] = None
    seconds: Optional[float] = None

    def refused(self) -> bool:
        """vLLM's rewrite raised on this checkpoint: the config loaded, and the rule for it said no."""
        return (self.engine == "vllm" and self.target_arch is not None and self.draft_arch is None
                and self.errors.get("draft_arch", "").startswith("ValueError"))

    def settled(self) -> bool:
        """Whether keeping the answer is safe: nothing went wrong but a refusal."""
        return not self.errors or (self.refused() and set(self.errors) <= {"draft_arch", "layers"})


@dataclass(frozen=True)
class Verdict:
    # True: the engine has an MTP rule for the architecture (the caller still needs the checkpoint to hold the head);
    # False: it builds no MTP draft from this checkpoint; None: could not be told.
    loads: Optional[bool]
    why: str
    draft_arch: Optional[str] = None
    layers: Optional[int] = None


def verdict(answer: Optional[HeadAnswer], registered: Optional[Collection[str]] = None) -> Verdict:
    """What `answer` says. `registered` is the engine's model registry (its install probe's `archs`), or None when
    that was not answered."""
    if answer is None:
        return Verdict(None, "the engine was not asked (probing is off, or its Python was not found)")
    engine, target, arch, layers = answer.engine, answer.target_arch, answer.draft_arch, answer.layers
    if target is None:
        return Verdict(None, f"{engine} could not load the model's config ({answer.errors.get('target_arch', '')})")
    if answer.refused():
        return Verdict(False, f"{engine} refuses an MTP draft from this checkpoint ({answer.errors['draft_arch']})")
    if arch is None:
        return Verdict(None, f"{engine} could not be asked ({answer.errors.get('draft_arch', 'not answered')})")
    if arch == target:
        if registered is not None and target not in registered:  # served by a generic backend, which may have its own
            return Verdict(None, f"{engine} has no class of its own for {target}; its generic backend was not asked")
        return Verdict(False, f"{engine} has no MTP draft for {target}", arch, layers)
    if registered is not None and arch not in registered:
        return Verdict(None, f"{engine} would build {arch}, which its model registry does not list", arch, layers)
    if engine == "vllm":  # vLLM's rules read the count from the config; SGLang's often set it, so it says nothing
        if layers is None:
            return Verdict(None, f"vllm's rule for {target} found no MTP layer count in this config", arch, layers)
        if layers < 1:
            return Verdict(False, f"vllm's rule for {target} reads no MTP layers from this config", arch, layers)
        return Verdict(True, f"vllm builds {arch} with {layers} MTP layer{'s' if layers != 1 else ''}", arch, layers)
    return Verdict(True, f"{engine} builds {arch} for {target}", arch, layers)


def local_model(model: str) -> str:
    """An existing local directory as an absolute path (the probe runs from an empty folder elsewhere); a Hub id as
    it is."""
    return str(Path(model).resolve()) if os.path.isdir(model) else model


def config_libs(python: str, run: Callable[..., object] = subprocess.run) -> Optional[Dict[str, str]]:
    """CONFIG_LIBS' versions in `python`, asked without importing the engine (a fraction of a second), from an empty
    folder as every probe runs; None when it cannot say."""
    try:
        with tempfile.TemporaryDirectory(prefix="rigtune-probe-") as empty:
            out = run([python, "-c", LIBS_CODE], capture_output=True, timeout=30, cwd=empty,
                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        found = json.loads(facts._text(out.stdout).strip().splitlines()[-1]) if out.returncode == 0 else None
    except Exception:
        return None
    return found if isinstance(found, dict) else None


def _model_part(model: str, revision: Optional[str]) -> str:
    """The model's share of the cache key: its id or absolute path and revision, and for a local directory when its
    config last changed."""
    config = Path(model) / "config.json"
    stamp = str(facts._mtime(config)) if config.is_file() else ""
    return f"{model}\n{revision or ''}\n{stamp}"


def cache_file(engine: str, python: str, package: Path, model: str, revision: Optional[str]) -> Path:
    """Named by the install (facts.cache_key), this probe script (a rigtune whose probe asks otherwise asks again) and
    the model."""
    script = (facts.PROBES / SCRIPTS[engine]).read_bytes()
    key = "\n".join([facts.cache_key(engine, python, package), hashlib.sha1(script).hexdigest(),
                     _model_part(model, revision)])
    return paths.home() / "engines" / "heads" / f"{engine}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}.json"


def run_head_probe(engine: str, python: str, model: str, revision: Optional[str],
                   timeout: float = HEAD_TIMEOUT, run: Optional[Callable[..., object]] = None) -> HeadAnswer:
    """Ask `python` which MTP draft `engine` builds from `model`. Never raises."""
    started = datetime.now(timezone.utc)
    kwargs = {"run": run} if run is not None else {}
    stdout, why = facts.run_script(python, facts.PROBES / SCRIPTS[engine], [local_model(model), revision or ""],
                                   timeout, **kwargs)
    answers, errors = facts.marked(stdout)
    data: Dict[str, object] = {"engine": engine}
    for key in ("target_arch", "draft_arch"):
        if isinstance(answers.get(key), str) and answers[key]:
            data[key] = answers[key]
        elif key in answers:
            errors[key] = "the engine's config names no architecture"
    layers = answers.get("layers")
    if isinstance(layers, int) and not isinstance(layers, bool):
        data["layers"] = layers
    libs = answers.get("libs")
    if isinstance(libs, dict) and all(isinstance(v, str) for v in libs.values()):
        data["libs"] = libs
    # Each step is asked only when the one before it answered; a step that was due and left no line is an error.
    due = ["target_arch"] + (["draft_arch"] if "target_arch" in data else []) \
        + (["layers"] if "draft_arch" in data else [])
    for key in due:
        if key not in answers and key not in errors:
            errors[key] = why
    data["errors"] = {k: facts._scrub(v) for k, v in errors.items() if k in KEYS}
    data["probed_at"] = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    data["seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    return HeadAnswer.model_validate(data)


def cached_answer(engine: str, python: Optional[str], package: Optional[Path], model: str, revision: Optional[str],
                  runner: Optional[Callable[[str, str, str, Optional[str]], HeadAnswer]] = None,
                  libs_of: Callable[[str], Optional[Dict[str, str]]] = config_libs) -> Optional[HeadAnswer]:
    """The install's answer for `model` at `revision`: the one kept in <home>/engines/heads/ while `python` still has
    the config libraries it was given with, else asked now, and kept when it settles the question, says which
    libraries gave it, and the model cannot move under it (a commit, or a local directory). None when probing is off
    or the engine was not found."""
    if not facts.enabled() or not python or package is None:
        return None
    model = local_model(model)
    path = cache_file(engine, python, package, model, revision)
    if path.exists():
        try:
            kept = HeadAnswer.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("ignoring unreadable MTP answer %s: %s", path, exc)
        else:
            if kept.libs and kept.libs == libs_of(python):
                return kept
    logger.info("asking %s's own Python whether it has an MTP draft for %s (up to %.0f s)", engine, model,
                HEAD_TIMEOUT)
    answer = (runner or run_head_probe)(engine, python, model, revision)
    fixed = os.path.isdir(model) or bool(revision and COMMIT.fullmatch(revision))
    if answer.settled() and answer.libs and fixed:
        _keep(path, answer)
    return answer


def _keep(path: Path, answer: HeadAnswer) -> None:
    """Write through a temporary file of this writer's own (facts._keep), removed if the write fails."""
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(answer.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("could not keep the MTP answer at %s: %s", path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass


def ask(engine: str, python: Optional[str], package: Optional[Path], model: str, revision: Optional[str],
        registered: Optional[Collection[str]] = None,
        runner: Optional[Callable[[str, str, str, Optional[str]], HeadAnswer]] = None,
        libs_of: Callable[[str], Optional[Dict[str, str]]] = config_libs) -> Tuple[Verdict, Optional[HeadAnswer]]:
    """verdict() of the install's answer for `model`, with the answer."""
    answer = cached_answer(engine, python, package, model, revision, runner, libs_of)
    return verdict(answer, registered), answer
