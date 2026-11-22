"""Find drafters for a target on the Hugging Face Hub: EAGLE3 heads, DFlash and DSpark drafters.

A name proposes; files decide. The search reads only listings until it has a short list, then each short-listed
repository's config.json and file listing, which rigtune.heads judges per engine. What the Hub research of 21 Nov 2026
found (about 300 anonymous requests over eleven targets) shaped it:

  * The base_model tag misses most of the heads people use (RedHatAI's, z-lab's, Google's assistants), so the search
    is by name: "<family-size> <word>" for each kind's word, which the Hub matches case- and separator-insensitively.
    base_model children are added, one page. Downloads rank the short list, after rigtune's own listed heads: on the
    real Hub (21 Nov 2026) ranking base_model children first spent Qwen3-8B's reads on experiments with a few hundred
    downloads and never read RedHatAI's head (51,787); a base_model tie only breaks equal counts.
  * A listing's config holds the architectures and model type only, so the listing can only short-list by
    architecture and name; every verdict needs the repository's own config.json.
  * Names are traps as often as evidence: heads for the Coder, VL, Thinking or -2507 variant of a target, quantized
    copies, heads for another size. A candidate whose name carries a variant marker or a size the target's lacks is
    dropped before anything is read.
  * Anonymous use is limited to 500 API calls in 5 minutes: a search costs 7 to 17 (the target, 5 or 6 listings, the
    best of each kind read: READ_QUOTA), kept for the day per target commit when every part answered; an offline run
    reads what was kept, else the listed heads in the local Hugging Face cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rigtune.heads import ENGINE_ARCHS, check_head, head_kind
from rigtune.models import ArchInfo

logger = logging.getLogger(__name__)

SEARCH_WORDS = ("eagle", "dflash", "dspark", "speculator")
LIST_LIMIT = 50  # results per listing; the drafters searched for are few per target
LISTING_FIELDS = ["downloads", "config", "cardData", "gated", "sha"]
# Words a candidate's name may carry only when the target's does: another variant of the model, or a quantized,
# converted or modified copy, each trained or stored for something else.
VARIANT_MARKERS = frozenset({
    "coder", "vl", "omni", "base", "math", "guard", "abliterated", "uncensored", "qat", "distill",
    "quantized", "fp8", "nvfp4", "mxfp4", "awq", "gptq", "int4", "int8", "w4a16", "w8a8", "w4a4", "bnb", "4bit",
    "8bit", "exl2", "gguf", "mlx", "onnx", "ov"})
OPPOSED = (("thinking", "instruct"),)  # a head for one is a head for another model
ALL_ARCHS = frozenset().union(*(kinds for engine in ENGINE_ARCHS.values() for kinds in engine.values()))
_SIZE = re.compile(r"(?<![a-z0-9])(?:[ae])?\d+(?:\.\d+)?[bm](?![a-z0-9])")
_DATE = re.compile(r"^2\d(?:0[1-9]|1[0-2])$")  # a release, YYMM: 2507, 2512; not 2048


def _sizes(text: str) -> List[str]:
    return _SIZE.findall(text.lower().replace("_", "-"))


def core_name(hf_id: str) -> str:
    """The family and size a drafter's name carries for `hf_id`: "Qwen3-30B-A3B" for Qwen/Qwen3-30B-A3B-Instruct-2507,
    "Llama-3.1-8B" for unsloth/Meta-Llama-3.1-8B-Instruct; the whole name when it states no size."""
    name = hf_id.rsplit("/", 1)[-1]
    name = re.sub(r"^Meta-", "", name)
    end = None
    for m in _SIZE.finditer(name.lower().replace("_", "-")):
        end = m.end()
    return name[:end] if end else name


def _sequence(text: str) -> List[str]:
    """A name's words in order, lower case, letters and digits apart ("LLaMA3.1-8B" -> llama, 3, 1, 8b), sizes whole."""
    text = _SIZE.sub(lambda m: f" {m.group(0)} ", text.lower().replace("_", "-"))
    out: List[str] = []
    for token in re.split(r"[^a-z0-9]+", text):
        out += [token] if _SIZE.fullmatch(token) else re.findall(r"[a-z]+|\d+", token)
    return out


def _words(text: str) -> set:
    """The words of _sequence, and each separator-delimited token whole ("fp8", "w4a16"), for the variant markers."""
    lowered = text.lower().replace("_", "-")
    return set(_sequence(text)) | {t for t in re.split(r"[^a-z0-9]+", lowered) if t}


def _has_family(words: List[str], family: List[str]) -> bool:
    """Whether the family's words appear together in order, and not followed by another number: Qwen3 is not
    Qwen3.5, nor Llama-3 Llama-3.1."""
    n = len(family)
    for i in range(len(words) - n + 1):
        if words[i:i + n] == family and not (i + n < len(words) and words[i + n].isdigit()):
            return True
    return False


def name_consistent(repo: str, target_id: str) -> bool:
    """Whether a candidate's name is that of a drafter for `target_id`: the target's family words appear together,
    with its total size, it states no other size, it carries no variant marker or release date the target's name
    lacks, and no variant word opposed to the target's own."""
    name = repo.rsplit("/", 1)[-1]
    target_name = target_id.rsplit("/", 1)[-1]
    core = core_name(target_id)
    want_sizes = _sizes(core)
    have_sizes = set(_sizes(name))
    words, target_words = _words(name), _words(target_name)
    family = [w for w in _sequence(core) if not _SIZE.fullmatch(w)]
    if not _has_family(_sequence(name), family):
        return False
    if want_sizes and (want_sizes[0] not in have_sizes or not have_sizes <= set(want_sizes)):
        return False
    if (words - target_words) & VARIANT_MARKERS:
        return False
    for a, b in OPPOSED:
        if (a in words and b in target_words and a not in target_words) or (
                b in words and a in target_words and b not in target_words):
            return False
    dates, target_dates = {w for w in words if _DATE.match(w)}, {w for w in target_words if _DATE.match(w)}
    return dates == target_dates


def looks_like_a_drafter(repo: str, architectures: Iterable[str]) -> bool:
    """From a listing alone: an architecture one of the engines loads as a drafter, a speculators-format class, or a
    plain architecture under a name that says EAGLE3 (vLLM renames those into its Eagle3 model; the config decides)."""
    archs = [str(a) for a in architectures or []]
    if any(a in ALL_ARCHS for a in archs):
        return True
    if any(re.search(r"speculator$|eagle3|dflash|dspark", a, re.I) for a in archs):
        return True
    return bool(archs) and re.search(r"eagle[-_]?3", repo, re.I) is not None


@dataclass
class Candidate:
    """A drafter found by the search, with what was read of it (config and files: only for those read)."""
    repo: str
    downloads: int = 0
    tier: int = 2  # 0 listed by rigtune, 1 declares the target as its base model, 2 found by name only
    sha: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, Optional[int]]] = None
    size_bytes: Optional[int] = None


@dataclass
class Search:
    """What a search found for one target commit, kept for the day."""
    target: str
    revision: Optional[str]
    day: str
    aliases: List[str] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)  # ranked; the best of each kind were read
    notes: List[str] = field(default_factory=list)
    complete: bool = True  # no listing or read failed in a way that may pass (transient): only then reused

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Search":
        raw = json.loads(text)
        raw["candidates"] = [Candidate(**c) for c in raw.get("candidates", [])]
        return cls(**raw)


class Hub:
    """The three Hub calls a search makes; tests pass a fake."""

    def __init__(self, token: Optional[str] = None):
        from huggingface_hub import HfApi

        self.api, self.token = HfApi(token=token), token

    def list_models(self, **kw) -> Iterable[Any]:
        return self.api.list_models(expand=LISTING_FIELDS, limit=LIST_LIMIT, **kw)

    def model_info(self, repo: str, revision: Optional[str] = None) -> Any:
        return self.api.model_info(repo, revision=revision, files_metadata=True)

    def download(self, repo: str, filename: str, revision: Optional[str] = None) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, filename, revision=revision, token=self.token)


def _base_models(card: Any) -> List[str]:
    base = (card or {}).get("base_model") if isinstance(card, dict) else getattr(card, "base_model", None)
    if isinstance(base, str):
        return [base]
    return [str(b) for b in base] if isinstance(base, (list, tuple)) else []


def _card(model: Any) -> Any:
    return getattr(model, "card_data", None) or getattr(model, "cardData", None)


def transient(exc: BaseException) -> bool:
    """Whether a Hub request's failure may pass: a rate limit (429), a server error, no connection. A missing file,
    repository or revision, a refusal (401, 403, 404) and a malformed config are the repository's answer, and a search
    that met only those is complete: inference-optimization/ctest-Qwen3.5-9B-sliding-window-speculator.dflash has no
    config.json (21 Nov 2026), which would otherwise have every search for Qwen3.5-9B redone on every call."""
    from huggingface_hub import errors

    if isinstance(exc, (errors.LocalEntryNotFoundError, errors.OfflineModeIsEnabled)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    final = (errors.EntryNotFoundError, errors.RepositoryNotFoundError, errors.RevisionNotFoundError,
             FileNotFoundError, ValueError)
    return not isinstance(exc, final)


def _listed_config(model: Any) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    return cfg if isinstance(cfg, dict) else {}


def guessed_kind(repo: str, architectures: Iterable[str]) -> str:
    """The kind a short-listed repository's name and listed architectures suggest, for READ_QUOTA only; its config
    decides what it is."""
    text = " ".join([repo, *[str(a) for a in architectures or []]]).lower()
    return "dflash" if "dflash" in text else "dspark" if "dspark" in text else "eagle3"


# Repositories read per kind of drafter, at most: the DFlash and DSpark drafters of a popular target outnumber its
# EAGLE3 heads (gemma-4-12B-it: seven of the eight best), and a single shared budget left none for EAGLE3.
READ_QUOTA = {"eagle3": 5, "dflash": 3, "dspark": 2}


def search(target_id: str, revision: Optional[str], seeds: Iterable[str] = (), hub: Optional[Hub] = None,
           today: Optional[str] = None) -> Search:
    """Search the Hub for drafters of `target_id` and read the best of each kind (READ_QUOTA). Raises when no listing
    was answered; a listing or read that fails is noted and the search marked incomplete, so it is not reused."""
    from rigtune.hfconfig import INDEX, hub_listing, listed_bytes

    hub = hub or Hub()
    found = Search(target=target_id, revision=revision, day=today or date.today().isoformat())
    try:
        info = hub.model_info(target_id, revision)
        found.aliases = [target_id, *_base_models(_card(info))]
    except Exception as exc:  # a gated target answers no card anonymously: searched by name alone
        found.aliases = [target_id]
        found.notes.append(f"{target_id}'s own card not read ({type(exc).__name__}); its base model is not an alias")
    seen: Dict[str, Candidate] = {}
    kinds: Dict[str, str] = {}
    answered = 0
    queries = [{"search": f"{core_name(target_id)} {w}"} for w in SEARCH_WORDS]
    queries += [{"filter": f"base_model:{alias}"} for alias in found.aliases]
    for query in queries:
        try:
            listed = list(hub.list_models(**query))
            answered += 1
        except Exception as exc:
            found.complete = found.complete and not transient(exc)
            found.notes.append(f"Hub listing {query} failed ({type(exc).__name__})")
            continue
        for m in listed:
            repo = getattr(m, "id", None)
            if not isinstance(repo, str) or repo in seen or repo == target_id:
                continue
            gated = getattr(m, "gated", False)
            archs = _listed_config(m).get("architectures") or []
            if gated not in (False, None) or not looks_like_a_drafter(repo, archs) or not name_consistent(
                    repo, target_id):
                continue
            tie = {b.lower() for b in _base_models(_card(m))} & {a.lower() for a in found.aliases}
            seen[repo] = Candidate(repo=repo, downloads=int(getattr(m, "downloads", 0) or 0), tier=1 if tie else 2)
            kinds[repo] = guessed_kind(repo, archs)
    if not answered:
        raise RuntimeError("the Hub answered none of the search's listings")
    for repo in seeds:
        seen[repo] = Candidate(repo=repo, downloads=seen[repo].downloads if repo in seen else 0, tier=0)
        kinds[repo] = "eagle3"
    found.candidates = sorted(seen.values(), key=lambda c: (c.tier != 0, -c.downloads, c.tier, c.repo))
    left = dict(READ_QUOTA)
    for cand in found.candidates:
        kind = kinds.get(cand.repo, "eagle3")
        if left.get(kind, 0) <= 0:
            continue
        left[kind] -= 1
        try:
            info = hub.model_info(cand.repo)
            cand.sha = getattr(info, "sha", None)
            cand.files = hub_listing(info)
            with open(hub.download(cand.repo, "config.json", cand.sha), "r", encoding="utf-8") as fh:
                config = json.load(fh)
            if not isinstance(config, dict):
                raise ValueError("config.json is not an object")
            cand.config = config
            cand.size_bytes = listed_bytes(cand.files, lambda: hub.download(cand.repo, INDEX, cand.sha))
        except Exception as exc:
            found.complete = found.complete and not transient(exc)
            found.notes.append(f"{cand.repo} not read ({type(exc).__name__})")
    return found


def _cache_path(target_id: str, revision: Optional[str]) -> Path:
    from rigtune.paths import heads_dir

    key = hashlib.sha256(json.dumps([target_id, revision]).encode()).hexdigest()[:24]
    return heads_dir() / f"{key}.json"


def _read_kept(path: Path) -> Optional[Search]:
    """The search kept at `path`, or None when there is none or it cannot be read (a corrupt file is replaced by the
    next search, never fatal)."""
    try:
        kept = Search.from_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return kept if isinstance(kept, Search) and isinstance(kept.candidates, list) else None


def _keep(path: Path, found: Search) -> None:
    """Write atomically: both engines' backends search in one calibration and read what the first one kept."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as fh:
            fh.write(found.to_json())
        os.replace(fh.name, path)
    except OSError as exc:
        logger.warning("could not keep the drafter search for %s: %s", found.target, exc)


def from_local_cache(target_id: str, revision: Optional[str], seeds: Iterable[str], today: str) -> Optional[Search]:
    """With the Hub off and no search kept: the listed heads whose config.json is in the local Hugging Face cache, as
    rigtune offered them offline before the search existed; sized from the snapshot when its weights are there."""
    from huggingface_hub import try_to_load_from_cache

    from rigtune.hfconfig import local_checkpoint_bytes

    found = Search(target=target_id, revision=revision, day=today, aliases=[target_id], complete=False,
                   notes=[f"drafters for {target_id} from the local Hugging Face cache only; the Hub is off"])
    for repo in seeds:
        try:
            path = try_to_load_from_cache(repo, "config.json")
            if not isinstance(path, str):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot = Path(path).parent
            files = {f.name: f.stat().st_size for f in snapshot.iterdir() if f.is_file()}
            found.candidates.append(Candidate(repo=repo, tier=0, sha=snapshot.name,
                                              config=config if isinstance(config, dict) else None, files=files,
                                              size_bytes=local_checkpoint_bytes(snapshot)))
        except Exception as exc:
            logger.debug("listed head %s not read from the cache: %s", repo, exc)
    return found if found.candidates else None


def kept_or_search(target_id: str, revision: Optional[str], seeds: Iterable[str] = (), *, hub_off: bool = False,
                   hub: Optional[Hub] = None, today: Optional[str] = None) -> Tuple[Optional[Search], List[str]]:
    """Today's search for this target commit: the one kept from earlier today when it was complete, else a new one
    (kept). With the Hub off, the last one kept whatever its day, else the listed heads in the local Hugging Face
    cache, else None. Returns (search or None, notes)."""
    today = today or date.today().isoformat()
    path = _cache_path(target_id, revision)
    kept = _read_kept(path)
    if hub_off:
        if kept is not None:
            return kept, [f"drafters for {target_id} from the search kept on {kept.day}; the Hub is off"]
        cached = from_local_cache(target_id, revision, seeds, today)
        if cached is not None:
            return cached, list(cached.notes)
        return None, [f"no drafter search kept for {target_id}; the Hub is off, so none was made"]
    if kept is not None and kept.day == today and kept.complete:
        return kept, []
    found = search(target_id, revision, seeds, hub=hub, today=today)
    _keep(path, found)
    return found, []


def heads_for(found: Search, target: ArchInfo, engine: str, kind: str = "eagle3") -> List[Candidate]:
    """The read candidates `engine` loads as `kind` and that fit `target` (rigtune.heads.check_head "ok"), in the
    search's order: rigtune's listed heads, then downloads. Each engine gets its own best: calibration then chooses
    between the engines on what each does at its best. A config the check cannot read is skipped."""
    out = []
    for cand in found.candidates:
        if not isinstance(cand.config, dict):
            continue
        try:
            fits = check_head(target, cand.config, engine, files=cand.files, aliases=found.aliases).status == "ok"
            fits = fits and head_kind(cand.config, engine) == kind
        except Exception as exc:  # a config shaped like no other: not a drafter rigtune can judge
            logger.info("%s not judged on %s: %s", cand.repo, engine, exc)
            continue
        if fits:
            out.append(cand)
    return out
