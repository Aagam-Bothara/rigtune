"""One reasoning experiment end to end: problems, every method in a random order, runs and report written.

    python -m rigtune.reasoning --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B --split dev \\
        --methods direct,self-consistency:3,self-consistency:5,self-correction,adaptive:8 --thinking off --out out/

Before anything runs it checks what would spoil the comparison: that the server's context holds a critique turn
(problem, a first answer of max_tokens, the critique, a final answer of max_tokens), since overflowing it fails the
hardest problems; and, for the held-out set, that math-verify works. Methods run in an order shuffled by a seed, so
none always meets the prefix cache warm from the one before. Each run is written as it finishes (runs.jsonl), then the
report (report.md) and a manifest of what was asked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rigtune.reasoning import methods as rm
from rigtune.reasoning import problems as rp
from rigtune.reasoning import report as rep
from rigtune.reasoning import run as rr

# Characters a token, for the context check only: 1.5 is below the worst MATH level 4-5 prompt under Qwen3's
# tokenizer (1.68; the mean is 3.1), so the check never underestimates a prompt.
CHARS_PER_TOKEN = 1.5
WARMUP_PROBLEMS = 8  # answered once, untimed, before the first method, so no method pays the server's first requests
TEMPLATE_TOKENS = 64  # the chat template's own tokens a turn, generously
CRITIQUE_TOKENS = 64


def parse_specs(text: str) -> List[Tuple[str, Optional[int]]]:
    """"direct,self-consistency:3,adaptive:8" -> [("direct", None), ("self-consistency", 3), ("adaptive", 8)]."""
    out: List[Tuple[str, Optional[int]]] = []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        method, _, k = part.partition(":")
        if method not in rr.METHODS:
            raise ValueError(f"unknown method {method!r}; one of {', '.join(rr.METHODS)}")
        if method in ("self-consistency", "adaptive") and not k:
            raise ValueError(f"{method} needs k, as {method}:5")
        if method in ("direct", "self-correction") and k:
            raise ValueError(f"{method} takes no k")
        if method == "self-consistency" and k and int(k) < 2:
            raise ValueError("self-consistency needs k of 2 or more (k = 1 is direct)")
        if method == "adaptive" and k and int(k) < 3:
            raise ValueError("adaptive needs k of 3 or more (it always takes 2)")
        out.append((method, int(k) if k else None))
    if not out:
        raise ValueError("no methods given")
    return out


def context_needed(problems: Sequence[rp.Problem], max_tokens: int) -> int:
    """Tokens the longest critique turn may need: the problem, a first answer and a final one of max_tokens each, the
    critique, and the template's own tokens, the problem's counted conservatively."""
    longest = max(len(rm.SOLVE.format(problem=p.problem)) for p in problems)
    return int(longest / CHARS_PER_TOKEN) + 2 * max_tokens + CRITIQUE_TOKENS + 3 * TEMPLATE_TOKENS


async def preemptions(base_url: str) -> Optional[float]:
    """vLLM's count of requests it preempted (vllm:num_preemptions_total, summed over engines), or None. A method that
    keeps more requests running than the KV cache holds has some recomputed; recorded per run, since the cost falls
    on the methods that run many answers at once."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30) as client:
            text = (await client.get("/metrics")).text
    except Exception:
        return None
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith("vllm:num_preemptions_total") and not line.startswith("#")]
    return sum(values) if values else None


async def server_context(base_url: str) -> Optional[int]:
    """The served model's context length, as vLLM's /v1/models says it (max_model_len); None when it does not say."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30) as client:
            data = (await client.get("/v1/models")).json().get("data") or []
        lengths = [int(m["max_model_len"]) for m in data if m.get("max_model_len")]
        return min(lengths) if lengths else None
    except Exception:
        return None


def check(problems: Sequence[rp.Problem], settings: rm.Settings, split: str, context: Optional[int],
          specs: Sequence[Tuple[str, Optional[int]]], out: Optional[Path] = None) -> List[str]:
    """Why the experiment must not run (an empty list when it may)."""
    why = []
    if out is not None and (out / "runs.jsonl").exists():
        why.append(f"{out / 'runs.jsonl'} exists: runs from another experiment would mix with these")
    if not problems:
        why.append("no problems")
    if split == "heldout" and rp._math_verify() is None:
        why.append("the held-out run needs math-verify working (pip install \"rigtune[reason]\"); without it the "
                   "authors' normalisation accepts hedges and half answers")
    if any(m == "self-correction" for m, _ in specs) and problems:
        need = context_needed(problems, settings.max_tokens)
        if context is None:
            why.append(f"the server did not say its context length; a critique turn may need {need} tokens "
                       "(pass --context to state it)")
        elif context < need:
            why.append(f"the server's context ({context}) is shorter than a critique turn may need ({need}): serve "
                       f"with a longer --max-model-len or lower --max-tokens")
    elif problems and context is not None:
        need = int(max(len(rm.SOLVE.format(problem=p.problem)) for p in problems) / CHARS_PER_TOKEN) \
            + settings.max_tokens + TEMPLATE_TOKENS
        if context < need:
            why.append(f"the server's context ({context}) is shorter than an answer may need ({need})")
    return why


def fill_warning(problems: Sequence[rp.Problem], sequences: int) -> Optional[str]:
    """A warning when the batch is too small to keep `sequences` answers in flight: a method that sends one answer a
    problem at a time (direct, self-correction) can fill at most len(problems) of them, so its utilisation, and with it
    the criterion, cannot be read as GPU time."""
    if len(problems) < 4 * sequences:
        return (f"{len(problems)} problems cannot keep {sequences} answers in flight for direct answers or "
                f"self-correction: their utilisation will be low and the criterion undecided; use --sequences "
                f"{max(1, len(problems) // 4)} or fewer for a meaningful GPU time")
    return None


async def run_experiment(post: rm.Post, settings: rm.Settings, problems: Sequence[rp.Problem],
                         specs: Sequence[Tuple[str, Optional[int]]], sequences: int, order_seed: int,
                         out: Path, problem_set: str, server: str,
                         energy: Optional[Callable[[], Optional[float]]] = None,
                         log: Callable[[str], None] = print,
                         preempted: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Every method over the problems in a shuffled order, each run appended to out/runs.jsonl as it finishes, after
    one untimed warm-up batch. `preempted` is an async reader of the server's preemption count."""
    order = list(specs)
    random.Random(order_seed).shuffle(order)
    out.mkdir(parents=True, exist_ok=True)
    await rr.run_method(post, settings, problems[:WARMUP_PROBLEMS], "direct", sequences=sequences)  # untimed
    runs = []
    with open(out / "runs.jsonl", "w", encoding="utf-8") as fh:
        for method, k in order:
            t0 = time.time()
            p0 = await preempted() if preempted else None
            r = await rr.run_method(post, settings, problems, method, k, sequences=sequences, energy=energy,
                                    problem_set=problem_set, server=server)
            p1 = await preempted() if preempted else None
            row = rep.row(r.to_json())
            data = r.to_json()
            data["preemptions"] = (p1 - p0) if p0 is not None and p1 is not None else None
            log(f"{r.name}: {row.accuracy:.1%} of {row.n}, {row.gpu_seconds:.2f} GPU-s a problem, utilisation "
                f"{r.utilisation:.0%}, {row.errors} failed, {row.truncated} cut off, preempted "
                f"{data['preemptions']} ({time.time() - t0:.0f} s)")
            fh.write(json.dumps(data) + "\n")
            fh.flush()
            runs.append(data)
    return runs


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m rigtune.reasoning", description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=("dev", "heldout"), default="dev")
    ap.add_argument("--levels", default="4,5")
    ap.add_argument("--limit", type=int, default=0, help="the first N problems by id (0: all)")
    ap.add_argument("--methods", default="direct,self-consistency:3,self-consistency:5,self-correction,adaptive:8")
    ap.add_argument("--thinking", choices=("on", "off", "default"), default="off")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0: the model's own (vLLM reads its generation_config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequences", type=int, default=256)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=0, help="the server's context length, when it does not say")
    ap.add_argument("--server", default="", help="what serves the model, recorded so only like runs are compared")
    ap.add_argument("--keep-texts", action="store_true", help="keep every reply's text, for training a verifier")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


async def _main(args: argparse.Namespace) -> int:
    specs = parse_specs(args.methods)
    levels = tuple(int(x) for x in args.levels.split(","))
    problems, digest = rp.load_math(levels=levels, split=args.split, cache_dir=args.cache_dir)
    if args.limit:
        problems = problems[: args.limit]
    thinking = {"on": True, "off": False, "default": None}[args.thinking]
    settings = rm.Settings(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature,
                           top_p=args.top_p, thinking=thinking, seed=args.seed, top_k=args.top_k or None,
                           keep_texts=args.keep_texts)
    context = args.context or await server_context(args.base_url)
    why = check(problems, settings, args.split, context, specs, args.out)
    if why:
        for w in why:
            print(f"not run: {w}", file=sys.stderr)
        return 2
    warning = fill_warning(problems, args.sequences)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    grader = "math-verify" if rp._math_verify() is not None else "the MATH authors' normalisation"
    print(f"{len(problems)} {args.split} problems (set {digest[:12]}), graded by {grader}; context {context}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "levels": levels, "limit": args.limit, "problem_set": digest,
                "dataset_revision": rp.MATH_REVISION, "methods": args.methods, "settings": settings.__dict__,
                "sequences": args.sequences, "order_seed": args.order_seed, "server": args.server,
                "context": context, "grader": grader}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    post, close = rm.http_post(args.base_url)
    try:
        runs = await run_experiment(post, settings, problems, specs, args.sequences, args.order_seed, args.out,
                                    digest, args.server, energy=rr.nvml_energy(),
                                    preempted=lambda: preemptions(args.base_url))
    finally:
        await close()
    refs = {p.id: p.answer for p in problems}
    title = f"{args.model}, {args.split} set ({len(problems)} problems), thinking {args.thinking}"
    (args.out / "report.md").write_text(rep.markdown(runs, refs, title=title), encoding="utf-8")
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(_main(_arguments(argv)))
