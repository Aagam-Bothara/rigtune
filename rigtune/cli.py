"""rigtune CLI: serve | probe | plan | recalibrate | profiles | bench"""

from __future__ import annotations

import json
import logging
import math
import re
import signal
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rigtune import __version__
from rigtune.models import OBJECTIVES, GiB, ModelSpec, Profile, TrialResult

app = typer.Typer(help="rigtune: an autotuner for LLM serving. It measures serving settings on your GPU and your "
                       "traffic, then serves the fastest behind one OpenAI-compatible API.",
                  no_args_is_help=True, add_completion=False)
console = Console()
err = Console(stderr=True)
if sys.platform == "win32":
    # Legacy Windows consoles are cp1252: print what they cannot encode (α, →) as '?' rather than crash.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(console=err, show_path=False, rich_tracebacks=verbose)])
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _objective(value: Optional[str]) -> Optional[str]:
    if value is not None and value not in OBJECTIVES:
        raise typer.BadParameter(f"objective must be one of {', '.join(OBJECTIVES)}")
    return value


def _fmt(x: Optional[float], nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}{suffix}"


def _trial_table(results: List[TrialResult], winner: Optional[str] = None) -> Table:
    t = Table(title="Calibration", show_lines=False)
    for col in ("stage", "config", "tok/s", "TTFT ms", "TPOT ms", "peak MB", "W", "MHz", "J/tok", "status"):
        t.add_column(col, justify="right" if col not in ("stage", "config", "status") else "left",
                     overflow="fold", min_width=(40 if col == "config" else None))
    for r in results:
        m = r.metrics
        status = "ok" if r.ok else (r.error or "failed").splitlines()[0][:40]
        key = _trial_key(r)
        style = "bold green" if winner and key == winner else None
        t.add_row(r.stage, key, _fmt(m.tok_s), _fmt(m.ttft_ms, 0), _fmt(m.tpot_ms, 1), _fmt(m.peak_mem_mb, 0),
                  _fmt(m.power_w, 0), _fmt(m.sm_clock_mhz, 0), _fmt(m.joules_per_token, 3), status, style=style)
    return t


def _trial_key(r: TrialResult) -> str:
    if r.disagg is not None:
        return r.disagg.key()
    return f"{r.config.key()} x{r.replicas}" if r.replicas > 1 else r.config.key()


def _print_profile(p: Profile) -> None:
    console.print(f"[bold]objective[/]: {p.objective}   [bold]workload[/]: {p.workload}")
    if p.calibration_constraints.get("goal"):  # a custom goal: what it optimised, within which limits
        console.print(f"[bold]goal[/]:      {p.calibration_constraints['goal']}")
    found = p.hardware.backends.get(p.backend)
    libs = ", ".join(f"{k} {v}" for k, v in sorted(found.libs.items())) if found is not None else ""
    console.print(f"[bold]backend[/]: {p.backend} {p.backend_version or ''}" + (f"  [dim]({libs})[/]" if libs else ""))
    console.print(f"[bold]config[/]:  {p.config.key()}")
    if p.disagg is not None:
        console.print(f"[bold]phases[/]:  disaggregated over {p.disagg.connector}; "
                      f"prefill GPU {p.disagg.prefill_gpu}: {p.disagg.prefill.key()}")
        console.print(f"          decode GPU {p.disagg.decode_gpu}: {p.disagg.decode.key()}")
    if p.replicas > 1:
        console.print(f"[bold]layout[/]:  {p.replicas} replicas, one per GPU, behind a load balancer")
    elif p.config.tp > 1:
        console.print(f"[bold]layout[/]:  tensor parallel across {p.config.tp} GPUs")
    if p.options:
        console.print(f"[bold]options[/]: {', '.join(f'{k}={v}' for k, v in sorted(p.options.items()))}")
    console.print(f"[bold]launch[/]:  {' '.join(p.launch_args)}")
    if p.launch_env:
        console.print(f"[bold]env[/]:     {' '.join(f'{k}={v}' for k, v in p.launch_env.items())}")
    for n in p.notes:
        console.print(f"[dim]note: {n}[/]")


def _winner_key(p: Profile) -> str:
    if p.disagg is not None:
        return p.disagg.key()
    return f"{p.config.key()} x{p.replicas}" if p.replicas > 1 else p.config.key()


def _progress_line(stage: str, cfg, res, objective: Optional[str] = None, cons=None) -> str:
    """One calibration trial as the progress log shows it: while it runs, then its result. With an objective,
    the result is the level the objective scores it at, not its fastest level, which may break the limits:
    on an L4 a trial printed as 695 tok/s counted at 219, because 32 users broke the p95 ceiling."""
    if res is None:
        return f"[cyan]{stage:>7}[/] {cfg.key()} ..."
    if not res.ok:
        return f"[red]{stage:>7}[/] {cfg.key()}  FAILED: {(res.error or '').splitlines()[0][:80]}"
    m, verdict = res.metrics, ""
    if objective is not None:
        from rigtune.calibrate.objectives import rank

        ranked = rank([res], objective, cons)
        if ranked:
            m = ranked[0].metrics
            verdict = "" if ranked[0].feasible else "  [yellow]breaks the limits at every level[/]"
    p95 = f" (p95 {_fmt(m.ttft_p95_ms, 0)})" if m.ttft_p95_ms is not None else ""
    energy = f"  {_fmt(m.joules_per_token, 3)} J/tok" if m.joules_per_token else ""
    return (f"[green]{stage:>7}[/] {cfg.key()}  {m.tok_s:.1f} tok/s at {m.concurrency} users  "
            f"TTFT {_fmt(m.ttft_ms, 0)} ms{p95}{energy}{verdict}")


def _progress(stage: str, cfg, res) -> None:
    err.print(_progress_line(stage, cfg, res))


def _progress_for(objective: str, cons):
    """A progress printer that reports each trial at the level the objective scores it."""
    return lambda stage, cfg, res: err.print(_progress_line(stage, cfg, res, objective, cons))


def _calibrating(view, fn, *args, **kwargs):
    """Run a calibration, and turn an eval-set problem found during it (an item too long for every feasible
    context, which needs the plan to see) into a one-line error instead of a traceback under the live view."""
    from rigtune.evalset import EvalSetError
    from rigtune.pipeline import OptionRefused

    try:
        return fn(*args, **kwargs)
    except EvalSetError as exc:
        view.stop()
        err.print(f"[red]--eval-set: {exc}[/]")
        raise typer.Exit(2)
    except OptionRefused as exc:  # found once the hardware is probed, before anything is downloaded or measured
        view.stop()
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2)


def _search_view(objective: str, cons):
    """The search drawn as it runs. On a terminal it is a picture; piped, it is the same per-trial lines as
    before, so logs that scripts grep are unchanged."""
    from rigtune.searchview import SearchView

    return SearchView(err, objective=objective, constraints=cons, fallback=_progress_for(objective, cons))


def _portable_profile(path: Path, model: str) -> Profile:
    """`serve --profile`: the profile at `path`, checked against this machine (cache.portable_problems)."""
    from rigtune import cache as profile_cache
    from rigtune.hardware import probe

    try:
        profile = Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"{path} is not a rigtune profile: {exc}", param_hint="--profile") from None
    if profile.model_id.lower() != model.lower():
        raise typer.BadParameter(f"{path} is a profile of {profile.model_id}, not {model}", param_hint="--profile")
    refuse, warn = profile_cache.portable_problems(profile, probe())
    for w in warn:
        err.print(f"[yellow]--profile: {w}[/]")
    if refuse:
        for r in refuse:
            err.print(f"[red]--profile: not served here: {r}[/]")
        raise typer.Exit(2)
    return profile


def _find_profile(which: str) -> Profile:
    """A path to a profile JSON, or part of a cached profile's model id."""
    from rigtune import cache as profile_cache

    path = Path(which)
    if path.is_file():
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    matches = [pr for _, pr in profile_cache.list_profiles() if which.lower() in pr.model_id.lower()]
    if not matches:
        err.print(f"[red]no cached profile matching {which}[/]")
        raise typer.Exit(2)
    return matches[0]


def _print_tradeoffs(profile: Profile, objective: str, cons) -> None:
    """After a calibration: what the goal's limits cost, from its own recorded trials (rigtune/tradeoff.py). It
    only reads what was measured, so a failure here is said and never loses the calibration before it is saved."""
    from rigtune.tradeoff import to_lines, tradeoffs as trade_offs

    if not profile.calibration_table:
        return
    gate = [n for n in profile.notes if n.startswith("accuracy gate: refusing")]
    try:
        t = trade_offs(profile.calibration_table, objective, cons, accuracy=gate, winner=_winner_key(profile))
        lines = to_lines(t)
    except Exception as exc:  # the table is a reading; the profile is what matters
        err.print(f"[yellow]trade-off table not shown: {exc}[/]", highlight=False)
        return
    console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)


def _trace(which: str) -> None:
    """Redraw a finished search: a path to a profile JSON, or part of a cached profile's model id."""
    from rigtune.searchview import SearchView

    p = _find_profile(which)
    if not p.calibration_table:
        err.print("[yellow]that profile has no calibration table: it was served without a search[/]")
        raise typer.Exit(2)
    from rigtune.goals import recorded_constraints

    cons = recorded_constraints(p.calibration_constraints) if p.calibration_constraints.get("goal") else None
    view = SearchView.from_trials(p.calibration_table, console, winner=_winner_key(p), objective=p.objective,
                                  constraints=cons, notes=p.notes)
    console.print(view.render())


def _workload(name: str, file: Optional[Path] = None, sessions: bool = False, session_pool: Optional[int] = None,
              session_turns: Optional[int] = None, history: Optional[str] = None,
              session_system: Optional[Path] = None):
    from rigtune.calibrate.workload import WORKLOAD_NAMES, get_workload, workload_from_file

    if name not in WORKLOAD_NAMES:
        raise typer.BadParameter(f"workload must be one of {', '.join(WORKLOAD_NAMES)}")
    options = {"--sessions": sessions or None, "--session-pool": session_pool, "--session-turns": session_turns,
               "--history": history, "--session-system": session_system}
    if file is None:
        given = [k for k, v in options.items() if v is not None]
        if given:
            raise typer.BadParameter(f"{', '.join(given)} need a --workload-file", param_hint=given[0])
        return get_workload(name)
    try:
        system = session_system.read_text(encoding="utf-8-sig") if session_system is not None else None
        return workload_from_file(file, template=get_workload(name), session_pool=session_pool,
                                  turns_cap=session_turns, history=history, session_system=system, sessions=sessions)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--workload-file")


def _constraints(ttft_ceiling: Optional[float], tok_s_floor: Optional[float], workload=None,
                 tpot_ceiling: Optional[float] = None, ttft_percentile: int = 95,
                 arrival_rate: Optional[float] = None, strict: bool = False):
    from rigtune.calibrate.objectives import Constraints

    ceiling = ttft_ceiling if ttft_ceiling is not None else (workload.ttft_ceiling_ms if workload else 500.0)
    tpot = tpot_ceiling if tpot_ceiling is not None else (workload.tpot_ceiling_ms if workload else None)
    return Constraints(ttft_ceiling_ms=ceiling, tok_s_floor_abs=tok_s_floor, tpot_ceiling_ms=tpot,
                       ttft_percentile=ttft_percentile, arrival_rate=arrival_rate, strict=strict)


def _goal(objective: Optional[str], optimise: Optional[str], limits: Optional[List[str]],
          ttft_ceiling: Optional[float], ttft_percentile: int, tok_s_floor: Optional[float],
          tpot_ceiling: Optional[float], wl, arrival_rate: Optional[float], strict: bool,
          eval_set: Optional[str] = None, max_accuracy_loss: float = 3.0, gpu_hourly_usd: Optional[float] = None,
          kwh_usd: Optional[float] = None):
    """(objective, constraints, TTFT percentile and accuracy-loss tolerance for the search options): a preset named
    by --objective (balanced when none is), or a goal stated with --optimise and --limit, which is a preset when one
    says the same thing (rigtune/goals.py). Refused before anything is probed; a stated goal is said once before
    calibration."""
    if optimise is None and not limits and gpu_hourly_usd is None and kwh_usd is None:
        return (objective or "balanced", _constraints(ttft_ceiling, tok_s_floor, wl, tpot_ceiling, ttft_percentile,
                                                      arrival_rate, strict=strict), ttft_percentile,
                max_accuracy_loss)
    from rigtune.goals import GoalError, resolve

    try:
        r = resolve(objective, optimise, limits or [], ttft_ceiling=ttft_ceiling, ttft_percentile=ttft_percentile,
                    tok_s_floor=tok_s_floor, tpot_ceiling=tpot_ceiling, workload_ttft_ms=wl.ttft_ceiling_ms,
                    workload_tpot_ms=wl.tpot_ceiling_ms, arrival_rate=arrival_rate, strict=strict,
                    eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, gpu_hourly_usd=gpu_hourly_usd,
                    kwh_usd=kwh_usd)
    except GoalError as exc:
        raise typer.BadParameter(str(exc), param_hint="--optimise/--limit") from None
    err.print(r.line, style="dim", markup=False, highlight=False)
    return r.objective, r.constraints, r.ttft_percentile, r.max_accuracy_loss


def _power_mode(value: str) -> str:
    from rigtune.power import MODES

    if value not in MODES:
        raise typer.BadParameter(f"power must be one of {', '.join(MODES)}")
    return value


def _power_controller(profile: Profile):
    """A controller for serving/comparing a profile that carries a power setting, else None."""
    if profile.config.power_limit_w is None and profile.config.sm_clock_mhz is None:
        return None
    from rigtune.power import controller_for

    if profile.disagg is not None:  # disaggregated: the decode GPU is the one that is capped
        return controller_for(profile.disagg.decode_gpu)
    return controller_for(profile.hardware.gpu.index if profile.hardware.gpu else 0)


WORKLOAD_OPT = typer.Option("default", "--workload", "-w", help="Workload preset; see `rigtune workloads`")
WORKLOAD_FILE_OPT = typer.Option(
    None, "--workload-file",
    help="Calibrate on your own prompts: a JSONL file with one prompt per line, as a string, {\"prompt\": ...} or "
         "{\"messages\": [...]}. --workload then sets only the concurrency levels and latency ceilings.")
SESSIONS_OPT = typer.Option(
    False, "--sessions",
    help="Replay every line of --workload-file as a session, turn by turn, even without a \"session\" key (a file "
         "whose lines have one is a session file anyway). Without it, multi-turn {\"messages\": [...]} lines are "
         "flattened into one prompt each, as before.")
SESSION_POOL_OPT = typer.Option(
    None, "--session-pool", min=1,
    help="Session files: sessions live at once per level (default: one per concurrency slot). A session's next turn "
         "waits behind the others, so its reuse distance is this many minus one; a larger pool tests whether its "
         "cached history survives. A level then draws on at least twice this many sessions.")
SESSION_TURNS_OPT = typer.Option(None, "--session-turns", min=1,
                                 help="Session files: turns replayed per session (default 8)")
HISTORY_OPT = typer.Option(
    None, "--history",
    help="Session files: what a later turn carries as the earlier answers: generated (the model's own, default) or "
         "recorded (the file's, the same for every configuration)")
SESSION_SYSTEM_OPT = typer.Option(None, "--session-system",
                                  help="Session files: a text file whose contents are every session's system prompt, "
                                       "in place of the file's own")
POWER_OPT = typer.Option(
    "off", "--power", callback=_power_mode,
    help="Energy tuning: off | cap (power limit) | clock (locked SM clock) | both. Needs root; machine-wide; "
         "restored on exit and by `rigtune power reset`.",
)


def _phases(value: str) -> str:
    from rigtune.disagg import PHASES

    if value not in PHASES:
        raise typer.BadParameter(f"phases must be one of {', '.join(PHASES)}")
    return value


TPOT_OPT = typer.Option(None, "--tpot-ceiling", help="Per-token decode latency ceiling in ms (default: the workload's)")
PHASES_OPT = typer.Option(
    "unified", "--phases", callback=_phases,
    help="unified (default): one engine, prefill and decode knobs tuned separately. disaggregated: prefill and "
         "decode on separate GPUs joined by KV transfer (vLLM, two GPUs, a KV connector). auto: measure both, "
         "keep the better.",
)
KV_OPT = typer.Option("nixl", "--kv-connector", help="KV-cache transfer connector for --phases disaggregated/auto")


def _on_off(value: str) -> str:
    if value not in ("on", "off"):
        raise typer.BadParameter("expected on or off")
    return value


def _on_off_all(value: str) -> str:
    if value not in ("on", "off", "all"):
        raise typer.BadParameter("expected on, off or all")
    return value


def _layout(value: str) -> str:
    from rigtune.layout import LAYOUTS

    if value not in LAYOUTS:
        raise typer.BadParameter(f"layout must be one of {', '.join(LAYOUTS)}")
    return value


def _quant_list(value: str) -> str:
    from rigtune.gguf import GGUF_QUANTS

    known = {"auto", "bf16", "fp16", "fp8", "awq", "gptq", "w8a8", *GGUF_QUANTS}
    if value != "auto":
        bad = [q for q in value.split(",") if q.strip() not in known]
        if bad:
            raise typer.BadParameter(f"unknown precision {', '.join(bad)}; known: auto, {', '.join(sorted(known))}")
    return value


def _duration(value: Optional[str]) -> Optional[float]:
    """Seconds from '90s', '10m', '1h' or a plain number of seconds."""
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(value))
    if not m or float(m.group(1)) <= 0:
        raise typer.BadParameter("expected a duration such as 90s, 10m or 1h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def _rate(value: Optional[str]) -> Optional[float]:
    """Requests per second from '0.5', '0.5/s', '12/min' or '720/h'."""
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?|\.\d+)\s*(?:/\s*(s|sec|min|m|h|hr|hour))?\s*", str(value))
    if not m or float(m.group(1)) <= 0:
        raise typer.BadParameter("expected requests per second such as 0.5, or a rate such as 12/min or 720/h")
    per = {None: 1, "s": 1, "sec": 1, "min": 60, "m": 60, "h": 3600, "hr": 3600, "hour": 3600}[m.group(2)]
    return float(m.group(1)) / per


def _check_rate(objective: str, rate: Optional[float]) -> None:
    if rate is not None and objective != "efficiency":
        raise typer.BadParameter(f"--arrival-rate scores energy per request, so it applies to --objective efficiency "
                                 f"only, not {objective}", param_hint="--arrival-rate")


def _percentile(value: int) -> int:
    if value not in (50, 95):
        raise typer.BadParameter("expected 95 (the default) or 50 (the median)")
    return value


def _kv_tier(value: str) -> str:
    if value not in ("off", "auto"):
        raise typer.BadParameter("expected off or auto")
    return value


def _check_kv_tier(kv_tier: str, wl) -> None:
    """--kv-tier auto sizes a tier from the working set session replay measures: it needs a session file."""
    if kv_tier != "off" and not wl.sessions:
        raise typer.BadParameter("needs a session file (--workload-file with sessions): a tier is sized from "
                                 "the working set session replay measures", param_hint="--kv-tier")


def _in_flight(value: Optional[str]) -> Optional[str]:
    """--max-in-flight: a positive whole number, or `measured`."""
    if value is None or value == "measured":
        return value
    try:
        if int(value) >= 1:
            return value
    except ValueError:
        pass
    raise typer.BadParameter(f"{value!r} is neither a positive whole number nor 'measured'")


def _check_min_requests(min_requests: Optional[int], wl) -> None:
    """--min-requests on a session file, or more than its prompts can give fresh, fails now, not after the download."""
    from rigtune.calibrate.workload import with_minimum_requests

    try:
        with_minimum_requests(wl, min_requests)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--min-requests") from None


def _seed(command: Optional[str]) -> Optional[str]:
    """--seed read now, so a command rigtune cannot launch fails before any download, with the reason."""
    if not command:
        return None
    from rigtune.seed import parse_command

    try:
        parse_command(command)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--seed") from None
    return command


def _opts(quant: str, kv_quant: str, speculative: str, prefix_cache: str, combine: str = "on",
          budget: Optional[float] = None, ttft_percentile: int = 95, confirm: str = "on", all_levels: bool = False,
          explore: str = "off", quality_probe: str = "off", eval_set: Optional[str] = None,
          max_accuracy_loss: float = 3.0, eval_items: int = 300, report_reuse: bool = False,
          kv_tier: str = "off", seed: Optional[str] = None,
          arrival_rate: Optional[float] = None, rate_check: str = "on",
          confirm_rounds: int = 1, min_requests: Optional[int] = None, eval_attempts: int = 3,
          drafts: str = "table", draft: Optional[str] = None, hub: str = "on",
          kernel: Optional[str] = None):
    from rigtune.pipeline import SearchOptions

    if eval_set:
        from rigtune.evalset import EvalSetError, load_eval_set

        try:  # read and check every item now, not after the weights have downloaded
            load_eval_set(eval_set, eval_items)
        except EvalSetError as exc:
            raise typer.BadParameter(str(exc), param_hint="--eval-set") from None
    return SearchOptions(
        quants=None if quant == "auto" else [q.strip() for q in quant.split(",") if q.strip()],
        kv_quant=kv_quant == "on", speculative=speculative != "off", speculative_all=speculative == "all",
        prefix_cache=prefix_cache == "on", combine=combine == "on", budget_s=budget, ttft_percentile=ttft_percentile, confirm=confirm == "on",
        all_levels=all_levels, explore=explore == "on", quality_probe=quality_probe == "on",
        eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, eval_items=eval_items, report_reuse=report_reuse,
        kv_tier=kv_tier, seed=_seed(seed),
        arrival_rate=arrival_rate, rate_check=rate_check == "on",
        confirm_rounds=confirm_rounds, min_requests=min_requests, eval_attempts=eval_attempts,
        drafts=drafts, draft=draft, hub=hub, kernel=kernel,
    )


def _drafts(value: str) -> str:
    if value not in ("table", "verified"):
        raise typer.BadParameter("expected table or verified")
    return value


DRAFTS_OPT = typer.Option("table", "--drafts", callback=_drafts,
                          help="table: existing proposals; verified: check files and pin second models.")
DRAFT_OPT = typer.Option(None, "--draft", help="Extra draft repository or directory; enables verified checking.")
HUB_OPT = typer.Option("on", "--hub", callback=_on_off,
                       help="off: pair evidence from local files/cache only (also HF_HUB_OFFLINE=1).")


def _kernel(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    from rigtune.kernels import canonical

    try:
        return canonical(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--kernel") from None


KERNEL_OPT = typer.Option(
    None, "--kernel", callback=_kernel,
    help="Try one engine kernel setting on each tuned engine's leader, e.g. cgmax=64 (SGLang's decode CUDA-graph "
         "cap) or attn=TRITON_ATTN (vLLM). Kept only if it measurably wins. Has not run on hardware.")


KV_TIER_OPT = typer.Option(
    "off", "--kv-tier", callback=_kv_tier,
    help="Session files: auto tries a host-RAM KV-cache tier (vLLM's offloading connector, SGLang's "
         "hierarchical cache), and a disk tier under rigtune's cache dir when host RAM cannot hold the "
         "working set, only when the leader's working set outgrows the GPU pool. Exercised by tests only; "
         "has not run on hardware. off (default): no tier is ever tried.")
SEED_OPT = typer.Option(
    None, "--seed",
    help="The command you would run yourself, or a model's published recipe, e.g. \"vllm serve MODEL "
         "--max-model-len 32768 --speculative-config '{...}'\". It is measured first, so the pick is never "
         "measured below it, and the later stages improve from it. Settings it does not keep are noted. Has not run "
         "on hardware.")
REPORT_REUSE_OPT = typer.Option(
    False, "--report-reuse",
    help="Launch the engines with their per-request cached-token reports for a flat file or a preset too (a session "
         "file always has them), so each level records how much of its prompts the prefix cache served. The profile "
         "is cached under its own name.")
QUANT_OPT = typer.Option("auto", "--quant", callback=_quant_list,
                         help="Weight precisions calibration may choose. auto: everything supported except the "
                              "Hub's pre-quantized checkpoints, 4-bit AWQ/GPTQ and 8-bit W8A8, which can cost quality "
                              "(auto,awq,gptq,w8a8 adds them; with --eval-set, auto includes them and the accuracy "
                              "gate decides). Or a list such as bf16, bf16,fp8 or gptq (also fp16, "
                              "Q4_K_M, Q5_K_M, Q6_K, Q8_0).")
KVQ_OPT = typer.Option("on", "--kv-quant", callback=_on_off,
                       help="Try quantized KV caches (fp8 on vLLM and SGLang, q8_0 and q4_0 on llama.cpp)")
SPEC_OPT = typer.Option("on", "--speculative", callback=_on_off_all,
                        help="Try speculative decoding (n-gram prompt lookup, a small draft model, suffix decoding). "
                             "all: also SGLang's n-gram, EAGLE3 and draft model, vLLM's EAGLE3 head, and the "
                             "checkpoint's own MTP head (on vLLM for any family vLLM has a draft for, on SGLang for "
                             "Qwen3.5), choosing engines after speculation (opt-in)")
PREFIX_OPT = typer.Option("on", "--prefix-cache", callback=_on_off,
                          help="Keep prefix caching on and tune it for workloads whose prompts share a prefix")
QUALITY_OPT = typer.Option("off", "--quality-probe", callback=_on_off,
                           help="Measure how far each weight precision's answers drift from the most faithful "
                                "precision that runs, on your own calibration prompts, and record it in the "
                                "profile. It refuses nothing: the comparison saturates (identical weights "
                                "differ from themselves on ~10% of answer openings, cheaper ones on 83-100%). "
                                "Costs two short generation passes per precision.")


def _max_loss(value: float) -> float:
    if not 0 < value < 100:
        raise typer.BadParameter("expected a loss in accuracy points, above 0 and below 100 (3 = three points)")
    return value


EVAL_SET_OPT = typer.Option(None, "--eval-set",
                            help="Gate weight precisions on labelled accuracy: gsm8k (bundled, 1319 problems) or a "
                                 "JSONL file of {prompt or messages, answer, grader} items. Each precision answers "
                                 "them during calibration, and one below the most precise format that runs is "
                                 "served only if its loss is certified to be under --max-accuracy-loss. With "
                                 "--quant auto it also tries the AWQ, GPTQ and W8A8 checkpoints auto otherwise "
                                 "leaves out.")
MAX_LOSS_OPT = typer.Option(3.0, "--max-accuracy-loss", callback=_max_loss,
                            help="With --eval-set: the largest accuracy loss, in points, a cheaper precision may "
                                 "carry. It is admitted only when the lower end of the 95% interval for its "
                                 "difference from the reference is above minus this.")
EVAL_ITEMS_OPT = typer.Option(300, "--eval-items", min=0,
                              help="With --eval-set: grade the first N items (0 = all). 300 items certify at best "
                                   "about 1.3 points, when no answer changes.")
EVAL_ATTEMPTS_OPT = typer.Option(3, "--eval-attempts", min=0,
                                 help="With --eval-set: configurations the final check may grade, faster ones first, "
                                      "before falling back to one already certified (each costs a trial and a "
                                      "grading pass)")
LAYOUT_OPT = typer.Option("single", "--layout", callback=_layout,
                          help="Multi-GPU layout: single (default), replicas (one engine per GPU behind a load "
                               "balancer), tp (tensor parallel), auto (measure both, keep the better)")
COMBINE_OPT = typer.Option("on", "--combine", callback=_on_off,
                           help="After tuning one setting at a time, measure the leader with each adopted "
                                "change undone, then combinations of the settings that came close on their "
                                "own (up to 8 extra trials in all)")
CONFIRM_OPT = typer.Option("on", "--confirm", callback=_on_off,
                           help="Re-measure the best three configurations at the end and choose on those "
                                "runs (on|off)")
CONFIRM_ROUNDS_OPT = typer.Option(1, "--confirm-rounds", min=1,
                                  help="Fresh runs of each configuration --confirm re-measures; each is judged on "
                                       "its median run (one engine on one GPU only)")
MIN_REQUESTS_OPT = typer.Option(None, "--min-requests", min=1,
                                help="Send at least this many requests at every measured concurrency (flat "
                                     "workloads; 72 is the fewest that bound a p95 at 95% confidence)")
STRICT_OPT = typer.Option(False, "--strict-constraints",
                          help="Fail the calibration when no measured configuration meets the TTFT/TPOT ceilings "
                               "and throughput floor, instead of serving the least-violating one")
EXPLORE_OPT = typer.Option("off", "--explore", callback=_on_off,
                           help="After the other stages, spend about 30% more trials on random configurations of "
                                "the leading engine and precision, for settings that only pay together (on|off; off "
                                "by default: on an RTX 4090 it did not close the gap to random search)")
ALL_LEVELS_OPT = typer.Option(False, "--all-levels",
                              help="Measure every concurrency level of every trial instead of stopping at the busiest "
                                   "level that meets the objective; slower, for checking that shortcut")
BUDGET_OPT = typer.Option(None, "--budget", callback=_duration,
                          help="Stop calibrating after about this long (90s, 10m, 1h). Stages run in a fixed order "
                               "(precision, memory, batch, then variations and combinations); later trials are "
                               "skipped, the best so far wins, and the profile lists what was skipped.")
RATE_OPT = typer.Option(None, "--arrival-rate", callback=_rate,
                        help="efficiency: the requests per second the server will see (0.5, 12/min, 720/h). Scores "
                             "each configuration by joules per request at that rate, the card's idle draw between "
                             "requests and its wind-down after them included, instead of joules per token at full "
                             "load. Not yet run on hardware.")
RATE_CHECK_OPT = typer.Option("on", "--rate-check", callback=_on_off,
                              help="With --arrival-rate: launch the pick once more and send it up to 24 requests at "
                                   "Poisson arrival times at that rate (at most 5 minutes), and record the measured "
                                   "J/request beside the predicted one (on|off)")
OBJECTIVE_HELP = ("A preset goal: balanced (default; most tok/s within the TTFT ceiling), throughput, latency or "
                  "efficiency. Leave it out with --optimise")
OPTIMISE_OPT = typer.Option(
    None, "--optimise", "--optimize",
    help="State the goal instead of naming a preset: the one metric to optimise (tok_s, ttft_p95_ms, ttft_p50_ms, "
         "tpot_ms, request_latency_ms, joules_per_token, cost_per_mtok), within every --limit. A goal a preset expresses is that "
         "preset, with its cached profile; the workload's TPOT ceiling still applies")
GPU_PRICE_OPT = typer.Option(None, "--gpu-hourly-usd",
                             help="What the GPU costs per hour, in US dollars: prices cost_per_mtok (dollars per "
                                  "million output tokens) for --optimise or --limit")
ENERGY_PRICE_OPT = typer.Option(None, "--kwh-usd",
                                help="With --gpu-hourly-usd: electricity per kWh, so cost_per_mtok counts the "
                                     "power measured too (needs NVML power readings)")
LIMIT_OPT = typer.Option(
    None, "--limit",
    help="A limit on the goal, quoted: \"ttft_p95_ms<=800\", \"tok_s>=200\", \"tok_s>=50%best\", \"power_w<=250\", "
         "\"mem_mb<=20GiB\" (<= or >=; units ms, s, W, MiB, GiB). Repeatable; added to --objective's own goal when "
         "given with it")
TTFT_OPT = typer.Option(None, "--ttft-ceiling", help="balanced: TTFT ceiling in ms (default: the workload's)")
TTFT_PCT_OPT = typer.Option(95, "--ttft-percentile", callback=_percentile,
                            help="balanced: which time to first token the ceiling applies to: 95 (default; at "
                                 "most 1 request in 20 may be slower) or 50 (the median)")


# --------------------------------------------------------------------------- commands


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print the rigtune version."""
    console.print(__version__)


@app.command()
def workloads() -> None:
    """List workload presets."""
    from rigtune.calibrate.workload import workload_table

    t = Table(title="Workload presets (latency ceilings in ms)")
    for col in ("name", "prefill", "shared", "decode", "concurrency", "TTFT", "TPOT"):
        t.add_column(col, justify="right" if col != "name" else "left")
    for w in workload_table():
        t.add_row(str(w["name"]), str(w["prefill_tokens"]), str(w.get("shared_prefix_tokens") or "-"),
                  str(w["decode_tokens"]),
                  "/".join(str(c) for c in w["concurrencies"]), f"{w['ttft_ceiling_ms']:.0f}",
                  f"{w['tpot_ceiling_ms']:.0f}" if w.get("tpot_ceiling_ms") else "-")
    console.print(t)


@app.command()
def selfcheck(
    users: str = typer.Option("1,32,64,128", "--users", help="Comma-separated numbers of users to check"),
    ttft_ms: float = typer.Option(20.0, "--ttft-ms", help="The fake engine's time to first token"),
    itl_ms: float = typer.Option(10.0, "--itl-ms", help="Its time between tokens; 0 sends as fast as it can"),
    tokens: int = typer.Option(128, "--tokens", help="Tokens in each answer"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Check that the load generator measures a fake engine of known speed faithfully at each number of users.
    Exits 1 when a level misses a threshold."""
    from rigtune import selfcheck as sc
    from rigtune.hardware import probe_cpu

    try:
        levels = [int(u) for u in users.split(",") if u.strip()]
    except ValueError:
        raise typer.BadParameter(f"--users takes comma-separated numbers, not {users!r}")
    reports = sc.run(levels, ttft_ms=ttft_ms, itl_ms=itl_ms, tokens=tokens)
    cpu = probe_cpu().model_name
    if as_json:
        console.print_json(json.dumps({"cpu": cpu, "ttft_ms": ttft_ms, "itl_ms": itl_ms, "tokens": tokens,
                                       "levels": [r.to_dict() for r in reports]}))
    else:
        t = Table(title=f"Client self-check on {cpu} (engine: {ttft_ms:g} ms to first token, {itl_ms:g} ms a token, "
                        f"{tokens} tokens)")
        for col in ("users", "TTFT p50 ms", "TTFT p95 ms", "engine p95 ms", "p95 error ms", "tok/s ratio",
                    "loop lag p99 ms", "result"):
            t.add_column(col, justify="right" if col != "result" else "left")
        for r in reports:
            t.add_row(str(r.users), _fmt(r.client_ttft_p50_ms), _fmt(r.client_ttft_p95_ms),
                      _fmt(r.engine_ttft_p95_ms), f"{r.ttft_error_p95_ms:.1f} (< {r.ttft_allowance_ms:.1f})",
                      f"{r.throughput_ratio:.3f}", _fmt(r.loop_lag_p99_ms),
                      "ok" if r.passed else "; ".join(r.problems()))
        console.print(t)
        if itl_ms == 0:
            for r in reports:
                console.print(f"client ceiling at {r.users} users: {r.client_tok_s:.0f} tokens/s")
        missed = [r for r in reports if not r.passed]
        if missed:
            console.print(f"[yellow]{len(missed)} of {len(reports)} levels missed a threshold.[/] At "
                          f"{missed[-1].users} users, try: {sc.suggestion(missed[-1].users)}.")
    if any(not r.passed for r in reports):
        raise typer.Exit(1)


@app.command()
def conformance(
    engine: str = typer.Argument(..., help="vllm or sglang"),
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", help="The small model most checks launch"),
    hybrid_model: Optional[str] = typer.Option(None, "--hybrid-model", help="A hybrid-attention model with its own MTP "
                                               "head, e.g. Qwen/Qwen3.5-9B: adds C10 and MTP to C06"),
    multimodal_model: Optional[str] = typer.Option(None, "--multimodal-model", help="A multimodal checkpoint for "
                                                   "vLLM's text-only launch (C08)"),
    out: Path = typer.Option(Path("conformance"), "--out", help="Where the record and the engine logs go"),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated check ids to run, e.g. C01,C04"),
    startup_timeout: float = typer.Option(900.0, "--startup-timeout", help="Seconds each launch may take to start"),
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    hub: str = HUB_OPT,
) -> None:
    """Check on this machine's GPU that an installed engine release does what rigtune plans for, and write the record
    that adds it to the checked releases. Exits 0 when the release is certified. Has never run on a GPU."""
    from rigtune import conformance as cf
    from rigtune.backends import registry
    from rigtune.hardware import probe as _probe

    if engine not in ("vllm", "sglang"):
        err.print(f"[red]conformance checks vllm or sglang, not {engine!r}[/]")
        raise typer.Exit(2)
    hw = _probe()
    if hw.gpu is None or hw.gpu.vendor != "nvidia":
        err.print("[red]rigtune conformance needs an NVIDIA GPU: every check but the probe launches the engine[/]")
        raise typer.Exit(2)
    found = hw.backends.get(engine)
    if found is None or not found.available:
        err.print(f"[red]{engine} is not available here: {found.reason if found else 'not found'}[/]")
        raise typer.Exit(2)
    roles = {"small": model, "hybrid": hybrid_model, "multimodal": multimodal_model}
    backend = registry()[engine]
    run = cf.Conformance(engine, backend, hw, _conformance_models(backend, hw, roles), out,
                         drafts=drafts, draft=draft, hub=hub,
                         startup_timeout=startup_timeout, reporting=registry(report_reuse=True)[engine],
                         only=[c.strip() for c in only.split(",") if c.strip()] if only else None)

    def progress(r) -> None:
        colour = {"pass": "green", "fail": "red", "skip": "yellow"}[r.status]
        console.print(f"[{colour}]{r.status:4}[/] {r.id} {r.name}" + (f": {r.detail}" if r.detail else ""),
                      highlight=False)

    results = run.run(progress=progress)
    facts = backend.facts()
    version = (facts.version if facts is not None and facts.version else None) or hw.backend_version(engine) or "unknown"
    rec = cf.record(engine, version, hw, results, facts, {k: v for k, v in roles.items() if v})
    path = cf.write(out, rec)
    verdict = "certified" if rec["certified"] else "not certified (C01 to C05 must pass and C07 must not fail)"
    console.print(f"{engine} {version} on {rec['card']}: {verdict}. Record: {path}")
    raise typer.Exit(0 if rec["certified"] else 1)


def _conformance_models(backend, hw, roles: dict) -> dict:
    """PreparedModel per role ("small", "hybrid", "multimodal") for the models given."""
    return {role: backend.prepare(ModelSpec(hf_id=hf_id), hw) for role, hf_id in roles.items() if hf_id}


@app.command()
def probe(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    engines: bool = typer.Option(False, "--engines", help="Also ask vLLM and SGLang, in their own Python, what they "
                                 "offer (once per install; kept under rigtune's home)"),
    reprobe: bool = typer.Option(False, "--reprobe", help="With --engines: ask again instead of reading what was kept"),
) -> None:
    """Print the HardwareDescriptor."""
    from rigtune.hardware import hardware_hash, probe as _probe
    from rigtune.selector import explain

    hw = _probe()
    facts = _engine_facts(reprobe) if engines or reprobe else {}
    if as_json:
        data = hw.model_dump(mode="json")
        data["hardware_hash"] = hardware_hash(hw)
        if engines or reprobe:
            data["engine_facts"] = {name: f.model_dump(mode="json") if f else None for name, f in facts.items()}
            data["feature_checks"] = {name: {check: {"probe": _plain(p), "source": _plain(s)}
                                             for check, (p, s) in report.items()}
                                      for name, report in _feature_reports().items()}
        console.print_json(json.dumps(data))
        return
    console.print(f"[bold]hardware hash[/]: {hardware_hash(hw)}   [dim]{hw.os}, python {hw.python}[/]")
    if hw.gpus:
        for g in hw.gpus:
            console.print(f"[bold]GPU {g.index}[/]: {g.name} (cc {g.cc[0]}.{g.cc[1]}) "
                          f"{g.vram_free_bytes / GiB:.1f}/{g.vram_total_bytes / GiB:.1f} GiB free, "
                          f"driver {g.driver_version or '?'}, CUDA {g.cuda_version or '?'}")
    else:
        console.print("[bold]GPU[/]: none")
    c = hw.cpu
    console.print(f"[bold]CPU[/]: {c.model_name} ({c.physical_cores}c/{c.logical_cores}t, {c.arch}) "
                  f"AVX2={'y' if c.avx2 else 'n'} AVX512={'y' if c.avx512 else 'n'}  "
                  f"RAM {c.ram_free_bytes / GiB:.1f}/{c.ram_total_bytes / GiB:.1f} GiB free")
    console.print("[bold]backends[/]:")
    for line in explain(hw):
        console.print(f"  {line}")
    _print_checked_engines(hw)
    if engines or reprobe:
        _print_engine_facts(facts)
        _print_feature_reports(_feature_reports())


def _engine_facts(refresh: bool) -> dict:
    """vLLM's and SGLang's answers to the engine probe (rigtune.backends.facts), None for one not found or with
    probing off."""
    from rigtune.backends import registry

    reg = registry()
    return {name: reg[name].facts(refresh=refresh) for name in ("vllm", "sglang")}


def _feature_reports() -> dict:
    """Each engine's feature checks, answered by the probe and by the source (backends' feature_report)."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl

    return {"vllm": vl.feature_report(), "sglang": sg.feature_report()}


def _plain(value):
    return sorted(value) if isinstance(value, (set, frozenset)) else value


def _print_feature_reports(reports: dict) -> None:
    t = Table(title="Feature checks: the probe's answer, the source's, and which one rigtune uses",
              title_justify="left")
    for col in ("engine", "check", "probe", "source", "used"):
        t.add_column(col)
    for name, report in reports.items():
        for check, (probed, source) in report.items():
            used = "source (no probe answer)" if probed is None else (
                "probe" if _plain(probed) == _plain(source) else "probe; [yellow]the source disagrees[/]")
            t.add_row(name, check, "-" if probed is None else str(_plain(probed)), str(_plain(source)), used)
    console.print(t)


def _print_engine_facts(facts: dict) -> None:
    t = Table(title="What each engine says it offers (asked in its own Python)", title_justify="left")
    for col in ("engine", "version", "flags", "architectures", "speculative methods", "asked", "not answered"):
        t.add_column(col)
    for name, f in facts.items():
        if f is None:
            t.add_row(name, "-", "-", "-", "-", "-", "not asked: not found, or RIGTUNE_ENGINE_PROBE=off")
            continue
        missing = f.unanswered()
        t.add_row(name, f.version or "?", str(len(f.flags)) if f.flags is not None else "-",
                  str(len(f.archs)) if f.archs is not None else "-",
                  ", ".join(f.spec_methods) if f.spec_methods is not None else "-",
                  f"{f.probed_at or '?'} ({f.seconds if f.seconds is not None else '?'} s)",
                  "; ".join(f"{k}: {f.errors.get(k, '')}" for k in missing) or "none")
    console.print(t)


def _print_checked_engines(hw) -> None:
    """The engine releases rigtune was checked with, and a warning for an installed one that is not among them."""
    from rigtune import engines

    t = Table(title="Engine releases rigtune was checked with", title_justify="left")
    for col in ("engine", "card", "date", "torch", "transformers", "checks"):
        t.add_column(col)
    for engine in engines.CHECKED_ENGINES:
        for r in engines.tested().get(engine, []):
            failed = [k for k, v in r.checks.items() if v == "fail"]
            t.add_row(f"{engine} {r.version}", r.card, r.date, r.torch or "-", r.transformers or "-",
                      f"{len(r.checks)} run" + (f", failed: {'; '.join(failed)}" if failed else ""))
    console.print(t)
    for engine in engines.CHECKED_ENGINES:
        note = engines.support_note(engine, hw.backend_version(engine))
        if note:
            console.print(f"[yellow]note[/]: {note}")


@app.command()
def plan(
    model: str,
    backend: Optional[str] = typer.Option(None, "--backend", help="Force one backend"),
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    as_json: bool = typer.Option(False, "--json"),
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    hub: str = HUB_OPT,
    speculative: str = SPEC_OPT,
) -> None:
    """Print feasible configs (after the memory planner) without running them."""
    from rigtune.hardware import probe as _probe
    from rigtune.pipeline import prepare_and_plan, select

    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    if drafts == "verified" or draft:
        from rigtune.pairing import pin_target
        spec = pin_target(spec, draft=draft, hub=hub)
    candidates, reg = select(hw, spec, force=backend)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    result = prepare_and_plan(hw, spec, candidates, reg, workload=wl, speculative_all=speculative == "all",
                              drafts=drafts, draft=draft, hub=hub)
    from rigtune.memory import layout_note

    # A hybrid-attention model whose KV fell back to the every-layer-full estimate (rigtune/memory.py).
    notes = {name: n for name, p in result.prepared.items() if (n := layout_note(p, reg[name].memory_model(hw)))}
    if as_json:
        out = {
            name: [{"config": c.model_dump(), "estimate": e.model_dump()} for c, e in cfgs]
            for name, cfgs in result.feasible.items()
        }
        out["_errors"] = result.errors
        if notes:
            out["_notes"] = notes
        console.print_json(json.dumps(out))
        return
    console.print(f"candidates: {', '.join(candidates)}   budget: {hw.available_memory_bytes / GiB:.1f} GiB   "
                  f"workload: {wl.describe()}")
    for name in candidates:
        if name in result.errors:
            console.print(f"[red]{name}: {result.errors[name]}[/]")
            continue
        cfgs = result.feasible.get(name, [])
        t = Table(title=f"{name}: {len(cfgs)}/{result.considered.get(name, 0)} feasible")
        for col in ("config", "weights GiB", "kv GiB", "workspace GiB", "margin GiB", "total GiB", "budget GiB"):
            t.add_column(col, justify="right" if col != "config" else "left", overflow="fold",
                         min_width=(40 if col == "config" else None))
        for c, e in cfgs:
            t.add_row(c.key(), f"{e.weights / GiB:.2f}", f"{e.kv_cache / GiB:.2f}", f"{e.runtime_workspace / GiB:.2f}",
                      f"{e.safety_margin / GiB:.2f}", f"{e.total / GiB:.2f}", f"{e.budget / GiB:.2f}")
        console.print(t)
        if name in notes:
            console.print(f"note: {notes[name]}", style="yellow", markup=False)


@app.command()
def bench(
    model: str,
    objective: Optional[str] = typer.Option(None, "--objective", callback=_objective, help=OBJECTIVE_HELP),
    optimise: Optional[str] = OPTIMISE_OPT,
    limit: Optional[List[str]] = LIMIT_OPT,
    gpu_hourly_usd: Optional[float] = GPU_PRICE_OPT,
    kwh_usd: Optional[float] = ENERGY_PRICE_OPT,
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    ttft_percentile: int = TTFT_PCT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None, help="latency/efficiency: absolute tok/s floor"),
    save: bool = typer.Option(False, "--save", help="Also write the winning profile to the cache"),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    kernel: Optional[str] = KERNEL_OPT,
    hub: str = HUB_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
    confirm: str = CONFIRM_OPT,
    confirm_rounds: int = CONFIRM_ROUNDS_OPT,
    min_requests: Optional[int] = MIN_REQUESTS_OPT,
    strict_constraints: bool = STRICT_OPT,
    all_levels: bool = ALL_LEVELS_OPT,
    explore: str = EXPLORE_OPT,
    budget: Optional[str] = BUDGET_OPT,
    quality_probe: str = QUALITY_OPT,
    report_reuse: bool = REPORT_REUSE_OPT,
    kv_tier: str = KV_TIER_OPT,
    seed: Optional[str] = SEED_OPT,
    eval_set: Optional[str] = EVAL_SET_OPT,
    max_accuracy_loss: float = MAX_LOSS_OPT,
    eval_items: int = EVAL_ITEMS_OPT,
    eval_attempts: int = EVAL_ATTEMPTS_OPT,
    arrival_rate: Optional[str] = RATE_OPT,
    rate_check: str = RATE_CHECK_OPT,
) -> None:
    """Run calibration and print the table; do not serve."""
    from rigtune import cache as profile_cache
    from rigtune.hardware import probe as _probe
    from rigtune.pipeline import (
        calibrate,
        check_distributed_options,
        check_goal_measurable,
        prepare_and_plan,
        select,
    )

    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    objective, cons, ttft_percentile, max_accuracy_loss = _goal(
        objective, optimise, limit, ttft_ceiling, ttft_percentile, tok_s_floor, tpot_ceiling, wl, arrival_rate,
        strict_constraints, eval_set, max_accuracy_loss, gpu_hourly_usd, kwh_usd)
    _check_rate(objective, arrival_rate)
    _check_kv_tier(kv_tier, wl)
    _check_min_requests(min_requests, wl)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    if drafts == "verified" or draft:
        from rigtune.pairing import pin_target
        spec = pin_target(spec, draft=draft, hub=hub)
    candidates, reg = select(hw, spec, force=backend, workload=wl, report_reuse=report_reuse)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    opts = _opts(quant, kv_quant, speculative, prefix_cache, combine, budget, ttft_percentile, confirm, all_levels, explore, quality_probe,
                 eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, eval_items=eval_items,
                 eval_attempts=eval_attempts,
                 report_reuse=report_reuse, kv_tier=kv_tier, seed=seed, arrival_rate=arrival_rate, rate_check=rate_check,
                 confirm_rounds=confirm_rounds, min_requests=min_requests, drafts=drafts, draft=draft, hub=hub,
                 kernel=kernel)
    try:  # as resolve_profile does: refused before the weights download and the search runs
        check_distributed_options(opts, hw, layout, phases)
        check_goal_measurable(objective, cons, opts, hw, layout, phases)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if layout != "single" and opts.kv_tier != "off":  # as pipeline._resolve_layout does
        err.print(f"[yellow]--kv-tier {opts.kv_tier} is not tried under --layout {layout}: a KV-cache tier's "
                  "host RAM is budgeted for one engine on one GPU[/]")
        opts.kv_tier = "off"
    result = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl, quants=opts.search_quants(),
                              speculative_all=opts.speculative_all, drafts=opts.drafts, draft=opts.draft, hub=opts.hub)
    err.print(f"{len(result.all_feasible)}/{result.total_considered} configs feasible; "
              f"calibrating for {objective} on workload {wl.name}")
    view = _search_view(objective, cons)
    profile = _calibrating(view, calibrate, hw, spec, objective, result, reg, workload=wl, constraints=cons,
                           progress=view.progress, power_mode=power, options=opts)
    if phases != "unified":
        from rigtune.disagg import calibrate_disaggregated

        profile = calibrate_disaggregated(hw, spec, objective, profile, reg, workload=wl, constraints=cons,
                                          phases=phases, connector=kv_connector, progress=view.progress,
                                          power_mode=power)
    if layout != "single":
        from rigtune.layout import calibrate_layout

        profile = calibrate_layout(hw, objective, profile, reg, layout, workload=wl, constraints=cons,
                                   progress=view.progress)
    view.stop()
    console.print(_trial_table(profile.calibration_table, winner=_winner_key(profile)))
    _print_profile(profile)
    if save:  # before the table: nothing printed after a calibration can lose it
        profile_cache.save(profile, ModelSpec(hf_id=model))
    _print_tradeoffs(profile, objective, cons)


@app.command()
def recalibrate(
    model: str,
    objective: Optional[str] = typer.Option(None, "--objective", callback=_objective, help=OBJECTIVE_HELP),
    optimise: Optional[str] = OPTIMISE_OPT,
    limit: Optional[List[str]] = LIMIT_OPT,
    gpu_hourly_usd: Optional[float] = GPU_PRICE_OPT,
    kwh_usd: Optional[float] = ENERGY_PRICE_OPT,
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    ttft_percentile: int = TTFT_PCT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    kernel: Optional[str] = KERNEL_OPT,
    hub: str = HUB_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
    confirm: str = CONFIRM_OPT,
    confirm_rounds: int = CONFIRM_ROUNDS_OPT,
    min_requests: Optional[int] = MIN_REQUESTS_OPT,
    strict_constraints: bool = STRICT_OPT,
    all_levels: bool = ALL_LEVELS_OPT,
    explore: str = EXPLORE_OPT,
    budget: Optional[str] = BUDGET_OPT,
    quality_probe: str = QUALITY_OPT,
    report_reuse: bool = REPORT_REUSE_OPT,
    kv_tier: str = KV_TIER_OPT,
    seed: Optional[str] = SEED_OPT,
    eval_set: Optional[str] = EVAL_SET_OPT,
    max_accuracy_loss: float = MAX_LOSS_OPT,
    eval_items: int = EVAL_ITEMS_OPT,
    eval_attempts: int = EVAL_ATTEMPTS_OPT,
    arrival_rate: Optional[str] = RATE_OPT,
    rate_check: str = RATE_CHECK_OPT,
) -> None:
    """Force a calibration rerun and overwrite the cached profile."""
    from rigtune.pipeline import resolve_profile

    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    objective, cons, ttft_percentile, max_accuracy_loss = _goal(
        objective, optimise, limit, ttft_ceiling, ttft_percentile, tok_s_floor, tpot_ceiling, wl, arrival_rate,
        strict_constraints, eval_set, max_accuracy_loss, gpu_hourly_usd, kwh_usd)
    _check_rate(objective, arrival_rate)
    _check_kv_tier(kv_tier, wl)
    _check_min_requests(min_requests, wl)
    view = _search_view(objective, cons)
    profile = _calibrating(view, resolve_profile, ModelSpec(hf_id=model), objective, force_backend=backend, recalibrate=True,
                              workload=wl, constraints=cons,
                              progress=view.progress, on_stage=view.on_stage,
                              power_mode=power, phases=phases, kv_connector=kv_connector,
                              options=_opts(quant, kv_quant, speculative, prefix_cache, combine, budget, ttft_percentile, confirm, all_levels, explore, quality_probe,
                 eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, eval_items=eval_items,
                 eval_attempts=eval_attempts,
                 report_reuse=report_reuse, kv_tier=kv_tier, seed=seed, arrival_rate=arrival_rate, rate_check=rate_check,
                 confirm_rounds=confirm_rounds, min_requests=min_requests, drafts=drafts, draft=draft, hub=hub,
                 kernel=kernel), layout=layout)
    view.stop()
    console.print(_trial_table(profile.calibration_table, winner=_winner_key(profile)))
    _print_profile(profile)
    _print_tradeoffs(profile, objective, cons)


@app.command()
def compare(
    model: str,
    objective: Optional[str] = typer.Option(None, "--objective", callback=_objective, help=OBJECTIVE_HELP),
    optimise: Optional[str] = OPTIMISE_OPT,
    limit: Optional[List[str]] = LIMIT_OPT,
    gpu_hourly_usd: Optional[float] = GPU_PRICE_OPT,
    kwh_usd: Optional[float] = ENERGY_PRICE_OPT,
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    eval_workload_file: Optional[Path] = typer.Option(
        None, "--eval-workload-file",
        help="Measure every row on these prompts instead (JSONL, as --workload-file). Calibration never sees them, "
             "so the comparison shows whether the pick holds on held-out traffic"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Restrict rigtune's candidates to one backend"),
    ollama_model: Optional[str] = typer.Option(None, "--ollama-model", help="Ollama tag for the ollama row"),
    include: Optional[List[str]] = typer.Option(None, "--include", help="Only these reference rows"),
    out: Optional[Path] = typer.Option(None, "--out", help="Results directory (default benchmarks/results)"),
    repeats: int = typer.Option(1, "--repeats", min=1,
                                help="Measure every row this many times, interleaved, and report the median and "
                                     "the spread; rows whose runs overlap rigtune's are flagged as within noise"),
    rows: Optional[List[Path]] = typer.Option(
        None, "--rows",
        help="Also measure the rows in this JSON file (repeatable): other tools' picks, such as vLLM "
             "auto_tune's from benchmarks/vllm_auto_tune.py, each recorded with its kind and source"),
    expert: bool = typer.Option(
        False, "--expert",
        help="Also measure the expert rows: stock rows with the settings the engines' own tuning guides "
             "recommend (rigtune/data/expert_rows.json, each quoted with its page); a row this card or "
             "engine cannot use is skipped with a note"),
    verdict: bool = typer.Option(
        False, "--verdict",
        help="Keep a record of every run of every row, append each to <name>.runs.jsonl as it is measured, "
             "score the result as a win, tie or loss against every row at the default thresholds, and write "
             "the verdict as JSON and Markdown beside it (`rigtune verdict` re-scores at others); never "
             "writes over an earlier run's files"),
    bare_stock: bool = typer.Option(
        False, "--bare-stock",
        help="Also measure vLLM and SGLang launched with the model and port alone, every other setting at "
             "the engine's own default (`<engine>-bare` rows), beside the `<engine>-default` rows, which "
             "pass what rigtune takes the defaults to be"),
    levels: Optional[str] = typer.Option(
        None, "--levels",
        help="Measure every row at these numbers of users (comma-separated, e.g. 64,128,256) instead of the "
             "workload's own, to see settings act that cannot at its levels; results are named "
             "<workload>-c<levels>"),
    check_answers: Optional[str] = typer.Option(
        None, "--check-answers",
        help="Grade every row's answers on this labelled set (gsm8k, or a JSONL file as --eval-set takes), "
             "all of it unless --check-items says fewer, once per row while its engine is up, against its "
             "engine's stock row, which answers twice to show its own noise; a row that loses more than "
             "--max-accuracy-loss points, or was not checked, is not applicable in the verdict, and if "
             "rigtune's own row is either, no row counts"),
    check_items: int = typer.Option(
        0, "--check-items", min=0,
        help="With --check-answers: grade only the first N items (0, the default, grades them all: 1319 for "
             "gsm8k)"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    ttft_percentile: int = TTFT_PCT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    kernel: Optional[str] = KERNEL_OPT,
    hub: str = HUB_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
    confirm: str = CONFIRM_OPT,
    confirm_rounds: int = CONFIRM_ROUNDS_OPT,
    min_requests: Optional[int] = MIN_REQUESTS_OPT,
    strict_constraints: bool = STRICT_OPT,
    all_levels: bool = ALL_LEVELS_OPT,
    explore: str = EXPLORE_OPT,
    budget: Optional[str] = BUDGET_OPT,
    quality_probe: str = QUALITY_OPT,
    report_reuse: bool = REPORT_REUSE_OPT,
    kv_tier: str = KV_TIER_OPT,
    seed: Optional[str] = SEED_OPT,
    eval_set: Optional[str] = EVAL_SET_OPT,
    max_accuracy_loss: float = MAX_LOSS_OPT,
    eval_items: int = EVAL_ITEMS_OPT,
    eval_attempts: int = EVAL_ATTEMPTS_OPT,
    arrival_rate: Optional[str] = RATE_OPT,
    rate_check: str = RATE_CHECK_OPT,
) -> None:
    """Measure rigtune's pick vs stock defaults (and Ollama) on one workload; write a results JSON."""
    from rigtune.bench import compare as _compare, to_markdown
    from rigtune.bench.compare import save
    from rigtune.bench.rows import RowsFileError, expert_rows, read_rows_file
    from rigtune.hardware import probe as _probe
    from rigtune.pipeline import prepare_and_plan, resolve_profile, select

    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    objective, cons, ttft_percentile, max_accuracy_loss = _goal(
        objective, optimise, limit, ttft_ceiling, ttft_percentile, tok_s_floor, tpot_ceiling, wl, arrival_rate,
        strict_constraints, eval_set, max_accuracy_loss, gpu_hourly_usd, kwh_usd)
    _check_rate(objective, arrival_rate)
    if verdict and cons.goal is not None:  # a verdict counts tok/s: it cannot judge a goal that optimises another
        from rigtune.calibrate.objectives import parse_goal

        metric = parse_goal(cons.goal).optimise
        if metric != "tok_s":
            raise typer.BadParameter(f"a verdict weighs tok/s, and this goal optimises {metric}: leave out "
                                     "--verdict, or optimise tok_s", param_hint="--verdict")
    _check_kv_tier(kv_tier, wl)
    _check_min_requests(min_requests, wl)
    # Held-out prompts are read before calibrating, so a bad file fails in seconds rather than after the search.
    eval_wl = _workload(workload, eval_workload_file, sessions, session_pool, session_turns, history, session_system) if eval_workload_file else wl
    # So are added rows. The expert rows' labels (`*-expert-*`) are reserved whether or not --expert is given,
    # so a label means the same row in every results file.
    extra: dict = {}
    for path in rows or []:
        try:
            extra.update(read_rows_file(path, taken=extra))
        except RowsFileError as exc:
            raise typer.BadParameter(str(exc), param_hint="--rows")
    grader = None
    if check_answers:  # read now, so a bad set fails in seconds rather than after the search
        from rigtune.bench.answers import RowGrader
        from rigtune.evalset import EvalSetError, load_eval_set

        try:
            graded = load_eval_set(check_answers, check_items)
        except EvalSetError as exc:
            raise typer.BadParameter(str(exc), param_hint="--check-answers")
        grader = RowGrader(items=graded.items, set_name=graded.name, sha256=graded.sha256)
    users = None
    if levels:
        try:
            users = [int(x) for x in levels.split(",") if x.strip()]
            if not users or min(users) < 1:
                raise ValueError
        except ValueError:
            raise typer.BadParameter(f"{levels!r}: numbers of users of at least 1, comma-separated",
                                     param_hint="--levels")
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    view = _search_view(objective, cons)
    profile = _calibrating(view, resolve_profile, spec, objective, force_backend=backend, workload=wl, constraints=cons,
                              progress=view.progress, hw=hw, on_stage=view.on_stage,
                              power_mode=power, phases=phases, kv_connector=kv_connector,
                              options=_opts(quant, kv_quant, speculative, prefix_cache, combine, budget, ttft_percentile, confirm, all_levels, explore, quality_probe,
                 eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, eval_items=eval_items,
                 eval_attempts=eval_attempts,
                 report_reuse=report_reuse, kv_tier=kv_tier, seed=seed, arrival_rate=arrival_rate, rate_check=rate_check,
                 confirm_rounds=confirm_rounds, min_requests=min_requests, drafts=drafts, draft=draft, hub=hub,
                 kernel=kernel), layout=layout)
    view.stop()
    _print_profile(profile)
    # Compare the checked target commit, including stock rows prepared on another backend.
    if profile.prepared is not None and profile.prepared.spec.revision:
        spec = profile.prepared.spec
    if grader is not None:  # a template that thinks is graded with thinking off, or every answer runs out of tokens
        from rigtune.accuracy_gate import grading_template_kwargs

        grader.extra_body, grader.graded_with = grading_template_kwargs(spec)
        if grader.graded_with:
            err.print(f"[dim]answer check: {grader.graded_with}[/]")
    candidates, reg = select(hw, spec, force=backend, workload=eval_wl, report_reuse=report_reuse)
    planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl,
                               quants=_opts(quant, kv_quant, speculative, prefix_cache, combine, budget, ttft_percentile, confirm, all_levels, explore, quality_probe,
                 eval_set=eval_set, max_accuracy_loss=max_accuracy_loss, eval_items=eval_items,
                 eval_attempts=eval_attempts,
                 report_reuse=report_reuse, arrival_rate=arrival_rate, rate_check=rate_check).search_quants())

    def _row_progress(label: str, row) -> None:
        if row is None:
            err.print(f"[cyan]compare[/] {label} ...")
        elif row.ok:
            err.print(f"[green]compare[/] {label}: {row.scored_tok_s:.1f} tok/s @c{row.scored_concurrency}, "
                      f"TTFT {_fmt(row.scored_ttft_ms, 0)} ms, SLO {'met' if row.meets_slo else 'missed'}")
        else:
            err.print(f"[red]compare[/] {label} FAILED: {(row.error or '').splitlines()[0][:80]}")

    from rigtune import cache as profile_cache

    if expert:
        extra = {**expert_rows(hw, planned.prepared, reg), **extra}
    target = runs = origin = requests = probe = None
    if verdict:  # a name no earlier run used, and every run on disk beside it the moment it is measured
        from rigtune.bench.compare import (copy_inputs, free_results_path, gpu_state, name_only, provenance,
                                           results_path, runs_path_for, with_levels)
        from rigtune.bench.verdict import Thresholds
        from rigtune.hardware import hardware_hash

        named = ModelSpec(hf_id=name_only(spec.hf_id))  # a local model folder by its name, not its path
        measured = with_levels(eval_wl, users)[0].name if users else eval_wl.name
        target = free_results_path(results_path(hardware_hash(hw), named, measured,
                                                profile.objective, out, goal=cons.goal))
        runs = runs_path_for(target)
        requests, probe = target.with_name(target.name.removesuffix(".json") + ".requests"), gpu_state
        if grader is not None:  # every graded answer, so a failed check can be read item by item
            grader.answers_file = target.with_name(target.name.removesuffix(".json") + ".answers.jsonl")
        prompts = {"calibration_prompts": workload_file, "measured_prompts": eval_workload_file}
        options = {"objective": objective, "workload": wl.name, "measured_on": measured, "levels": users,
                   "repeats": repeats, "include": list(include or []), "backend": backend,
                   "ttft_ceiling_ms": cons.ttft_ceiling_ms, "ttft_percentile": ttft_percentile,
                   "tpot_ceiling_ms": cons.tpot_ceiling_ms, "expert": expert, "bare_stock": bare_stock,
                   "rows": [p.name for p in rows or []],
                   "check_answers": name_only(check_answers) if check_answers else None,
                   "check_items": check_items if check_answers else None,
                   "max_accuracy_loss_points": max_accuracy_loss if check_answers else None,
                   "verdict_thresholds": vars(Thresholds()), **({"goal": cons.goal} if cons.goal else {})}
        origin = provenance(hw, prompts, model=spec, workload=eval_wl, options=options, command=sys.argv,
                            rows=list(rows or []), expert=expert, inputs=copy_inputs(target, prompts),
                            logs=f"{named.safe_id if named.hf_id == spec.hf_id else '<the model>'}"
                                 f"/compare-{eval_wl.name}")
        err.print(f"[dim]each run is appended to {runs} as it is measured[/]")
    result = _compare(hw, spec, profile, planned.prepared, reg, workload=eval_wl, constraints=cons,
                      ollama_tag=ollama_model, include=include or None, progress=_row_progress,
                      log_dir=profile_cache.logs_dir() / spec.safe_id / f"compare-{eval_wl.name}",
                      power=_power_controller(profile), repeats=repeats, extra=extra or None,
                      record_runs=verdict, runs_path=runs, answers=verdict, grader=grader,
                      answer_tolerance=max_accuracy_loss / 100, origin=origin, gpu_probe=probe,
                      requests_dir=requests, bare_stock=bare_stock, rotate=verdict, levels=users)
    path = save(result, out, path=target)
    console.print(to_markdown(result))
    console.print(f"[dim]saved {path}[/]")
    if verdict:
        _write_verdict(path)


def _write_verdict(path: Path, thresholds=None, out: Optional[Path] = None) -> None:
    from rigtune.bench import verdict as vd

    try:
        v, js, md = vd.write(path, thresholds, out)
    except ValueError as exc:  # a runs file with nothing to score, or not a comparison at all
        err.print(f"[red]rigtune verdict: {exc}[/]")
        raise typer.Exit(2)
    console.print(vd.to_markdown(v), markup=False)
    console.print(f"[dim]wrote {js} and {md}[/]")


@app.command()
def verdict(
    results: Path = typer.Argument(..., exists=True, dir_okay=False,
                                   help="A results JSON written by `rigtune compare`, or the "
                                        "<name>.runs.jsonl beside it (`compare --verdict`), which holds "
                                        "every finished run even when the comparison stopped early"),
    stock_gain: float = typer.Option(20.0, "--stock-gain",
                                     help="Percent rigtune's median must be above a stock row's to win"),
    tuner_gain: float = typer.Option(10.0, "--tuner-gain",
                                     help="Percent above a tuner row's (another tool's pick) to win"),
    expert_gain: float = typer.Option(10.0, "--expert-gain",
                                      help="Percent above an expert row's (a guide's settings) to win"),
    loss_margin: float = typer.Option(0.0, "--loss-margin",
                                      help="Percent a row must be ahead, ranges apart, to count as a loss"),
    out: Optional[Path] = typer.Option(None, "--out", help="Directory for the verdict (default: beside it)"),
) -> None:
    """Score a comparison as a win, tie or loss against every row. Only runs that met the limits count; a win
    needs the threshold at the median and in the worst paired round, every rigtune run above every run of the
    row, 3 runs on each side, and rigtune's answers checked and unchanged (else it is ahead on speed, answers
    unchecked); a row that could not start is not applicable, never a win."""
    from rigtune.bench.verdict import Thresholds

    thresholds = Thresholds(stock=stock_gain / 100, tuner=tuner_gain / 100, expert=expert_gain / 100,
                            loss_margin=loss_margin / 100)
    _write_verdict(results, thresholds, out)


@app.command()
def report(
    results: Optional[Path] = typer.Option(None, "--results", help="Results dir (default benchmarks/results)"),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write RESULTS.md and the SVG"),
) -> None:
    """Aggregate compare results: median gain over the best SLO-meeting default, RESULTS.md, plot."""
    from rigtune.bench.report import write_report

    md, svg, summary = write_report(results, out)
    console.print(summary.headline())
    console.print(f"[dim]wrote {md} and {svg}[/]")


@app.command()
def predict(
    model: str,
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    top: int = typer.Option(20, "--top", help="Rows to show"),
) -> None:
    """Predict tok/s and TTFT for every feasible config without launching anything."""
    from rigtune.hardware import hardware_hash, probe as _probe
    from rigtune.pipeline import prepare_and_plan, select
    from rigtune.predict import Predictor

    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    candidates, reg = select(hw, spec, force=backend)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    planned = prepare_and_plan(hw, spec, candidates, reg, workload=wl)
    pred = Predictor(hw)
    rows = []
    for name, cfgs in planned.feasible.items():
        for cfg, _ in cfgs:
            p = pred.best_level(planned.prepared[name], cfg, wl.prefill_tokens, wl.decode_tokens, wl.concurrencies,
                                wl.ttft_ceiling_ms)
            rows.append((cfg, p))
    rows.sort(key=lambda r: -r[1].tok_s)
    d = pred.dev
    console.print(f"[bold]{d.name}[/]: {d.mem_bw_gbs:.0f} GB/s, {d.tflops:.0f} TFLOPS"
                  f"{'' if d.known else ' (estimated)'}   workload {wl.name}   hardware {hardware_hash(hw)}")
    t = Table(title=f"Predicted ({len(rows)} feasible configs, top {top})")
    for col in ("config", "pred tok/s", "@c", "pred TTFT ms", "pred TPOT ms", "bound", "params"):
        t.add_column(col, justify="left" if col in ("config", "bound", "params") else "right", overflow="fold",
                     min_width=(40 if col == "config" else None))
    for cfg, p in rows[:top]:
        t.add_row(cfg.key(), f"{p.tok_s:.0f}", str(p.concurrency), f"{p.ttft_ms:.0f}", f"{p.tpot_ms:.1f}",
                  "memory" if p.memory_bound else "compute", "fitted" if p.fitted else "prior")
    console.print(t)
    if not any(p.fitted for _, p in rows):
        err.print("[dim]no fitted parameters for this machine yet; run `rigtune fit` after a calibration[/]")


@app.command()
def fit(
    apply: bool = typer.Option(False, "--apply", help="Write fitted parameters to ~/.rigtune/perf-model.json"),
    all_machines: bool = typer.Option(False, "--all", help="Report every machine's profiles (no --apply)"),
) -> None:
    """Fit the performance predictor from cached calibration profiles; report leave-one-out accuracy."""
    from rigtune import cache as profile_cache
    from rigtune import predict as P
    from rigtune.hardware import hardware_hash, probe as _probe

    if apply and all_machines:
        err.print("[red]--apply needs this machine's profiles only; drop --all[/]")
        raise typer.Exit(2)
    hh = hardware_hash(_probe())
    # One fit per machine, against the GPU its trials ran on: a roofline fitted to another card's
    # bandwidth and compute says nothing about either card.
    groups: dict = {}
    for _, p in profile_cache.list_profiles():
        if (all_machines or p.hardware_hash == hh) and p.hardware is not None:
            groups.setdefault(p.hardware_hash, (p.hardware, []))[1].extend(P.observations_from_profile(p))
    if not groups:
        console.print(f"no calibration profiles for hardware {hh} under {profile_cache.profiles_dir()}")
        return
    for machine, (machine_hw, obs) in sorted(groups.items()):
        dev = P.device_spec(machine_hw)
        evals = P.fit_all(obs, dev)
        if len(groups) > 1:
            console.print(f"[bold]hardware {machine}[/]")
        console.print(P.render_markdown(evals, dev))
        if apply and machine == hh:
            path = P.save_params(hh, {b: e.params for b, e in evals.items() if not e.keeps_prior})
            console.print(f"[green]wrote {path}[/]; calibration on hardware {hh} now prunes with these parameters")


@app.command("memory-report")
def memory_report(
    results: Optional[Path] = typer.Option(None, "--results", help="compare results dir (default benchmarks/results)"),
    apply: bool = typer.Option(False, "--apply", help="Write fitted workspace/margin to ~/.rigtune/memory-model.json"),
    all_machines: bool = typer.Option(False, "--all", help="Include profiles/results from other hardware hashes"),
) -> None:
    """Planner prediction vs measured peak memory across cached profiles and compare results."""
    from rigtune import cache as profile_cache
    from rigtune import memcal
    from rigtune.bench.report import load_results
    from rigtune.hardware import hardware_hash, probe as _probe
    from rigtune.models import TrialResult

    hw = _probe()
    hh = hardware_hash(hw)
    obs = []
    for _, p in profile_cache.list_profiles():
        if all_machines or p.hardware_hash == hh:
            obs += memcal.observations_from_trials(p.calibration_table, p.hardware_hash)
    for r in load_results(results):
        if all_machines or r.hardware_hash == hh:
            trials = [TrialResult(config=row.config, stage=row.label, metrics=row.metrics, launched=row.ok or row.error is None,
                                  error=row.error, memory=row.memory) for row in r.rows if row.memory is not None]
            obs += memcal.observations_from_trials(trials, r.hardware_hash)
    cals = memcal.analyse(obs)
    console.print(memcal.render_markdown(cals, hardware=None if all_machines else (hw.gpu.name if hw.gpu else hw.cpu.model_name)))
    if apply:
        if all_machines:
            err.print("[red]--apply needs this machine's observations only; drop --all[/]")
            raise typer.Exit(2)
        path = memcal.save_overrides(hh, cals)
        console.print(f"[green]wrote {path}[/]; the planner now uses these constants for hardware {hh}")


@app.command()
def profiles(
    trace: Optional[str] = typer.Option(None, "--trace", help="Draw the search behind one profile: part of a "
                                                             "cached profile's model id, or a path to a profile JSON"),
) -> None:
    """List cached profiles, or draw the search behind one."""
    from rigtune import cache as profile_cache

    if trace is not None:
        _trace(trace)
        return
    rows = profile_cache.list_profiles()
    if not rows:
        console.print(f"no profiles under {profile_cache.profiles_dir()}")
        return
    t = Table(title=str(profile_cache.profiles_dir()))
    for col in ("hardware", "model", "objective", "workload", "backend", "config", "trials", "created"):
        t.add_column(col)
    import datetime as dt

    for path, p in rows:
        t.add_row(p.hardware_hash, p.model_id, p.objective, p.workload, f"{p.backend} {p.backend_version or ''}",
                  p.config.key(), str(len(p.calibration_table)),
                  dt.datetime.fromtimestamp(p.created_at).strftime("%Y-%m-%d %H:%M"))
    console.print(t)


@app.command()
def tradeoffs(
    which: str = typer.Argument(..., help="A profile JSON, or part of a cached profile's model id"),
    objective: Optional[str] = typer.Option(None, "--objective", callback=_objective,
                                            help="Re-score under this preset instead of the profile's own goal"),
    optimise: Optional[str] = OPTIMISE_OPT,
    limit: Optional[List[str]] = LIMIT_OPT,
    gpu_hourly_usd: Optional[float] = GPU_PRICE_OPT,
    kwh_usd: Optional[float] = ENERGY_PRICE_OPT,
) -> None:
    """Show what a profile's limits cost, from its recorded trials, under its own goal or another one stated
    here (prices alone add a $/Mtok column). Launches nothing and changes no profile."""
    from rigtune.goals import GoalError, rescore_goal
    from rigtune.tradeoff import profile_constraints, to_lines, tradeoffs as trade_offs

    p = _find_profile(which)
    if not p.calibration_table:
        err.print("[yellow]that profile has no calibration table: it was served without a search[/]")
        raise typer.Exit(2)
    own = profile_constraints(p)
    try:
        goal, cons, prices, note = rescore_goal(p.objective, own, objective, optimise, limit or [],
                                                gpu_hourly_usd, kwh_usd)
    except GoalError as exc:
        raise typer.BadParameter(str(exc), param_hint="--optimise/--limit") from None
    gate = [n for n in p.notes if n.startswith("accuracy gate: refusing")]
    same = (goal, cons) == (p.objective, own)
    t = trade_offs(p.calibration_table, goal, cons, accuracy=gate, prices=prices,
                   winner=_winner_key(p) if same else None)
    if note:
        t.notes.insert(0, note)
    console.print("\n".join(to_lines(t)), markup=False, highlight=False, soft_wrap=True)


def _is_loopback(host: str) -> bool:
    """True for an address only this machine can reach."""
    import ipaddress

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname, which may well resolve to an address other machines reach


@app.command()
def serve(
    model: str,
    objective: Optional[str] = typer.Option(None, "--objective", callback=_objective, help=OBJECTIVE_HELP),
    optimise: Optional[str] = OPTIMISE_OPT,
    limit: Optional[List[str]] = LIMIT_OPT,
    gpu_hourly_usd: Optional[float] = GPU_PRICE_OPT,
    kwh_usd: Optional[float] = ENERGY_PRICE_OPT,
    workload: str = WORKLOAD_OPT,
    workload_file: Optional[Path] = WORKLOAD_FILE_OPT,
    sessions: bool = SESSIONS_OPT,
    session_pool: Optional[int] = SESSION_POOL_OPT,
    session_turns: Optional[int] = SESSION_TURNS_OPT,
    history: Optional[str] = HISTORY_OPT,
    session_system: Optional[Path] = SESSION_SYSTEM_OPT,
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("127.0.0.1", "--host", envvar=["RIGTUNE_HOST", "POLYSERVE_HOST"],
                             help="Address to listen on. The default is this machine only; 0.0.0.0 exposes the "
                                  "server, which has no authentication, to every network it can reach"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Force a backend by name"),
    skip_calibration: bool = typer.Option(False, "--skip-calibration", help="Serve with backend defaults"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    ttft_percentile: int = TTFT_PCT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None, help="latency/efficiency: absolute tok/s floor"),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    drafts: str = DRAFTS_OPT,
    draft: Optional[str] = DRAFT_OPT,
    kernel: Optional[str] = KERNEL_OPT,
    hub: str = HUB_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
    confirm: str = CONFIRM_OPT,
    confirm_rounds: int = CONFIRM_ROUNDS_OPT,
    min_requests: Optional[int] = MIN_REQUESTS_OPT,
    strict_constraints: bool = STRICT_OPT,
    all_levels: bool = ALL_LEVELS_OPT,
    explore: str = EXPLORE_OPT,
    budget: Optional[str] = BUDGET_OPT,
    quality_probe: str = QUALITY_OPT,
    report_reuse: bool = REPORT_REUSE_OPT,
    kv_tier: str = KV_TIER_OPT,
    seed: Optional[str] = SEED_OPT,
    eval_set: Optional[str] = EVAL_SET_OPT,
    max_accuracy_loss: float = MAX_LOSS_OPT,
    eval_items: int = EVAL_ITEMS_OPT,
    eval_attempts: int = EVAL_ATTEMPTS_OPT,
    arrival_rate: Optional[str] = RATE_OPT,
    rate_check: str = RATE_CHECK_OPT,
    max_in_flight: Optional[str] = typer.Option(
        None, "--max-in-flight", callback=_in_flight,
        help="Serve at most N inference requests at once, or `measured`: the busiest level the pick met its "
             "constraints at, capped by the engines' batch. Off by default"),
    queue: Optional[int] = typer.Option(None, "--queue", min=0,
                                        help="With --max-in-flight: requests that may wait for a slot before "
                                             "429 (default twice the limit)"),
    queue_timeout: float = typer.Option(30.0, "--queue-timeout",
                                        help="With --max-in-flight: seconds a queued request waits before 429"),
    idle_timeout: Optional[float] = typer.Option(None, "--idle-timeout",
                                                 help="Close a stream that sends nothing for this many seconds, "
                                                      "with an SSE error event. Off by default"),
    max_body_mb: Optional[float] = typer.Option(None, "--max-body-mb",
                                                help="Refuse request bodies over this many MiB with 413. Off by "
                                                     "default"),
    body_timeout: Optional[float] = typer.Option(None, "--body-timeout",
                                                 help="Refuse with 408 a request body that takes longer than this "
                                                      "many seconds to arrive. Off by default"),
    api_key_env: str = typer.Option("RIGTUNE_API_KEY", "--api-key-env",
                                    help="Environment variable holding a bearer API key; when it is set, every "
                                         "route but /health requires it"),
    require_api_key: bool = typer.Option(False, "--require-api-key",
                                         help="Refuse to start unless the API key variable is set"),
    profile_file: Optional[Path] = typer.Option(
        None, "--profile", exists=True, dir_okay=False,
        help="Serve this profile's pick, calibrated on this or another machine with the same GPU, instead of the "
             "cached one; nothing is calibrated, and the calibration options are not used"),
) -> None:
    """Discover hardware, calibrate once (cached), then serve an OpenAI-compatible API."""
    import dataclasses
    import os

    import uvicorn

    from rigtune import cache as profile_cache
    from rigtune.backends import get_backend
    from rigtune.pipeline import resolve_profile
    from rigtune.serve import Supervisor, create_app
    from rigtune.serve.limits import ServingLimits, ServingMiddleware, measured_in_flight

    api_key = os.environ.get(api_key_env) or None
    if require_api_key and not api_key:
        raise typer.BadParameter(f"environment variable {api_key_env} is empty or unset", param_hint="--api-key-env")
    if max_body_mb is not None and not (math.isfinite(max_body_mb) and max_body_mb > 0):
        raise typer.BadParameter(f"{max_body_mb} is not a size above 0 MiB; leave it out for no cap",
                                 param_hint="--max-body-mb")
    try:  # checked before an hour of calibration, not after
        limits = ServingLimits(
            max_in_flight=1 if max_in_flight == "measured" else (int(max_in_flight) if max_in_flight else None),
            queue=queue, queue_timeout=queue_timeout, idle_timeout=idle_timeout,
            max_body_bytes=int(max_body_mb * 2**20) if max_body_mb is not None else None,
            body_timeout=body_timeout, api_key=api_key)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    wl = _workload(workload, workload_file, sessions, session_pool, session_turns, history, session_system)
    objective, cons, ttft_percentile, max_accuracy_loss = _goal(
        objective, optimise, limit, ttft_ceiling, ttft_percentile, tok_s_floor, tpot_ceiling, wl, arrival_rate,
        strict_constraints, eval_set, max_accuracy_loss, gpu_hourly_usd, kwh_usd)
    _check_rate(objective, arrival_rate)
    _check_kv_tier(kv_tier, wl)
    _check_min_requests(min_requests, wl)
    spec = ModelSpec(hf_id=model)
    if profile_file is not None:
        profile = _portable_profile(profile_file, model)
    else:
        view = _search_view(objective, cons)
        profile = _calibrating(view, resolve_profile, spec, objective, force_backend=backend,
                               skip_calibration=skip_calibration,
                               workload=wl, constraints=cons,
                               progress=view.progress, on_stage=view.on_stage,
                               power_mode=power, phases=phases, kv_connector=kv_connector,
                               options=_opts(quant, kv_quant, speculative, prefix_cache, combine, budget,
                                             ttft_percentile, confirm, all_levels, explore, quality_probe,
                                             eval_set=eval_set, max_accuracy_loss=max_accuracy_loss,
                                             eval_items=eval_items, eval_attempts=eval_attempts,
                                             report_reuse=report_reuse, kv_tier=kv_tier, seed=seed, arrival_rate=arrival_rate,
                                             rate_check=rate_check, confirm_rounds=confirm_rounds,
                                             min_requests=min_requests, drafts=drafts, draft=draft, hub=hub,
                                             kernel=kernel), layout=layout)
        view.stop()
    _print_profile(profile)
    if profile.prepared is None:
        err.print("[red]profile has no prepared model; run `rigtune recalibrate`[/]")
        raise typer.Exit(2)
    # A session profile was calibrated, and its launch_args recorded, with the engine reporting cached tokens.
    be = get_backend(profile.backend, report_reuse=bool(profile.workload_spec.get("sessions"))
                     or profile.options.get("reuse") == "on")
    be.materialize(profile.prepared, [profile.config.quant])  # no-op if already on disk
    if profile.disagg is not None:
        from rigtune.disagg import DisaggSupervisor, create_pd_app

        sup = DisaggSupervisor(be, profile.disagg, profile.prepared, log_dir=profile_cache.logs_dir() / spec.safe_id,
                               power=_power_controller(profile))
        err.print(f"[dim]starting prefill engine on GPU {profile.disagg.prefill_gpu} and decode engine on GPU "
                  f"{profile.disagg.decode_gpu} ...[/]")
        sup.start()
        app_ = create_pd_app(sup.prefill_url, sup.decode_url, profile=profile, status_fn=sup.status)
        upstream = f"prefill {sup.prefill_url}, decode {sup.decode_url}"
    elif profile.replicas > 1:
        from rigtune.layout import ReplicaSupervisor, create_lb_app

        gpus = [g.index for g in profile.hardware.gpus if g.vendor == "nvidia"][: profile.replicas]
        sup = ReplicaSupervisor(be, profile.config, profile.prepared, gpus, log_dir=profile_cache.logs_dir() / spec.safe_id)
        err.print(f"[dim]starting {len(gpus)} {profile.backend} replicas on GPUs {gpus} ...[/]")
        sup.start()
        app_ = create_lb_app(sup.urls, profile=profile, status_fn=sup.status)
        upstream = ", ".join(sup.urls)
    else:
        sup = Supervisor(be, profile.config, profile.prepared,
                         log_path=profile_cache.logs_dir() / spec.safe_id / "serve.log",
                         power=_power_controller(profile))
        err.print(f"[dim]starting {profile.backend} ...[/]")
        sup.start()
        from rigtune.drift import TrafficWatch

        app_ = create_app(sup.base_url, profile=profile, status_fn=sup.status,
                          watch=TrafficWatch(profile.workload_spec))
        upstream = sup.base_url

    if max_in_flight == "measured":
        limits = dataclasses.replace(limits, max_in_flight=measured_in_flight(profile))
    if limits.active:  # none asked for: the app is served exactly as it was before these limits existed
        app_.add_middleware(ServingMiddleware, limits=limits)
        err.print("[dim]serving limits: " + "; ".join(limits.describe()) + "[/]")

    def _shutdown(*_: object) -> None:
        sup.stop()

    signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(0)))
    console.print(f"[bold green]rigtune listening on http://{host}:{port}/v1[/]  "
                  f"(backend {profile.backend} on {upstream})")
    if not _is_loopback(host) and not api_key:
        err.print(f"[yellow]{host} is reachable from other machines, and no API key is set: anyone "
                  f"who can reach port {port} can use the model and read /rigtune/profile. Set "
                  f"{api_key_env}, put an authenticating proxy in front of it, or pass --host 127.0.0.1.[/]")
    try:
        uvicorn.run(app_, host=host, port=port, log_level="warning")
    finally:
        _shutdown()


# --------------------------------------------------------------------------- power control

power_app = typer.Typer(help="GPU power cap and clock lock used by --power (needs root).", no_args_is_help=True)
app.add_typer(power_app, name="power")


@power_app.command("status")
def power_status() -> None:
    """Show what this GPU allows: power limit range, supported clocks, and whether control is permitted."""
    from rigtune import power as P
    from rigtune.hardware import probe as _probe

    hw = _probe()
    if hw.gpu is None:
        err.print("[red]no visible GPU; power tuning needs an NVIDIA GPU[/]")
        raise typer.Exit(1)
    try:
        caps = P.controller_for(hw.gpu.index).capabilities()
    except P.PowerControlUnavailable as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]GPU {caps.gpu_index}[/]: {hw.gpu.name}")
    console.print(f"power limit: default {caps.power_limit_default_w} W, current {caps.power_limit_current_w} W, "
                  f"allowed {caps.power_limit_min_w}-{caps.power_limit_max_w} W")
    if caps.sm_clocks_mhz:
        console.print(f"SM clocks: {caps.sm_clocks_mhz[-1]}-{caps.sm_clocks_mhz[0]} MHz "
                      f"({len(caps.sm_clocks_mhz)} steps), max {caps.sm_clock_max_mhz} MHz")
    for knob, ok in (("cap", caps.can_cap), ("clock", caps.can_lock)):
        why = "" if ok else f"  ({caps.reasons.get(knob, 'unknown')})"
        console.print(f"{'power capping' if knob == 'cap' else 'clock locking'}: "
                      f"{'[green]permitted[/]' if ok else '[red]not permitted[/]'}{why}")
    trying = [p.label() for p in P.candidate_points(caps, "both") if not p.is_default]
    console.print(f"--power both would try: {', '.join(trying) if trying else 'nothing'}")
    if P.pending_restore():
        err.print("[yellow]a previous run left a power change behind; run `rigtune power reset`[/]")


@power_app.command("reset")
def power_reset(gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU index (default: the recorded one)")) -> None:
    """Undo a power cap or clock lock left behind by a crashed run."""
    from rigtune import power as P

    try:
        state = P.reset_from_file(gpu)
    except P.PowerControlUnavailable as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    if state:
        console.print(f"restored GPU {state.get('gpu_index')} power limit to {int(state['power_limit_mw']) // 1000} W "
                      "and unlocked clocks")
    else:
        console.print("no recorded change; unlocked clocks anyway")


# --------------------------------------------------------------------------- bundles

bundle_app = typer.Typer(help="Carry a calibrated pick, and the caches that start it warm, to another machine with "
                              "the same GPU.", no_args_is_help=True)
app.add_typer(bundle_app, name="bundle")


@bundle_app.command("export")
def bundle_export(
    which: str = typer.Argument(..., help="A profile JSON, or part of a cached profile's model id"),
    out: Path = typer.Option(..., "--out", help="The bundle to write: .tar, or .tar.gz to compress it"),
) -> None:
    """Write a profile and rigtune's kernel cache (Triton's kernels, vLLM's compile caches, FlashInfer's kernels)
    to one file, with a manifest of what it is for and every file's sha256. Compiled caches can hold this machine's
    paths: a bundle is meant for machines you control."""
    from rigtune import bundle as bd

    profile = _find_profile(which)
    try:
        manifest = bd.export(profile, out)
    except (bd.BundleError, OSError) as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2)
    console.print(f"wrote {out}: {profile.model_id}, {profile.backend} {profile.backend_version}, "
                  f"{len(manifest['files'])} cache files ({manifest['cache_bytes'] / 2**20:.0f} MiB)")
    if profile.backend != "vllm":
        err.print(f"[yellow]{profile.backend} launches do not use the kernel cache: the bundle starts it no "
                  f"warmer[/]")


@bundle_app.command("import")
def bundle_import(
    bundle: Path = typer.Argument(..., exists=True, dir_okay=False, help="A bundle from `rigtune bundle export`"),
    profile_out: Optional[Path] = typer.Option(None, "--profile-out",
                                               help="Where to write its profile (default: beside the bundle, "
                                                    "<name>.profile.json)"),
) -> None:
    """Check a bundle against this machine, as `serve --profile` checks a profile, then add its caches to rigtune's
    kernel cache (never over a file already there) and write its profile out to serve."""
    from rigtune import bundle as bd
    from rigtune.hardware import probe

    try:
        res = bd.import_(bundle, probe())
    except (bd.BundleError, OSError) as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2)
    for w in res.warnings:
        err.print(f"[yellow]{w}[/]")
    if res.refusals:
        for r in res.refusals:
            err.print(f"[red]not imported: {r}[/]")
        raise typer.Exit(2)
    dest = profile_out or bundle.with_name(bundle.name.split(".")[0] + ".profile.json")
    dest.write_text(res.profile.model_dump_json(indent=1), encoding="utf-8")
    console.print(f"kernel cache: {res.written} files added, {res.kept} already there; profile written to {dest}")
    console.print(f"serve it with: rigtune serve {res.profile.model_id} --profile {dest}", markup=False)


# --------------------------------------------------------------------------- trial ledger

ledger_app = typer.Typer(help="The trial ledger: every calibration trial, kept under rigtune's home.",
                         no_args_is_help=True)
app.add_typer(ledger_app, name="ledger")


def _ledger_root() -> Path:
    from rigtune import ledger as lg

    base = lg.root()
    if base is None:
        err.print("[yellow]the trial ledger is off (RIGTUNE_LEDGER=off)[/]")
        raise typer.Exit(2)
    return base


@ledger_app.command("show")
def ledger_show(
    model: Optional[str] = typer.Option(None, "--model", help="Only calibrations of models whose id contains this"),
    calibration: Optional[str] = typer.Option(None, "--calibration",
                                              help="One calibration's trials: its id, or its first 6+ characters"),
) -> None:
    """List the calibrations in the ledger, or one calibration's trials. Paths are cut to their last part."""
    from rigtune import ledger as lg

    base = _ledger_root()
    if calibration is not None:
        try:
            cal = lg.find(calibration, base)
        except lg.LedgerError as exc:
            err.print(f"[red]{exc}[/]")
            raise typer.Exit(2)
        t = Table(title=f"calibration {cal.calibration_id}")
        for col in ("seq", "stage", "config", "outcome", "tok/s", "p95 TTFT ms", "score", "", "error"):
            t.add_column(col, overflow="fold")
        for row in lg.trial_rows(cal):
            t.add_row(*row, style="bold green" if row[7] else None)
        console.print(t, markup=False, highlight=False)
        if cal.record is None:
            console.print("no profile record: the calibration failed or was stopped before its pick")
        for problem in cal.scan.problems():
            err.print(f"[yellow]{problem}[/]")
        return
    rows, notes = lg.calibration_rows(base, model)
    n, size = lg.sizes(base)
    if not rows:
        console.print(f"no calibrations in the ledger ({n} files)")
    else:
        t = Table(title=f"trial ledger: {n} file{'s' if n != 1 else ''}, {size / 2**20:.1f} MiB")
        for col in ("calibration", "model", "began (UTC)", "objective", "trials", "failed", "skipped", "pick"):
            t.add_column(col, overflow="fold")
        for row in rows:
            t.add_row(*row)
        console.print(t, markup=False, highlight=False)
    for note in notes:
        err.print(f"[yellow]{note}[/]")


@ledger_app.command("verify")
def ledger_verify(
    which: str = typer.Argument(..., help="A profile JSON, or part of a cached profile's model id"),
) -> None:
    """Check a profile against the ledger: its records are intact, they rebuild this profile and its table trial
    for trial, and the pick re-derived from the table is its configuration. Exits 0 when every check passed, 1 when
    any differs, 2 when there is no calibration to check against, 3 when nothing differs but something could not be
    checked (a --layout or --phases profile's own choice, an accuracy-gated pick)."""
    from rigtune import ledger as lg

    v = lg.verify(_find_profile(which), _ledger_root())
    for status, text in v.findings:
        console.print(f"{status:<12} {lg._scrub(text)}", markup=False, highlight=False)
    if v.exit_code:
        raise typer.Exit(v.exit_code)


@ledger_app.command("rebuild")
def ledger_rebuild(
    calibration_id: str = typer.Argument(..., help="The calibration's id, or its first 6+ characters"),
    out: Path = typer.Option(..., "--out", help="Where to write the rebuilt profile JSON"),
) -> None:
    """Write the profile a calibration recorded, rebuilt from the ledger: the same file it saved, for a profile that
    was lost or replaced by a later calibration. The profile cache is left alone."""
    from rigtune import ledger as lg

    try:
        profile = lg.rebuild(lg.find(calibration_id, _ledger_root()))
    except lg.LedgerError as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"wrote {out}: {lg._model_name(profile.model_id)} {profile.objective}, "
                  f"{lg._scrub(profile.config.key())}, "
                  f"{len(profile.calibration_table)} trials", markup=False, highlight=False)


if __name__ == "__main__":
    app()
