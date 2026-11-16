#!/usr/bin/env python
"""Write the phase plan for the next GPU session, for benchmarks/gpu_session.py (docs/gpu-session.md).

    python benchmarks/next_session_plan.py --out plan.json            # on the pod, from the repository's root
    python benchmarks/gpu_session.py run --plan plan.json --root /workspace/session --dry-run

The session verifies what has been built but never measured (docs/benchmarks.md, "Not yet measured"), and
measures Qwen3.5-9B and Gemma 4 12B again against stock rows, vLLM's auto_tune, the expert rows and two other
tuners. Its phases, in order:

1. `env`, `hub-access`, `prompts`: the machine and every package version; that the Hub token reaches every
   model the session loads; the Dolly calibration and held-out prompt files. `env` fails, and every other phase
   is then blocked, unless the rigtune this interpreter imports is the source at the plan's commit (--commit, or
   the COMMIT file beside the source): the runner's interpreter is vLLM's, with the source installed in it.
2. `selfcheck`, `smoke`: the load generator against the fake engine, and every search feature launched once.
3. `engine-probe`, `conformance-vllm`, `conformance-sglang`: `rigtune probe --engines` and `rigtune conformance`
   (items 20 and 21); skipped, with the reason, on a rigtune whose help does not list the options they pass.
4. `suffix-facts`, `suffix-cal`: why suffix decoding failed on 14 November (item 19).
5. `g1-...` to `g4-...`: custom goals on a GPU (item 18, G1 to G4). G2 fails unless serve names the efficiency
   preset and starts from the cached profile without calibrating. G4's energy limit is the balanced pick's
   J/token, which `g3-balanced` writes to g4_jpt.txt; a value in <root>/G4_JPT is used instead when that file
   exists. No second runner is needed.
6. `power`: `--power` (item 5, and G5), only where RIGTUNE_SESSION_POWER=1 says the runner has root on a
   machine it controls; on a rented pod it is skipped with that reason.
7. Per model: `cal-<m>` (rigtune's calibration, `--speculative all`, which also tries Qwen3.5's MTP head:
   item 16), `ledger-<m>` (the calibration's profile passes `rigtune ledger verify`, and `ledger rebuild` writes
   the same bytes), `lengths-<m>`, then the rows: `autotune-<m>`, `llmopt-vllm-<m>`, `llmopt-sglang-<m>`,
   `aic-vllm-<m>`, `aic-sglang-<m>`, each writing a rows.json even when its tool fails or does not apply, and
   `cmp-<m>`: every row on the held-out prompts, 3 interleaved repeats, answers graded, scored.

The runner runs each command with `bash -o pipefail`, and a phase ends non-zero when the file it exists to write
is missing or empty: request lengths that could not be read, a tuner command that could not be written, an
auto_tune run without a row. Each tool gets the calibration's wall-clock budget; one that runs over is cut off,
and its row says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MODELS = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it"]
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")

# Leftover GPU processes from a phase that was killed would shrink what the next one can start (an orphaned
# VLLM::EngineCore keeps its memory, and pkill on a name misses it), so each GPU phase starts on an empty card,
# killing leftovers by the PIDs nvidia-smi lists. Phases run one at a time, so anything listed is a leftover.
CLEAN = ("for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do "
         "echo \"killing leftover GPU process $p\"; kill -KILL \"$p\" 2>/dev/null; done; sleep 2; ")

HUB_CHECK = ("import sys, huggingface_hub as h\nbad = 0\nfor m in sys.argv[1:]:\n    try:\n"
             "        h.hf_hub_download(m, \"config.json\"); print(\"ok\", m)\n    except Exception as e:\n"
             "        bad = 1; print(\"NO ACCESS\", m, type(e).__name__, str(e)[:200])\nsys.exit(bad)")


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")


def safe_id(model: str) -> str:
    return model.replace("/", "__")


def pick_jpt(profile: dict) -> Optional[float]:
    """The J/token a profile's pick measured: its configuration's last calibration trial that recorded energy.
    None where no trial of that configuration did (no power readings on this machine)."""
    for trial in reversed(profile.get("calibration_table") or []):
        if trial.get("config") == profile.get("config"):
            value = (trial.get("metrics") or {}).get("joules_per_token")
            if value:
                return float(value)
    return None


def g2_check(log: str, flat: str) -> str:
    """Shell that decides G2 from serve's output: the goal named as the efficiency preset, the profile read from
    the cache, and no calibration. The log is joined into one line first, because the console wraps it."""
    return (f"tr -s ' \\r\\n' ' ' < {log} > {flat}; "
            f"grep -i -e 'efficiency preset' {flat} > /dev/null || "
            "{ echo 'G2 refuted: serve did not name the goal as the efficiency preset'; exit 1; }; "
            f"if grep -e '-> calibrate' {flat} > /dev/null; then "
            "echo 'G2 refuted: serve calibrated instead of using the cached efficiency profile'; exit 1; fi; "
            f"grep -e '-> cache hit' {flat} > /dev/null || "
            "{ echo 'G2 undecided: serve did not report a cache hit; read serve.log'; exit 1; }; "
            "echo 'G2 confirmed: the efficiency preset, served from the cache without calibrating'")


def build(a: argparse.Namespace) -> dict:
    py = "{python}"
    bench = f"{a.repo}/benchmarks"
    rig = f"{py} -m rigtune.cli"
    prompts = "{root}/prompts"
    cal_file, held_file = f"{prompts}/dolly.jsonl", f"{prompts}/dolly-heldout.jsonl"
    limits = f"--ttft-ceiling {a.ttft_ms:g} --tpot-ceiling {a.tpot_ms:g}"
    # Calibration and comparison must pass the same search options, or the comparison would not find the
    # cached profile and would calibrate again (SearchOptions.key() is the cache key).
    cal_opts = (f"--workload chat --workload-file {cal_file} --objective balanced {limits} "
                f"--speculative all --drafts verified --budget {a.cal_budget_min}m")
    budget_s = a.cal_budget_min * 60
    phases: List[Dict] = []

    def phase(name: str, cmd: str, why: str, needs: Optional[List[str]] = None, timeout_s: Optional[int] = None,
              requires: Optional[str] = None) -> None:
        p: Dict = {"name": name, "cmd": cmd, "why": why}
        # Nothing runs on a pod whose rigtune is not the plan's: every phase needs env, directly or through
        # one that does.
        needs = ([] if name == "env" else ["env"]) + list(needs or [])
        if needs:
            p["needs"] = needs
        if timeout_s:
            p["timeout_s"] = timeout_s
        if requires:
            p["requires"] = requires
        phases.append(p)

    # The source arrives as `git archive` output, without .git: the laptop sends the commit beside it, and the
    # rigtune this interpreter imports must be that source (an editable install), at the plan's commit.
    installed = (f"{py} -c 'import pathlib, rigtune; "
                 "print((pathlib.Path(rigtune.__file__).resolve().parents[1] / \"COMMIT\").read_text().strip())'")
    phase("env", (
        f"{rig} probe --json > {{out}}/probe.json; {rig} probe > {{out}}/probe.txt; "
        "nvidia-smi -q > {out}/nvidia-smi.txt; nvidia-smi topo -m > {out}/topo.txt; "
        f"{py} -m pip freeze > {{out}}/pip-freeze.txt; "
        "if [ -n \"${SGLANG_PYTHON:-}\" ]; then \"$SGLANG_PYTHON\" -m pip freeze > {out}/sglang-pip-freeze.txt; fi; "
        "head -1 \"$(command -v vllm)\" > {out}/vllm-shebang.txt; df -h / /workspace > {out}/disk.txt; "
        f"{rig} power status > {{out}}/power-status.txt 2>&1; "
        f"WANT={a.commit or 'unknown'}; GOT=$({installed} 2> {{out}}/rigtune-installed.err); "
        "echo \"plan: $WANT installed: $GOT\" > {out}/rigtune-commit.txt; "
        "if [ \"$WANT\" = unknown ] || [ \"$GOT\" != \"$WANT\" ]; then cat {out}/rigtune-installed.err; "
        f"echo \"the rigtune that {py} imports is not the source at the plan's commit ($WANT; it reports "
        f"'$GOT'): pip install -e {a.repo} into this interpreter, and write the plan again with --commit\"; "
        "exit 1; fi"),
        "what this session ran on: GPU, driver, every package version, the vLLM entry point's Python, disk; "
        "fails unless the rigtune installed in the runner's Python is the source at the plan's commit")
    hub_models = " ".join(sorted({*a.models, a.smoke_model, a.suffix_model, a.goal_model}))
    phase("hub-access", (
        f"{py} -c '{HUB_CHECK}' {hub_models} > {{out}}/hub-access.txt 2>&1; rc=$?; "
        "cat {out}/hub-access.txt; exit $rc"),
        "the Hub token in the runner's environment reaches every model the session loads, gated ones included "
        "(each one's config.json is fetched); a model listed NO ACCESS will fail its phases")
    # {root}/prompts is this phase's own directory, so the files every later phase reads are in its copy too.
    phase("prompts", (
        f"{py} {bench}/make_dolly_prompts.py --out {cal_file} && "
        f"{py} {bench}/make_dolly_prompts.py --offset 300 --out {held_file} && "
        f"sha256sum {cal_file} {held_file} > {{out}}/prompts.sha256"),
        "the 300 Dolly calibration prompts and 300 held out, as on 14 November (file-dolly-d03d5896 and "
        "file-dolly-heldout-fc5b6407 when Dolly is unchanged)")
    phase("selfcheck", f"{rig} selfcheck --json > {{out}}/selfcheck.json; {rig} selfcheck",
          "the load generator against the fake engine on this pod's CPU (Limits: checked on the laptop only, "
          "where it fails at 32 users and up)", timeout_s=1800)
    phase("smoke", CLEAN + f"{py} {bench}/smoke_features.py --model {a.smoke_model} --backends vllm "
          f"--log-dir {{out}}/logs",
          "every search feature launched once on the real engine, to catch launch failures before the long "
          "phases", timeout_s=5400)
    # A requirement greps the command's help for the option its phase passes, so a rigtune without it is
    # skipped with that reason. Not `grep -q`: under pipefail it ends the pipe early and can fail a help
    # that matched.
    phase("engine-probe", (
        f"{rig} probe --engines --reprobe --json > {{out}}/probe-engines.json && "
        f"{rig} probe --engines > {{out}}/probe-engines.txt"),
        "item 20: each engine's own answer, from its own Python, for its flags, architectures and speculative "
        "methods (what rigtune kept under its home is in this phase's rigtune-home/)",
        requires=f"COLUMNS=200 {rig} probe --help | grep -e --engines", timeout_s=3600)
    for engine in ("vllm", "sglang"):
        phase(f"conformance-{engine}", CLEAN + f"{rig} conformance {engine} --out {{out}}/conformance",
              f"item 21: the conformance run for the installed {engine} on this card; done only when the release "
              "is certified, and the record and engine logs are in conformance/ either way",
              requires=f"COLUMNS=200 {rig} conformance --help | grep -e --out", timeout_s=3 * 3600)
        for model in a.models:
            method = "assist" if "gemma-4-" in model else "mtp" if "Qwen3.5-" in model else None
            if method:
                phase(f"drafts-{engine}-{slug(model)}", CLEAN + (
                    f"{rig} conformance {engine} --model {model} --drafts verified --only C06 "
                    "--out {out}/conformance; "
                    f"{py} {bench}/next_session_plan.py --check-drafts {{out}}/conformance --expected-method {method}"),
                    f"draft checks G1/G2/G7/G8: {method} must actually pass; a missing/skipped method fails; "
                    "inspect accepted-token counts, revision and pool sizing in the kept logs",
                    needs=["hub-access"], timeout_s=3 * 3600)
    phase("drafts-llama-sglang", CLEAN + (
        f"{rig} bench unsloth/Llama-3.1-8B-Instruct --backend sglang --speculative all --drafts verified "
        f"--workload chat --workload-file {cal_file} --budget {a.cal_budget_min}m --save"),
        "draft checks G3/G4/G6: inspect the ledger for a pinned STANDALONE launch and compare logged KV tokens "
        "with the plan (within 10%); benchmark completion alone does not establish a gain",
        needs=["prompts"], timeout_s=budget_s + 3600)
    phase("suffix-facts", (
        f"{py} -m pip show -f arctic-inference > {{out}}/arctic-files.txt 2>&1; "
        "grep -c '_C' {out}/arctic-files.txt > {out}/arctic-native-count.txt; "
        f"{py} -m pip cache list arctic_inference > {{out}}/pip-cache.txt 2>&1; "
        "head -1 \"$(command -v vllm)\" > {out}/vllm-shebang.txt; "
        f"{py} -c 'import arctic_inference.suffix_decoding._C' > {{out}}/import-C.txt 2>&1; "
        "echo \"import exit $?\" >> {out}/import-C.txt; true"),
        "item 19: whether the installed suffix tree loads in the Python that runs vLLM. Confirmed broken if "
        "import-C.txt shows No module named 'arctic_inference.suffix_decoding._C'")
    phase("suffix-cal", CLEAN + (
        f"{rig} recalibrate {a.suffix_model} --backend vllm --workload chat --workload-file {cal_file} "
        f"--budget {a.goal_budget_min}m"),
        "item 19: with the build from the documented install line suffix:24 is tried and starts; with a broken "
        "build the profile's notes say suffix decoding was not tried and no suffix trial launches",
        needs=["prompts"], timeout_s=(a.goal_budget_min + 30) * 60)
    g = a.goal_model
    gb = f"--budget {a.goal_budget_min}m"
    gt = (a.goal_budget_min + 30) * 60
    phase("g1-power-preflight", CLEAN + (
        f"{rig} bench {g} --workload chat --optimise joules_per_token --limit 'power_w<=250' {gb}"),
        "G1: starts calibrating where NVML gives power readings, and is refused with the reason before any "
        "download where it does not (read output.log for which)", timeout_s=gt)
    # The budget is part of a profile's cache key (SearchOptions.key()), so both halves of G2 pass these same
    # options: with another budget serve would look under another key and calibrate, a false refutation.
    g2_opts = f"--workload chat {gb}"
    phase("g2-efficiency-cal", CLEAN + f"{rig} recalibrate {g} {g2_opts} --objective efficiency",
          "G2, first half: the efficiency preset's profile, cached", timeout_s=gt)
    phase("g2-goal-is-preset", CLEAN + (
        f"COLUMNS=200 timeout 600 {rig} serve {g} {g2_opts} --optimise joules_per_token "
        "--limit 'tok_s>=50%best' > {out}/serve.log 2>&1; " + CLEAN
        + g2_check("{out}/serve.log", "{out}/serve.flat")),
        "G2: confirmed (done) when serve names the goal as the efficiency preset and starts from the cached "
        "profile; refuted (failed) when serve.log shows a calibration or no preset",
        needs=["g2-efficiency-cal"], timeout_s=1200)
    phase("g3-balanced", CLEAN + (
        f"{rig} recalibrate {g} --workload chat --objective balanced {gb} && "
        f"P=$(ls -t $RIGTUNE_HOME/profiles/*/{safe_id(g)}/balanced-chat*.json | head -1) && [ -n \"$P\" ] && "
        f"{{ {rig} tradeoffs \"$P\" > {{out}}/tradeoffs.txt 2>&1 "
        "|| echo 'rigtune tradeoffs failed: run it by hand on the profile'; } && "
        f"{{ {py} {bench}/next_session_plan.py --pick-jpt \"$P\" > {{out}}/g4_jpt.txt || "
        "{ rm -f {out}/g4_jpt.txt; echo 'the balanced pick recorded no J/token (no power readings here): G4 is "
        "skipped unless {root}/G4_JPT gives a value'; }; }"),
        "G3, first half: the balanced pick and its trade-off table ('without ttft_p95_ms<=...: X tok/s'); also "
        "the pick's J/token in g4_jpt.txt, which G4 takes as its energy limit",
        timeout_s=gt)
    phase("g3-throughput", CLEAN + f"{rig} recalibrate {g} --workload chat --objective throughput {gb}",
          "G3: confirmed if the throughput pick reaches the table's X within 2%; refuted if it picks less",
          needs=["g3-balanced"], timeout_s=gt)
    # G4's limit is read from files just before each G4 phase: a value the user wrote to <root>/G4_JPT, else the
    # one g3-balanced computed. The runner already running reads it, so no second runner is ever started.
    g4_files = "{root}/G4_JPT {root}/g3-balanced/g4_jpt.txt"
    g4_need = (f"for f in {g4_files}; do [ -s \"$f\" ] && exit 0; done; echo 'no J/token limit for G4: "
               "g3-balanced recorded none; write one to {root}/G4_JPT and run the session again'; exit 1")
    g4_read = (f"G4_JPT=$(for f in {g4_files}; do [ -s \"$f\" ] && {{ head -1 \"$f\" | tr -d ' \\r\\n'; break; }}; "
               "done) && [ -n \"$G4_JPT\" ] && echo \"G4_JPT=$G4_JPT\" > {out}/g4_jpt.env && ")
    for name, extra in (("g4-custom", ""), ("g4-custom-all-levels", " --all-levels")):
        phase(name, CLEAN + g4_read + (
            f"{rig} recalibrate {g} --workload chat --optimise tok_s --limit \"joules_per_token<=$G4_JPT\" "
            f"{gb}{extra}"),
            "G4: with and without --all-levels the same configuration at the same level; refuted if they differ",
            needs=["g3-balanced"], requires=g4_need, timeout_s=gt)
    power_need = ("[ \"${RIGTUNE_SESSION_POWER:-0}\" = 1 ] || { echo '--power needs root on a machine you "
                  "control (NVML power limits are machine-wide), which a rented pod is not; set "
                  "RIGTUNE_SESSION_POWER=1 only there'; exit 1; }")
    phase("power", CLEAN + (f"cd {a.repo} && OUT={{out}}/results RIGTUNE=\"{rig}\" "
                            f"bash benchmarks/measure_power.sh {a.smoke_model}"),
          "item 5 and G5: --power on real hardware; whether the energy-optimal point sits near 70% of full "
          "power, and differs between rag and generation (results in this phase's results/)",
          requires=power_need, timeout_s=4 * 3600)

    for model in a.models:
        m, sid = slug(model), safe_id(model)
        lengths = f"{{root}}/lengths-{m}/lengths.env"
        # Sourced by every tuner's phase; one that set no lengths must not go on with empty ones.
        src = f". {lengths} && [ -n \"${{INPUT_LEN:-}}\" ] && [ -n \"${{OUTPUT_LEN:-}}\" ]"
        cal_profile = (f"P=$(ls -t {{root}}/cal-{m}/rigtune-home/profiles/*/{sid}/*.json | head -1) && "
                       "[ -n \"$P\" ]")
        phase(f"cal-{m}", CLEAN + f"{rig} recalibrate {model} {cal_opts}",
              f"rigtune's pick for {model} on the Dolly calibration prompts; for Qwen3.5 also item 16, its own "
              "MTP head under --speculative all", needs=["prompts"], timeout_s=budget_s + 3600)
        phase(f"ledger-{m}", (
            f"{cal_profile} && ID=$({py} -c 'import json, sys; "
            "print(json.load(open(sys.argv[1], encoding=\"utf-8\"))[\"calibration_id\"])' \"$P\") && "
            f"{rig} ledger verify \"$P\" > {{out}}/verify.txt 2>&1; v=$?; cat {{out}}/verify.txt; "
            f"{rig} ledger rebuild \"$ID\" --out {{out}}/rebuilt.json && cmp \"$P\" {{out}}/rebuilt.json; r=$?; "
            "echo \"ledger verify: exit $v; rebuild and byte comparison: exit $r\" > {out}/result.txt; "
            "cat {out}/result.txt; [ $v = 0 ] && [ $r = 0 ]"),
            f"the trial ledger on a real calibration: {model}'s profile passes `rigtune ledger verify`, and "
            "`ledger rebuild` writes the same file byte for byte", needs=[f"cal-{m}"], timeout_s=1800)
        phase(f"lengths-{m}", (
            f"{cal_profile} && {py} {bench}/vllm_auto_tune.py lengths --profile \"$P\" > {{out}}/lengths.txt && "
            "cat {out}/lengths.txt && cut -d' ' -f1,2 {out}/lengths.txt | tr ' ' '\\n' > {out}/lengths.env && "
            "grep -e '^INPUT_LEN=[0-9]' {out}/lengths.env > /dev/null && "
            "grep -e '^OUTPUT_LEN=[0-9]' {out}/lengths.env > /dev/null || "
            "{ echo 'no request lengths could be read from the calibration profile'; exit 1; }"),
            "the calibration's mean prompt and answer tokens, the request lengths every other tuner is given",
            needs=[f"cal-{m}"])
        lim = f"--ttft-ms {a.ttft_ms:g} --tpot-ms {a.tpot_ms:g}"
        sizes = "--input-len $INPUT_LEN --output-len $OUTPUT_LEN"
        na_tool = f"{py} {bench}/competitor_rows.py not-applicable --out {{out}}/rows.json"
        phase(f"autotune-{m}", CLEAN + (
            f"{src} && timeout {budget_s} {py} {bench}/vllm_auto_tune.py run --vllm-src {a.vllm_src} "
            f"--model {model} --rule ttft95 --limit-ms {a.ttft_ms:g} {sizes} --max-model-len {a.max_model_len} "
            "--out {out}; rc=$?; "
            f"if [ ! -s {{out}}/rows.json ]; then {na_tool} --label autotune-vllm --source \"vLLM auto_tune.sh\" "
            f"--reason \"auto_tune exited $rc (124: cut off at the calibration's budget of {budget_s} s) and "
            "wrote no row; see output.log\"; [ $rc != 0 ] || rc=1; fi; exit $rc"),
            "vLLM's own auto_tune, judged on the same p95 TTFT limit (rows.json; its search uses random "
            "weights and open-loop load); a run that wrote no row fails, with a not-applicable row in its place",
            needs=[f"lengths-{m}"], timeout_s=budget_s + 1800)
        for fw in ("vllm", "sglang"):
            label = f"llm-optimizer-{fw}"
            server = ""
            if fw == "sglang":
                server = (f" --server-cmd \"$SGLANG_PYTHON -m sglang.launch_server --model-path {model} "
                          "--host 127.0.0.1 --port 30000\"")
            na = f"{na_tool} --label {label} --source \"BentoML llm-optimizer at ${{VER:-unknown}}\" --reason"
            phase(f"llmopt-{fw}-{m}", CLEAN + (
                f"{src} && VER=$(git -C {a.llm_optimizer_src} rev-parse --short HEAD 2>/dev/null "
                "|| echo unknown) && "
                f"CMD=$({py} {bench}/competitor_rows.py llm-optimizer-cmd --framework {fw} --model {model} "
                f"{lim} {sizes} --max-model-len {a.max_model_len} --levels {a.levels} "
                f"--binary {a.llm_optimizer_bin} --out {{out}}/llm-optimizer.json --shell{server}) && "
                f"[ -n \"$CMD\" ] || {{ {na} \"the llm-optimizer command could not be written (no request "
                "lengths, or competitor_rows.py failed); see output.log\"; exit 1; }; "
                "echo \"$CMD\" > {out}/command.txt; "
                f"timeout {budget_s} bash -c \"$CMD\"; rc=$?; " + CLEAN +
                f"if [ -f {{out}}/llm-optimizer.json ]; then {py} {bench}/competitor_rows.py llm-optimizer-row "
                f"--result {{out}}/llm-optimizer.json --framework {fw} --max-model-len {a.max_model_len} "
                "--tool-version \"$VER\" --out {out}/rows.json; "
                f"else {na} \"llm-optimizer exited $rc (124: cut off at the calibration's budget of "
                f"{budget_s} s) and wrote no --output-json; see output.log\"; fi; exit $rc"),
                f"BentoML llm-optimizer's pick on {fw}, same model, limits and lengths (rows.json, written even "
                "when it fails)", needs=[f"lengths-{m}"], timeout_s=budget_s + 1800)
        for be in ("vllm", "sglang"):
            label = f"aiconfigurator-{be}"
            na = f"{na_tool} --label {label} --source \"NVIDIA AIConfigurator ${{VER:-unknown}}\" --reason"
            phase(f"aic-{be}-{m}", (
                f"{src} && VER=$({a.aiconfigurator_python} -c \"import importlib.metadata as m; "
                "print(m.version('aiconfigurator'))\" 2>/dev/null || echo unknown) && "
                "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) || "
                f"{{ {na} \"no request lengths, or no GPU name from nvidia-smi; see output.log\"; exit 1; }}; "
                f"if CMD=$({py} {bench}/competitor_rows.py aiconfigurator-cmd --model {model} --gpu \"$GPU\" "
                f"--backend {be} {lim} {sizes} --save-dir {{out}}/aic --binary {a.aiconfigurator_bin} --shell "
                "2> {out}/not-applicable.txt) && [ -n \"$CMD\" ]; then echo \"$CMD\" > {out}/command.txt; "
                "timeout 1800 bash -c \"$CMD\"; rc=$?; "
                f"if [ $rc = 0 ]; then {py} {bench}/competitor_rows.py aiconfigurator-row --save-dir {{out}}/aic "
                f"--backend {be} --max-model-len {a.max_model_len} --tool-version \"$VER\" --out {{out}}/rows.json "
                f"|| {na} \"AIConfigurator wrote no launch script to read\"; "
                f"else {na} \"AIConfigurator exited $rc; see output.log\"; fi; exit $rc; "
                f"else {na} \"$(cat {{out}}/not-applicable.txt) (no AIConfigurator command for this card)\"; fi"),
                f"NVIDIA AIConfigurator's predicted pick on {be}, or a not-applicable row where it has no data "
                "for this card", needs=[f"lengths-{m}"], timeout_s=2400)
        row_files = " ".join(f"{{root}}/{p}-{m}/rows.json" for p in (
            "autotune", "llmopt-vllm", "llmopt-sglang", "aic-vllm", "aic-sglang"))
        phase(f"cmp-{m}", CLEAN + (
            f"ROWS=''; for f in {row_files}; do if [ -s \"$f\" ]; then ROWS=\"$ROWS --rows $f\"; else "
            "echo \"ROW MISSING: $f (its phase left no row; this comparison runs without it)\" "
            ">> {out}/rows-missing.txt; fi; done; [ ! -f {out}/rows-missing.txt ] || cat {out}/rows-missing.txt; "
            f"{rig} compare {model} {cal_opts} --eval-workload-file {held_file} --repeats {a.repeats} "
            f"--expert --bare-stock $ROWS --verdict --check-answers gsm8k --check-items {a.check_items} "
            "--out {out}/results"),
            f"{model}: rigtune's pick against stock (as rigtune takes it, and bare), the expert rows, auto_tune, "
            "llm-optimizer and AIConfigurator, on held-out prompts, scored win/tie/loss with answers graded; a "
            "row file that is missing is named in rows-missing.txt",
            needs=[f"cal-{m}"], timeout_s=6 * 3600)
    # Kernel settings and start-up cost (docs/benchmarks.md, "Not yet measured"): what the ledger recorded of every
    # calibration's start-ups, then one named setting at a time on the kernel model's leader. Each check prints
    # confirmed, refuted or undecided and exits 0 only for confirmed, so an undecided check never reads as a pass.
    check = f"{py} {bench}/next_session_plan.py"
    cals = [f"cal-{slug(model)}" for model in a.models]
    ledgers = " ".join(f"{{root}}/{c}/rigtune-home/ledger" for c in cals)
    phase("k0-startup", f"{check} --check-startup {ledgers} | tee {{out}}/startup.txt",
          "P1: every launched trial has its start-up seconds in the ledger, and every vLLM trial that became ready "
          "has its graph-capture line read; refuted by a ready vLLM trial with no line read (the patterns are wrong "
          "for the installed release)", needs=cals, timeout_s=600)
    phase("k1-rewrites", f"{check} --check-startup {ledgers} --startup-mode rewrites | tee {{out}}/rewrites.txt",
          "P2: which CUDA-graph modes vLLM said it changed, per release and speculation; on 0.33 below SM90 none is "
          "expected under speculation. A table to read against rigtune.kernels' rules, not a pass or fail of them",
          needs=["k0-startup"], timeout_s=600)
    k, ksid = a.kernel_model, safe_id(a.kernel_model)
    has_kernel = f"{rig} recalibrate --help | grep -e --kernel > /dev/null"  # not -q: see pipefail
    k_profile = f"P=$(ls -t {{out}}/rigtune-home/profiles/*/{ksid}/*.json | head -1) && [ -n \"$P\" ]"
    k_budget = f"--budget {a.goal_budget_min}m"
    busy = f"--workload high-concurrency --workload-file {cal_file} --objective throughput {k_budget}"
    phase("k2-sglang-cap", CLEAN + (
        f"{rig} recalibrate {k} {busy} --backend sglang --kernel cgmax=64 > {{out}}/recalibrate.log 2>&1; "
        f"tail -5 {{out}}/recalibrate.log; {k_profile} && "
        f"{check} --check-kernel \"$P\" --setting cgmax=64 --level 64 --min-gain 0.03 | tee {{out}}/result.txt"),
        "P3: SGLang's decode CUDA-graph cap raised to 64 on a card whose default is lower gains 3% or more at 64 "
        "users (predicted +10% on a 35-60 GB card); refuted under 3%, or by an out-of-memory at the same fraction. "
        "On a 24 GB card with 0.5.23 the default is already 48: a weak test there",
        needs=["k0-startup"], requires=f"{has_kernel} && [ -n \"${{SGLANG_PYTHON:-}}\" ]",
        timeout_s=a.goal_budget_min * 60 + 3600)
    phase("k3-vllm-triton", CLEAN + (
        f"{rig} recalibrate {k} {cal_opts.replace(f'--budget {a.cal_budget_min}m', k_budget)} --backend vllm "
        f"--kernel attn=TRITON_ATTN > {{out}}/recalibrate.log 2>&1; tail -5 {{out}}/recalibrate.log; "
        f"{k_profile} && {check} --check-kernel \"$P\" --setting attn=TRITON_ATTN --max-abs 0.05 "
        "| tee {out}/result.txt"),
        "P4: on vLLM 0.33 Triton attention is within 5% of the leader and is not adopted; refuted by a larger "
        "difference (the stage's value was underestimated) or by a failure to start (then a rule, not a trial)",
        needs=["k0-startup"], requires=has_kernel, timeout_s=a.goal_budget_min * 60 + 3600)
    phase("k4-compile-cache", CLEAN + (
        f"{rig} recalibrate {k} {cal_opts.replace(f'--budget {a.cal_budget_min}m', '--budget 10m')} --backend vllm "
        f"--kernel O=1 > {{out}}/recalibrate.log 2>&1; tail -5 {{out}}/recalibrate.log; "
        f"{check} --check-startup {{out}}/rigtune-home/ledger --startup-mode cache | tee {{out}}/result.txt"),
        "P5: a configuration launched again loads its compiled graphs from vLLM's cache, except under ngram_gpu on "
        "the V1 runner; refuted by a second launch of one configuration that compiled again",
        needs=["k0-startup"], requires=has_kernel, timeout_s=3600)
    plan: Dict = {"session": a.session, "home": "rigtune-home", "phases": phases,
                  "notes": "written by benchmarks/next_session_plan.py; see docs/gpu-session.md"}
    if a.hf_home:
        plan["env"] = {"HF_HOME": a.hf_home}  # on the volume, so a pod stop keeps the downloaded models
    return plan


def check_drafts(directory: Path, method: str) -> int:
    """A C06-only run is not release certification; decide the requested method explicitly."""
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    if not files:
        return 1
    try:
        details = json.loads(files[-1].read_text(encoding="utf-8"))[0]["details"]
        checks = [v for k, v in details.items() if k == f"C06-{method}" or k.startswith(f"C06-{method}-")]
        return 0 if checks and all(c["status"] == "pass" for c in checks) else 1
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 1


def _ledger_trials(directories: List[Path]) -> List[dict]:
    """Every trial record of the ledgers under `directories`, read with the ledger's own reader (which skips a line
    that was cut short or changed)."""
    from rigtune import ledger

    out: List[dict] = []
    for directory in directories:
        for path in sorted(directory.rglob("*.jsonl")):
            out += [r for r in ledger.read(path) if r.get("type") == "trial"]
    return out


def check_startup(directories: List[Path], mode: str = "seconds") -> int:
    """What the ledger recorded of engine start-ups. Prints a verdict and returns 0 only for `confirmed`; an empty
    ledger, or one in which nothing was read from any log, is `undecided` (1), never a pass. `refuted` is 2."""
    trials = [t for t in _ledger_trials(directories) if (t.get("outcome") or {}).get("launched")]
    if not trials:
        print("undecided: no launched trial in the ledger")
        return 1
    ready = [t for t in trials if (t.get("startup") or {}).get("seconds") is not None]
    vllm = [t for t in ready if (t.get("engine") or {}).get("name") == "vllm"]
    read = [t for t in vllm if "capture" in t["startup"].get("parsed", [])]
    if mode == "rewrites":
        seen: Dict[tuple, int] = {}
        for t in vllm:
            spec = (t.get("trial") or {}).get("config", {}).get("spec_decode")
            key = ((t.get("engine") or {}).get("version"), "speculating" if spec else "plain",
                   ",".join(t["startup"].get("rewrites", [])) or "none")
            seen[key] = seen.get(key, 0) + 1
        for (version, spec, rewrites), n in sorted(seen.items(), key=str):
            print(f"vLLM {version} {spec}: {n} trials, changed: {rewrites}")
        if not read:
            print("undecided: no vLLM start-up log was read, so an empty list of changes says nothing")
            return 1
        print(f"confirmed: {len(vllm)} vLLM start-ups recorded what the engine changed")
        return 0
    if mode == "cache":
        by_config: Dict[str, List[dict]] = {}
        for t in vllm:
            by_config.setdefault(t.get("config_key", ""), []).append(t)
        again = [(key, runs[1:]) for key, runs in by_config.items() if len(runs) > 1 and "ngram_gpu" not in key]
        later = [t["startup"].get("compile_cached") for _, runs in again for t in runs]
        if False in later:
            print(f"refuted: {later.count(False)} later launches of a configuration compiled again")
            return 2
        if True not in later:
            print("undecided: no configuration was launched twice with its compile lines read")
            return 1
        print(f"confirmed: {later.count(True)} later launches loaded their graphs from the compile cache")
        return 0
    missing = [t for t in trials if "startup" not in t]
    if missing:
        print(f"refuted: {len(missing)} of {len(trials)} launched trials have no start-up record")
        return 2
    if not vllm:
        print(f"undecided: {len(ready)} start-ups timed, none of them vLLM's, so no log line was looked for")
        return 1
    if len(read) < len(vllm):
        print(f"refuted: {len(vllm) - len(read)} of {len(vllm)} ready vLLM trials had no graph-capture line read")
        return 2
    seconds = sorted(t["startup"]["seconds"] for t in ready)
    capture = sorted(t["startup"]["capture_s"] for t in read)
    print(f"confirmed: {len(ready)} start-ups timed, median {seconds[len(seconds) // 2]:.0f} s; "
          f"{len(read)} vLLM graph captures read, median {capture[len(capture) // 2]:.0f} s")
    return 0


def check_kernel(profile: dict, setting: str, level: Optional[int] = None, min_gain: Optional[float] = None,
                 max_abs: Optional[float] = None) -> int:
    """A --kernel trial against the leader it was built on, from a profile's calibration table: tok/s overall or at
    `level` users. Prints a verdict and returns 0 for `confirmed`, 2 for `refuted`, 1 for `undecided` (the setting
    was not tried, its trial or its leader's did not finish, or the level was not measured)."""
    from rigtune.kernels import canonical

    wanted = canonical(setting)
    table = profile.get("calibration_table") or []
    tried = [r for r in table if r.get("stage") == "kernels" and (r.get("config") or {}).get("kernels") == wanted]
    if not tried:
        notes = [n for n in profile.get("notes") or [] if n.startswith("kernel setting")]
        print(f"undecided: {wanted} was not tried" + (f" ({notes[0]})" if notes else ""))
        return 1
    row = tried[-1]
    if row.get("error") or not row.get("launched", True):
        print(f"undecided: the trial with {wanted} did not finish: {str(row.get('error'))[:160]}")
        return 1
    base = {k: v for k, v in row["config"].items() if k != "kernels"}
    leaders = [r for r in table if {k: v for k, v in (r.get("config") or {}).items() if k != "kernels"} == base
               and not (r.get("config") or {}).get("kernels") and not r.get("error")]

    def tok_s(r: dict) -> Optional[float]:
        m = r.get("metrics") or {}
        if level is not None:
            m = (m.get("by_concurrency") or {}).get(str(level)) or {}
        return m.get("tok_s") or None

    if not leaders or tok_s(leaders[-1]) is None or tok_s(row) is None:
        print(f"undecided: no finished trial of the leader without {wanted}"
              + (f" at {level} users" if level is not None else ""))
        return 1
    gain = tok_s(row) / tok_s(leaders[-1]) - 1
    adopted = (profile.get("config") or {}).get("kernels") == wanted
    said = f"{wanted}: {gain:+.1%} tok/s" + (f" at {level} users" if level is not None else "") + \
        (", adopted" if adopted else ", not adopted")
    if min_gain is not None:
        ok = gain >= min_gain
        print(f"{'confirmed' if ok else 'refuted'}: {said} (a gain of {min_gain:.0%} or more was predicted)")
        return 0 if ok else 2
    if max_abs is not None:
        ok = abs(gain) <= max_abs
        print(f"{'confirmed' if ok else 'refuted'}: {said} (within {max_abs:.0%} either way was predicted)")
        return 0 if ok else 2
    print(f"undecided: {said}; no prediction was given to judge it by")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="where to write the plan")
    ap.add_argument("--check-drafts", type=Path)
    ap.add_argument("--check-startup", type=Path, nargs="+", metavar="LEDGER_DIR",
                    help="say what the trial ledgers under these folders recorded of engine start-ups")
    ap.add_argument("--startup-mode", choices=("seconds", "rewrites", "cache"), default="seconds")
    ap.add_argument("--check-kernel", type=Path, metavar="PROFILE",
                    help="compare a --kernel trial with its leader in this profile's calibration table")
    ap.add_argument("--setting", help="the kernel setting --check-kernel looks for, as given to --kernel")
    ap.add_argument("--level", type=int, help="compare at this many users instead of overall")
    ap.add_argument("--min-gain", type=float)
    ap.add_argument("--max-abs", type=float)
    ap.add_argument("--kernel-model", default="google/gemma-4-12B-it",
                    help="the model the kernel phases recalibrate (k2 to k4)")
    ap.add_argument("--expected-method", choices=("assist", "draft", "mtp"))
    ap.add_argument("--pick-jpt", type=Path, metavar="PROFILE",
                    help="print the J/token a profile's pick measured and write no plan (g3-balanced runs this)")
    ap.add_argument("--session", default="next-gpu-session")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--repo", default="/workspace/rigtune-src", help="rigtune's source tree on the pod")
    ap.add_argument("--commit", default=None,
                    help="the commit of that source, which env checks the installed rigtune against (default: "
                         "the COMMIT file in --repo)")
    ap.add_argument("--hf-home", default="/workspace/hf",
                    help="HF_HOME for every phase ('' leaves the environment's)")
    ap.add_argument("--vllm-src", default="/workspace/src/vllm", help="sparse vLLM checkout for auto_tune")
    ap.add_argument("--llm-optimizer-src", default="/workspace/src/llm-optimizer")
    ap.add_argument("--llm-optimizer-bin", default="/workspace/venvs/llmopt/bin/llm-optimizer")
    ap.add_argument("--aiconfigurator-bin", default="/workspace/venvs/aic/bin/aiconfigurator")
    ap.add_argument("--aiconfigurator-python", default="/workspace/venvs/aic/bin/python")
    ap.add_argument("--smoke-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--suffix-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--goal-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ttft-ms", type=float, default=500.0)
    ap.add_argument("--tpot-ms", type=float, default=50.0)
    ap.add_argument("--levels", default="1,4,8")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--cal-budget-min", type=int, default=60)
    ap.add_argument("--goal-budget-min", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check-items", type=int, default=300)
    a = ap.parse_args(argv)
    if a.check_drafts is not None:
        if a.expected_method is None:
            ap.error("--check-drafts needs --expected-method")
        return check_drafts(a.check_drafts, a.expected_method)
    if a.check_startup is not None:
        return check_startup(a.check_startup, a.startup_mode)
    if a.check_kernel is not None:
        if a.setting is None:
            ap.error("--check-kernel needs --setting")
        try:
            profile = json.loads(a.check_kernel.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"undecided: the profile could not be read ({type(exc).__name__})")
            return 1
        return check_kernel(profile, a.setting, a.level, a.min_gain, a.max_abs)
    if a.pick_jpt is not None:
        value = pick_jpt(json.loads(a.pick_jpt.read_text(encoding="utf-8")))
        if value is None:
            print(f"{a.pick_jpt}: no trial of the pick's configuration recorded J/token", file=sys.stderr)
            return 1
        print(repr(value))
        return 0
    if a.out is None:
        ap.error("--out is required")
    if a.commit is None:
        try:
            a.commit = (Path(a.repo) / "COMMIT").read_text(encoding="utf-8").strip()
        except OSError:
            a.commit = ""  # env then fails: the plan names no commit to check the installed rigtune against
    if a.commit and not COMMIT.match(a.commit):
        ap.error(f"--commit {a.commit!r} is not a commit hash")
    plan = build(a)
    a.out.write_bytes((json.dumps(plan, indent=1) + "\n").encode("utf-8"))
    print(f"wrote {a.out}: {len(plan['phases'])} phases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
