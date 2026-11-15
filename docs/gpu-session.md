# Running a GPU session

On 14 November 2026 a rented A40 deleted itself three hours after its run, before its files were copied back, and
only the figures read over SSH during the run survived ([benchmarks](benchmarks.md#against-stock-engines-and-vllms-auto_tune-14-november-2026-raw-files-lost)).
The scripts below make each part of a session a unit that is complete on disk the moment it ends.

**What has run where.** Everything on this page has run against fakes on a laptop only (the tests named below).
None of it has run on a GPU or on a RunPod pod yet.

## The phase runner

`benchmarks/gpu_session.py` runs a session as named phases from a plan file, in order, each into its own
directory under the session root:

```bash
python benchmarks/gpu_session.py run --plan plan.json --root /workspace/session --dry-run   # print, run nothing
python benchmarks/gpu_session.py run --plan plan.json --root /workspace/session
python benchmarks/gpu_session.py status --root /workspace/session
```

- Each phase's directory holds `manifest.json` (phase, command and a hash of it, start and end in UTC, exit code,
  status, the phase's process id, and the versions of rigtune, vLLM, SGLang, torch, transformers and the other
  tools installed when it started, read from each tool's own interpreter that the plan names under `pythons`)
  and `output.log` (everything the command printed).
- **One runner per root.** `<root>/runner.lock` holds the running runner's pid. A second runner on the same root
  refuses to start while that process lives, and takes the lock over once it is gone (killed, or the pod
  restarted).
- **Resumable.** Run the same command again after a crash, a timeout or a lost SSH session: phases that finished
  (`done`) are skipped, every other phase runs again, and its earlier directory is moved whole to
  `_attempts/<phase>.<n>` first, so a failed or interrupted attempt's files are kept, never overwritten. A
  manifest still saying `running` means the runner itself was stopped mid-phase: before the move, the process
  group that phase recorded is ended and the plan's `cleanup` runs, and the move marks it `interrupted`.
- **Edited plans.** A finished phase whose command has changed since is reported `stale` and kept as it was,
  never skipped silently; `--redo-stale` runs it again (the old result goes to `_attempts`).
- **Independent.** A failed phase never stops the session. A phase that `needs` one that did not finish is
  recorded as `blocked`; every other phase still runs.
- **Hooks.** A phase with `requires` runs only when that command exits 0, and is recorded as `skipped`, with the
  command's last lines as the reason, when it does not; it is checked again on the next run. That is how a plan
  names a command another series is still adding (`rigtune conformance`) without failing on a rigtune that
  lacks it.
- **Nothing left writing.** String commands run under `bash -o pipefail`, so a failing command piped into `tee`
  fails its phase. On Linux each phase is its own process group: whatever is left in it when the phase ends, by
  timeout or not, is killed, then the plan's `cleanup` command runs (the next-session plan kills leftover GPU
  processes there and waits for an empty card), and only then is the phase's state written.
- With `"home"` in the plan, every phase shares one `RIGTUNE_HOME` under the root, so a comparison reuses the
  calibration an earlier phase cached, and the profiles, trial ledger and engine logs a phase added there are
  copied into its own directory (files over `home_copy_max_mb`, 512 by default, are listed in the manifest
  instead; unfinished `*.tmp` files are not copied, and are listed too).
- `touch <root>/STOP` stops at the next phase boundary; the runner renames the file to `STOP.seen-<time>`, so the
  next run goes on. `--redo PHASE` runs a finished phase again. `--only PHASE...` runs some phases and does not
  count as the session finishing.
- The plan's `env` is recorded in every manifest, so it must not carry a secret. Export a token the commands
  need (`HF_TOKEN`) in the shell that starts the runner; it reaches the phases and is neither recorded nor
  printed.

Tests: `tests/test_gpu_session.py` (a phase that kills the runner midway and a rerun, a second runner refused,
a dead runner's lock taken over and its phase's processes ended, a timeout, cleanup before the state, a failed
phase and its dependents, a missing requirement, a stale phase, STOP consumed, `--only`, the dry run, the copied
home files, the token kept out of manifests).

## Copy-back after every phase

When a directory reaches a final state (a phase `done`, `failed`, `timeout`, `skipped` or `blocked`, or an
attempt set aside as `interrupted`), the runner writes `SHA256SUMS` in it, in `sha256sum` format, over every
file but the list itself, and lists it with its state in `<root>/index.json`. A finished directory is never
written to again, so its list stays true; one recorded as finished without its list (the runner died between
the two) gets it when the next run starts. On the laptop, `benchmarks/gpu_pull.py` reads that index over SSH and copies each finished
directory it does not already hold, as one `tar` stream unpacked by Python (no rsync or local tar needed, so it
runs from Windows Git Bash), into a short staging folder (`.p/<hash>`, so the path is no longer than the final
one; on Windows every path is opened in its extended-length form, so a deep phase is not cut off at 260
characters); it checks every file against the pod's list (each listed file present with that hash, nothing
unlisted) and only then puts the copy in place. A local copy it replaces is moved to
`_superseded/<phase>.<n>`, never deleted. The root's own logs (`session.log`, `copyback.log`, the runner's
output) are copied as they are into `_root/` on every pull.

From Git Bash on the laptop, with the pod's public IP and the port mapped to its port 22 (the pod's Connect tab,
"SSH over exposed TCP"). `MSYS_NO_PATHCONV=1` stops Git Bash rewriting `/workspace/session` into a Windows path
before Python sees it; the puller refuses a `--remote` that looks like a Windows path, with that advice.

```bash
MSYS_NO_PATHCONV=1 python benchmarks/gpu_pull.py pull --host <pod IP> --port <its SSH port> --identity ~/.ssh/id_ed25519 \
    --remote /workspace/session --dest gpu-results/session --watch 300
python benchmarks/gpu_pull.py verify --dest gpu-results/session    # any time, offline; exit 0 = complete
```

- **Idempotent.** A directory whose local copy already verifies is not copied again; run it as often as you like.
  `--watch 300` pulls every five minutes until the runner has finished and everything is copied. A pass that
  fails for any reason (the pod unreachable, ssh timing out, a file locked on the laptop) is printed as
  `PULL FAILED` and retried at the next interval; only a configuration error or Ctrl-C ends the loop.
- **Complete means complete.** `verify` checks the copy against the pod's index as last pulled: every directory
  the pod recorded as finished must be present and match its list, and the runner must have finished a full
  run. It exits 3 otherwise, and on an empty destination.
- **Loud.** A copy cut short, a file changed after its list was written, or a finished directory without a list
  prints `INCOMPLETE <phase>: <what is wrong>` and exits 3, leaving the earlier local copy as it was; the next
  pull tries again. A phase still running is never copied as complete; `--running` also copies it, as it is, to
  `<phase>.running-snapshot` (unverified).
- **No secrets.** ssh reads the key from `--identity`; nothing here reads, stores or prints it. The plan's `env`
  is recorded in manifests, so a token goes in the environment of the shell that starts the runner, never in
  the plan.
- **Pushing instead.** `gpu_session.py run --after-phase CMD` runs `CMD` after each finished directory, with
  `{unit}`, `{rel}` and `{root}` replaced, for example a copy to object storage whose credentials are in the
  pod's environment. A failing `CMD` prints `COPY-BACK FAILED` and is logged in `<root>/copyback.log`; it never
  fails the phase.

What RunPod documents, read on 15 November 2026: `/workspace` (the volume disk) survives a pod stop but is
deleted when the pod is terminated, and the container disk is cleared on a stop; only a network volume outlives
the pod. scp, sftp and rsync need "SSH over exposed TCP" (a pod with a public IP and port 22 exposed, as the
RunPod PyTorch templates are); the proxied `ssh <pod>@ssh.runpod.io` login does not support scp or sftp, and the
puller's non-interactive `ssh host command` is not documented to work through it, so use the exposed port.
RunPod's rsync instructions assume Linux or WSL, which is why the puller does not use rsync. `runpodctl send
<file>` / `runpodctl receive <code>` (preinstalled on pods; one code per transfer, typed by hand) is a fallback
for a single phase: tar its directory, send it, unpack it under the destination and run `gpu_pull.py verify`.

Tests: `tests/test_gpu_pull.py` (a copy cut short with and without the transport reporting it, then a rerun; an
interrupted copy over a good earlier one; a verified copy kept when a rerun replaces it; a finished phase
without its list, repaired by the next run; a phase's own `.tmp` file; a file changed after its list; a missing
list; a tar member outside its directory; a running phase; `verify` on a missing phase, an unfinished session and
an empty destination; a damaged local copy; `--watch` through an ssh timeout and a locked file; a Windows-looking
`--remote`; a path over 260 characters; the root's logs; a pull through a fake `ssh` client, with a space in the
remote path).

## Competitor rows

`benchmarks/competitor_rows.py` turns two other tuners' picks into `rigtune compare --rows` entries, so each is
launched and scored like every other row (same runner, held-out prompts, levels, limits and interleaved
repeats); what the tool reported for itself goes into the row's source and is never scored.

| Tool | What it is (read from its repository on 15 November 2026) | Its row |
|---|---|---|
| [BentoML llm-optimizer](https://github.com/bentoml/llm-optimizer) | Package 0.0.1, no release: `pip install -e .` from a clone. A grid runner: launches every point of `--server-args` (`vllm serve {model} --host {host} --port {port}`, or SGLang's `launch_server`, each argument appended as `--name=value`), drives it with its own client (adapted from SGLang's bench_serving), filters by `--constraints`, writes `--output-json` with `best_configurations`. | `best_output_throughput_constrained`, the highest output tok/s within the limits; not applicable when no run met them, or when it ran without constraints. |
| [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) | `pip install aiconfigurator` (0.12.0, 18 September 2026; Linux x86-64 only; "maintenance-only" per its README). Predicts from timings collected per system and engine release; launches nothing. `cli default ... --save-dir D` writes a Dynamo launch script per configuration. | The engine flags on the worker line of `agg/top1/agg/node_0_run.sh` (`python3 -m dynamo.vllm --model ... <flags>`), launched under the engine's own server. Not applicable on a card with no AIConfigurator system (A40, RTX 4090, L4, H100 NVL or PCIe, H200 NVL, GH200...) or a pair its own `cli support` refuses. |

**Fairness.** Both get the comparison's model, card, p95 TTFT and TPOT limits, and request lengths: the
calibration's mean prompt and answer tokens (`benchmarks/vllm_auto_tune.py lengths`, as auto_tune gets them).
llm-optimizer gets `--constraints "ttft:p95<T ms;itl:mean<P ms"` (it documents ttft, itl and e2e_latency; mean
inter-token latency equals mean TPOT without speculation; a value equal to its limit fails, where rigtune's
passes), random prompts of those fixed lengths (`random_range_ratio=1.0`), `max_concurrency` at the comparison's
levels, and a 3 x 3 server grid over the batch cap and prefill budget with the context fixed at the comparison's.
AIConfigurator gets `--total-gpus 1 --isl --osl --ttft --tpot` and the system named for the card; the context
it sizes from `--isl` and `--osl` is replaced by the comparison's, as llm-optimizer's grid fixes it, so no row
serves a shorter context than the others. Each row records the tool's version (llm-optimizer's clone commit;
AIConfigurator's package version, read from the tool's own environment). Their workloads stay their own: random fixed-length prompts, or a model of
the card, not the comparison's prompts; each row's source says so.

**What is carried and what is not.** Settings the tool sets become the row's `Config` fields (batch cap,
prefill budget, context, memory fraction, KV dtype, prefix cache, fp8) or extra flags passed through unchanged
(`--schedule-conservativeness`, `--block-size`, `--page-size`, ...). A setting the tool left alone stays at the
engine's default, except the context and batch cap that every rigtune launch passes: the comparison's context,
and a batch cap of 256 (the value rigtune's stock rows take as the engine's default; vLLM's own is 1024 on cards
with 70 GiB or more, except an A100), which the source states. Both tools set the batch cap in practice, so
that value rarely matters. List-valued CUDA-graph sizes are left at the engine's own list, with a note. A pick needing more than
one card, speculative settings, no tokenizer, or a quantization other than fp8 makes the row not applicable,
with the reason, rather than measured as something else.

**Not verified until the GPU session.** The adapters were tested only against files built by hand from each
tool's source and templates (`tests/fixtures/competitors/*.SAMPLE.*`, each labelled as such), never against a
real run's output; the GPU session must save the real files and replace the samples. Assumed, not yet seen
working: that llm-optimizer (last code commit September 2025) runs against vLLM 0.29 / SGLang 0.5.19 at all,
that `--server-cmd` with `--framework sglang` launches SGLang from its own environment, that
`random_range_ratio=1.0` gives fixed lengths in its client, which systems AIConfigurator 0.12.0 really has data
for (its README lists h100_sxm, h200_sxm, b200_sxm, gb200 and a100_sxm; this session's plan read l40s data at a
later commit), and that its flags mean the same under `vllm serve` as under `dynamo.vllm`.

Tests: `tests/test_competitor_rows.py`.

## The next session, step by step

`benchmarks/next_session_plan.py --out plan.json` writes the plan for the next session (36 phases for the two
default models; `--models`, `--cal-budget-min`, `--repeats`, `--hf-home` and the tool paths are options):

| Phases | What they settle ([Not yet measured](benchmarks.md#not-yet-measured)) |
|---|---|
| `env`, `hub-access`, `prompts` | the machine and every package version, and that the rigtune the runner's Python imports is the source at the plan's commit (`env` fails otherwise, and every other phase is then `blocked`); that the Hub token reaches every model the session loads, gated ones included; the Dolly calibration and held-out prompt files |
| `selfcheck`, `smoke` | the load generator on the pod's CPU ([Limits](benchmarks.md#limits)); every search feature launched once |
| `engine-probe`, `conformance-vllm`, `conformance-sglang` | items 20 and 21: `rigtune probe --engines` and `rigtune conformance <engine> --out`; each is skipped, with the reason, only on a rigtune whose `--help` does not list the option it passes |
| `suffix-facts`, `suffix-cal` | item 19, why suffix decoding failed to start on 14 November |
| `g1-...` to `g4-...` | item 18, custom goals G1 to G4. G2 calibrates and serves with the same options (the budget is part of a profile's cache key) and fails unless serve names the efficiency preset and starts from the cache without calibrating. G4's energy limit is the balanced pick's J/token, which `g3-balanced` writes to `g4_jpt.txt` |
| `power` | item 5 and G5, into the phase's own `results/`; skipped unless `RIGTUNE_SESSION_POWER=1` says the runner has root on a machine it controls, which a rented pod is not |
| `cal-`, `ledger-`, `lengths-`, `autotune-`, `llmopt-vllm-`, `llmopt-sglang-`, `aic-vllm-`, `aic-sglang-`, `cmp-<model>` | per model (Qwen3.5-9B, Gemma 4 12B): rigtune's calibration (`--speculative all`, so Qwen3.5's MTP head too: item 16); that its profile passes `rigtune ledger verify` and `ledger rebuild` writes the same file byte for byte; the request lengths every tuner gets; each competitor's row (always written, as a pick or as not applicable); then `compare` on the held-out prompts with stock, bare stock, expert, auto_tune, llm-optimizer and AIConfigurator rows, 3 interleaved repeats, answers graded on 300 GSM8K problems, `--verdict` |

The comparison passes exactly the calibration's search options, so it reuses the cached profile; a competitor
phase that fails never blocks it, because each comparison needs only its calibration, and a row file that is
missing is named in the comparison's `rows-missing.txt`. Each GPU phase starts by killing leftover GPU processes
by the PIDs `nvidia-smi` lists (phases run one at a time). A phase fails when the file it exists to write is
missing or empty: `lengths-` without request lengths, a tuner whose command could not be written, an auto_tune
run that left no row (a not-applicable row is written in its place). The plan sets `HF_HOME=/workspace/hf` for
every phase, so the models are on the volume disk and a pod stop keeps them.

**Before renting.** Follow the recipe that worked before (image, the bandwidth check before keeping a pod,
`transformers<5` with vLLM 0.11, deleting uv's cache after installs), with SSH over the exposed TCP port. On the
pod: vLLM 0.29.0 in the main Python; SGLang 0.5.19 in its own venv with `SGLANG_PYTHON` pointing at it;
arctic-inference from the [install line](usage.md#suffix-decoding) (or the broken build first, for item 19's
comparison); a sparse vLLM checkout at the installed tag in `/workspace/src/vllm` (see
[Reproducing a comparison](benchmarks.md#reproducing-a-comparison)); llm-optimizer cloned to
`/workspace/src/llm-optimizer` and installed with `pip install -e .` into `/workspace/venvs/llmopt`;
`pip install aiconfigurator` into `/workspace/venvs/aic` (Linux x86-64, Python 3.11 to 3.13). Separate venvs
keep the tools' dependencies from moving vLLM's.

**From Git Bash on the laptop.**

```bash
export MSYS_NO_PATHCONV=1                        # or Git Bash rewrites /workspace/... into a Windows path
POD="-p <port> -i ~/.ssh/id_ed25519 root@<IP>"
# 1. Send the source of this checkout (what is committed, as LF files) and its commit.
git -c core.autocrlf=false archive --format=tar HEAD | ssh $POD 'mkdir -p /workspace/rigtune-src && tar -x -C /workspace/rigtune-src'
git rev-parse HEAD | ssh $POD 'cat > /workspace/rigtune-src/COMMIT'
# 2. Install that source into the Python that runs vLLM. Its absolute path is kept in /workspace/vllm-python,
#    and every later step uses it: the runner's interpreter is what each phase runs rigtune with.
ssh $POD 'VPY=$(head -1 "$(command -v vllm)" | sed "s/^#! *//") && test -x "$VPY" && echo "$VPY" > /workspace/vllm-python && "$VPY" -m pip install -e /workspace/rigtune-src && cd / && "$VPY" -c "import rigtune; print(rigtune.__file__)"'
# 3. Write the plan with this checkout's commit, which the env phase checks the pod's install against, and
#    look at it.
ssh $POD "VPY=\$(cat /workspace/vllm-python); cd /workspace/rigtune-src && \$VPY benchmarks/next_session_plan.py --out /workspace/plan.json --commit $(git rev-parse HEAD) && \$VPY benchmarks/gpu_session.py run --plan /workspace/plan.json --root /workspace/session --dry-run"
# 4. Start the runner, detached. The Hub token is typed unseen and goes over stdin: it is in no file, no
#    command line and no shell history.
read -rs HF_TOKEN
printf '%s\n' "$HF_TOKEN" | ssh $POD 'read -r HF_TOKEN; export HF_TOKEN HF_HOME=/workspace/hf SGLANG_PYTHON=/workspace/venvs/sglang/bin/python; VPY=$(cat /workspace/vllm-python); mkdir -p /workspace/session; cd /workspace/rigtune-src && setsid nohup "$VPY" benchmarks/gpu_session.py run --plan /workspace/plan.json --root /workspace/session > /workspace/session/runner.out 2>&1 < /dev/null &'
# 5. After a few minutes: env and hub-access must both say done before the session is left alone.
ssh $POD '$(cat /workspace/vllm-python) /workspace/rigtune-src/benchmarks/gpu_session.py status --root /workspace/session'
# 6. Keep pulling every finished phase, verified, until the runner has finished.
python benchmarks/gpu_pull.py pull --host <IP> --port <port> --identity ~/.ssh/id_ed25519 --remote /workspace/session --dest gpu-results/session --watch 300
# Any time: stop at the next phase boundary (the runner renames the file to STOP.seen-<time> when it stops).
ssh $POD 'touch /workspace/session/STOP'
```

If `env` failed, read `env/output.log`: the install in step 2 did not take, or the plan was written for another
commit. If `hub-access` lists a model as `NO ACCESS`, accept its licence on the Hub with the token's account (or
fix the token), then run step 4 again with `--redo hub-access` added.

**Resuming.** Step 4 again is how to resume after anything stops the runner: finished phases are skipped. Never
start it while a runner is still going; one that is started refuses, naming the first one's pid, and changes
nothing. Nothing in the session needs a second runner. G4 reads its energy limit from `g3-balanced/g4_jpt.txt`
just before it starts; to give it another value, write the number to `/workspace/session/G4_JPT` before G4 is
reached (that file wins). Where `g3-balanced` recorded no J/token (no power readings on the pod) and there is no
such file, both G4 phases are `skipped` with that reason. Do not send any SSH command containing `vllm serve`
while an `autotune-*` phase runs: auto_tune runs `pkill -if "vllm serve"`.

**Before terminating the pod**: the puller has printed "the session runner has finished" with nothing
`INCOMPLETE`, and `python benchmarks/gpu_pull.py verify --dest gpu-results/session` exits 0. Anything still
missing can be read on the pod with `status`, pulled again, or sent by hand with `runpodctl send`.

**Prediction, not a measurement: the session's length.** From recorded times (a 60-minute calibration budget,
auto_tune and llm-optimizer each cut at that budget, a compare of about 14 rows x 3 repeats at about 4 minutes
a run): about 7 hours per model, 2 to 3 hours of goal and suffix phases and 1 to 2 hours of setup, environment
and smoke, so 17 to 20 hours, about 9 to 10 USD on an A40 at 0.49 USD an hour and 15 to 17 on an L40S at 0.86.
The two conformance runs are not in that figure: they have never run on a GPU, so there is no time to predict
from. `--only` and `--models` cut it down; the comparisons go last, so a run that is stopped early still has
the checks.

**What stays unverified until the session.** Everything on this page. In particular: that the puller's
`ssh host command` works against a RunPod pod's exposed port and that RunPod's image has `tar` and `sha256sum`
(both standard on its Ubuntu images); that the first line of the `vllm` script names its Python directly, and
that `pip install -e` accepts a source tree sent without `.git`; that serve's log shows `-> cache hit` and
`-> calibrate` as G2's check reads them; that a profile's pick appears in its calibration table with the same
configuration, which is how `g3-balanced` finds its J/token; every competitor assumption listed
[above](#competitor-rows); and every prediction in [Not yet measured](benchmarks.md#not-yet-measured).

Tests: `tests/test_next_session_plan.py` (the comparison reuses the calibration's options; every tuner gets the
same model, limits and lengths; every competitor phase leaves a row; every rigtune command in the plan parses
with rigtune's own CLI; G2's two halves pass the same options. Where a bash runs, phases are run as the runner
runs them: `env` against a matching, a different and a missing commit; G2's check on a served, a calibrated and
a wrapped log; G4's requirement with and without a limit file; `lengths-` without a profile; a tuner given empty
lengths; auto_tune without a row; `measure_power.sh` with its output and command redirected; and every shell
command through `bash -n`).
