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
