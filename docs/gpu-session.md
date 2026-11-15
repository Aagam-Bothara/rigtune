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
