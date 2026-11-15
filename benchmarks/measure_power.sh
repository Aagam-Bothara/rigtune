#!/usr/bin/env bash
# Measure --power on real hardware: the one feature so far tested only against a simulated NVML.
# It needs root on a machine you control (NVML power limits and clock locks are machine-wide), so it
# cannot run on a rented container. From the repo root, in the environment where rigtune is installed:
#
#   sudo -E env PATH="$PATH" benchmarks/measure_power.sh [model] [workloads...]
#
# Defaults: Qwen/Qwen2.5-3B-Instruct on `generation` (decode-heavy) and `rag` (prefill-heavy), the pair the
# open question in docs/benchmarks.md asks about. Each workload is compared against stock settings twice:
# `efficiency` (lowest joules per token above a throughput floor) and `balanced` (a power setting is
# accepted only if it costs under 2% of throughput). The GPU's power limit and clocks are restored on exit.
# OUT (the results directory) and RIGTUNE (the command, such as "/path/python -m rigtune.cli") may be set in the
# environment.
set -euo pipefail
MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}
shift || true
WORKLOADS=${*:-generation rag}
OUT=${OUT:-benchmarks/strategies/results-power}
RIGTUNE=${RIGTUNE:-rigtune}

$RIGTUNE power status
trap '$RIGTUNE power reset || true' EXIT
for wl in $WORKLOADS; do
  for objective in efficiency balanced; do
    echo "=== $MODEL / $wl / $objective / --power both"
    $RIGTUNE compare "$MODEL" --workload "$wl" --objective "$objective" --power both --out "$OUT"
  done
done
echo "results in $OUT; regenerate SUMMARY.md with the command in benchmarks/summarize_strategies.py"
