#!/bin/bash
# CONSTRUCTED SAMPLE, not output of a real run: written by hand on 15 November 2026 in the shape of
# AIConfigurator's generated aggregated launch script, with engine flags as its
# generator/config/backend_templates/sglang/cli_args.j2 renders them (values double-quoted, booleans bare)
# (github.com/ai-dynamo/aiconfigurator). Replace it with <save-dir>/<run>/agg/top1/agg/node_0_run.sh from the
# GPU session.
trap 'echo "Cleaning up..."; kill 0 2>/dev/null' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}

python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed 's/^/[Frontend] /' &

for w in 0; do
  (
    export CUDA_VISIBLE_DEVICES=0
    python3 -m dynamo.sglang \
      --model-path "$MODEL_PATH" \
      --tensor-parallel-size "1" --pipeline-parallel-size "1" --data-parallel-size "1" --page-size "64" --kv-cache-dtype "fp8_e4m3" --mem-fraction-static "0.85" --context-length "4458" --max-running-requests "128" --chunked-prefill-size "8192" --cuda-graph-max-bs "128" --enable-mixed-chunk
  ) 2>&1 | sed "s/^/[Worker $w] /" &
done
wait
