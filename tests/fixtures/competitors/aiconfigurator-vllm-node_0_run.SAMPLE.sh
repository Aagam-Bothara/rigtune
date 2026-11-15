#!/bin/bash
# CONSTRUCTED SAMPLE, not output of a real run: written by hand on 15 November 2026 in the shape of
# AIConfigurator's generator/config/backend_templates/vllm/run.sh.j2 (aggregated branch) with engine flags
# as its cli_args.j2 renders them (github.com/ai-dynamo/aiconfigurator). Replace it with
# <save-dir>/<run>/agg/top1/agg/node_0_run.sh from the GPU session.
trap 'echo "Cleaning up..."; kill 0 2>/dev/null' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-8B"}
export HF_TOKEN=${HF_TOKEN:-""}

python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed 's/^/[Frontend] /' &

AGG_WORKERS=1
AGG_GPU=1
for w in $(seq 0 $((AGG_WORKERS - 1))); do
  GPU_LIST=$(seq -s, $((w * AGG_GPU)) $((w * AGG_GPU + AGG_GPU - 1)))
  (
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    export DYN_SYSTEM_PORT=$((8081 + w))
    python3 -m dynamo.vllm \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 --block-size 64 --kv-cache-dtype fp8 --max-model-len 4458 --max-num-seqs 96 --max-num-batched-tokens 8192 --cudagraph-capture-sizes 1 2 4 8 16 32 64 96
  ) 2>&1 | sed "s/^/[Worker $w] /" &
done
wait
