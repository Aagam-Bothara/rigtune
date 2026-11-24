#!/bin/bash
# Inkling-2-Mini session 2: rigtune itself against stock vLLM and the recipe on many users with long prompts, three
# interleaved rounds after rigtune's own calibration; then the reasoning probe on Omni-MATH. Progress lines "== ".
. /root/v/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
MODEL=thinkingmachines/Inkling-2-Mini
mkdir -p /root/s2
cd /root/s2
echo "== rigtune $(python -c 'import rigtune; print(rigtune.__version__)'), vllm $(python -c 'import vllm; print(vllm.__version__)')"
rigtune probe 2>&1 | grep -iE "gpu|vllm|cuda" | head -6
rigtune plan $MODEL --backend vllm --workload long-concurrent --speculative all > plan.txt 2>&1
grep -E "feasible|note|error" plan.txt | head -8 | sed 's/^/== plan: /'

# 1. rigtune's calibration and the comparison: rigtune's pick, stock vLLM, the recipe, three interleaved rounds.
timeout 7000 rigtune compare $MODEL --backend vllm --workload long-concurrent --speculative all --budget 40m \
  --repeats 3 --rows /root/recipe-rows.json --out /root/s2/compare.json > compare.log 2>&1
echo "== compare exited $?"
grep -E "tok/s|winner|picked|rigtune|stock|recipe|failed|error|Error" compare.log | tail -25 | cut -c1-220 | sed 's/^/== /'
echo "== SPEED DONE"

# 2. Reasoning probe on Omni-MATH, difficulty 5 and up (101 dev problems): effort low and high answered once, and
#    3 votes at low effort, the equal-GPU-time question; the recipe's server with --max-num-seqs 32.
vllm serve $MODEL --port 8000 --tensor-parallel-size 1 --reasoning-parser inkling --max-model-len 262144 \
  --max-num-seqs 32 --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  > reason-server.log 2>&1 &
SERVER=$!
for i in $(seq 1 360); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
echo "== reason server up after $((i * 5)) s"
for run in "low direct" "high direct" "low self-consistency:3"; do
  set -- $run
  python -m rigtune.reasoning --base-url http://127.0.0.1:8000 --model $MODEL --dataset omni --min-level 5 \
    --split dev --methods $2 --thinking default --reasoning-effort $1 --max-tokens 16384 --sequences 16 \
    --context 262144 --server "vllm nightly (recipe + --max-num-seqs 32) on H100 NVL" --keep-texts \
    --out /root/s2/omni-$1-${2%%:*} > /root/s2/omni-$1-${2%%:*}.log 2>&1
  grep -E "% of" /root/s2/omni-$1-${2%%:*}.log | head -2 | sed "s/^/== omni $1 $2: /"
done
kill $SERVER; sleep 10
echo "== SESSION DONE"
