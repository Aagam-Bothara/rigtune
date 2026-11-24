#!/bin/bash
# Inkling-2-Mini session 1 on one H100: the recipe's own command as the baseline, vLLM's KV facts and speed under the
# knobs its issues point at, then a reasoning probe. Results under /root/runs/<config>/; progress lines start "== ".
. /root/v/bin/activate
export HF_HUB_OFFLINE=1
MODEL=thinkingmachines/Inkling-2-Mini
SPEC='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
# The recipe's H100 command (recipes/models/thinkingmachines/Inkling-2-Mini.yaml): TP1, its reasoning parser, the
# bundled DSpark draft, the 262,144-token cap on H100.
RECIPE=(--tensor-parallel-size 1 --reasoning-parser inkling --max-model-len 262144)
mkdir -p /root/runs

start() {  # start NAME ARGS...: launch vLLM, wait for /health, keep its start-up facts
  local name=$1; shift
  mkdir -p /root/runs/$name
  echo "$ vllm serve $MODEL $*" > /root/runs/$name/server.log
  vllm serve $MODEL --port 8000 "$@" >> /root/runs/$name/server.log 2>&1 &
  echo $! > /root/runs/$name/pid
  local t0=$(date +%s)
  for i in $(seq 1 720); do
    if curl -sf localhost:8000/health >/dev/null; then
      echo "== $name ready after $(( $(date +%s) - t0 )) s"
      grep -E "KV cache size|Maximum concurrency|padding layers|Available KV cache memory|num_gpu_blocks|Model loading took|Graph capturing finished|max_num_seqs|max_num_batched_tokens|Using max model len|Chunked prefill is enabled" \
        /root/runs/$name/server.log | sed -E 's/^.*\] //' | cut -c1-240 | tee /root/runs/$name/facts.txt
      return 0
    fi
    if ! kill -0 $(cat /root/runs/$name/pid) 2>/dev/null; then
      echo "== $name EXITED"; grep -E "Error|error:|OutOfMemory|Traceback" /root/runs/$name/server.log | tail -6 | cut -c1-300
      return 1
    fi
    sleep 5
  done
  echo "== $name TIMEOUT"; return 1
}

stop() { kill $(cat /root/runs/$1/pid) 2>/dev/null; sleep 5; pkill -f "vllm serve" 2>/dev/null; sleep 10; }

bench() {  # bench NAME USERS...: Dolly prompts, 256 output tokens each (ignore_eos), at each concurrency
  local name=$1; shift
  for c in "$@"; do
    local n=$(( c * 4 )); [ $n -lt 16 ] && n=16; [ $n -gt 256 ] && n=256
    vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model $MODEL --port 8000 \
      --dataset-name custom --dataset-path /root/dolly.jsonl --custom-output-len 256 --ignore-eos \
      --max-concurrency $c --num-prompts $n --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir /root/runs/$name --result-filename c$c.json > /root/runs/$name/bench-c$c.log 2>&1
    python - "$name" "$c" <<'EOF'
import json, sys
name, c = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(f"/root/runs/{name}/c{c}.json"))
    print(f"== {name} users {c}: {r.get('output_throughput', 0):.0f} out tok/s, ttft p50 {r.get('median_ttft_ms', 0):.0f} ms, "
          f"tpot p50 {r.get('median_tpot_ms', 0):.1f} ms, completed {r.get('completed')}/{r.get('num_prompts')}")
except Exception as exc:
    print(f"== {name} users {c}: no result ({exc})")
EOF
  done
  grep -h "acceptance\|Mean acceptance length\|Accepted:" /root/runs/$name/server.log | tail -2 | cut -c1-200
}

run() {  # run NAME USERS ARGS...
  local name=$1 users=$2; shift 2
  if start "$name" "$@"; then bench "$name" $users; fi
  stop "$name"
}

# 1. the baseline, as published
run recipe "1 8 32 64" "${RECIPE[@]}" --speculative-config "$SPEC"
# 2. memory facts and speed under the knobs: no drafter; a 2,048-token budget (vLLM #69280: 62 vs 98 KiB a token);
#    a 32k context cap (more concurrent requests admitted)
run nospec "1 8 64" "${RECIPE[@]}"
run budget2048 "8 64" "${RECIPE[@]}" --speculative-config "$SPEC" --max-num-batched-tokens 2048
run ctx32k "8 64" --tensor-parallel-size 1 --reasoning-parser inkling --max-model-len 32768 --speculative-config "$SPEC"
echo "== SPEED DONE"

# 3. reasoning probe on the recipe server: the 98 development problems (MATH levels 4-5), thinking at three efforts
if start reason-server "${RECIPE[@]}" --speculative-config "$SPEC" --max-num-seqs 32; then
  for effort in low medium high; do
    python -m rigtune.reasoning --base-url http://127.0.0.1:8000 --model $MODEL --split dev --methods direct \
      --thinking default --reasoning-effort $effort --max-tokens 16384 --sequences 16 --context 262144 \
      --server "vllm nightly (recipe + --max-num-seqs 32) on $(nvidia-smi --query-gpu=name --format=csv,noheader)" \
      --keep-texts --out /root/runs/reason-$effort > /root/runs/reason-$effort.log 2>&1
    grep -E "^direct:|% of" /root/runs/reason-$effort.log | head -2 | sed "s/^/== effort $effort: /"
  done
  stop reason-server
fi
echo "== SESSION DONE"
