"""A fake vLLM or SGLang for the conformance run's tests: it takes the real launch arguments rigtune builds and
behaves, by them, the way the checks expect an engine to.

    python tests/conformance_engine.py --fake-engine vllm <the flags of rigtune's launch command>

- It writes the start-up lines rigtune's memory parser reads (vLLM's or SGLang's), then serves /health,
  /v1/completions, /v1/chat/completions and /metrics.
- A repeated chat prompt reports cached tokens in its usage, as the engines do, only when launched with vLLM's
  --enable-prompt-tokens-details or SGLang's --enable-cache-report.
- /metrics has vLLM's prefix-cache counters, and its accepted-token counter under --speculative-config.
- Batch 1024 with a 32768-token prefill budget fails at start-up with CUDA's out-of-memory error.
- A reserved fraction above 0.95 is refused the way vLLM's free-memory check refuses it (v1/worker/utils.py, line 761
  at v0.33.0): a ValueError before anything is loaded, which is not an out-of-memory error.
- FAKE_FAIL lists flag values that fail at start-up with another error, e.g. FAKE_FAIL=fp8_e5m2,suffix.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_LOG = ("INFO Model loading took 0.93 GiB memory and 4.10 seconds\n"
            "INFO Total non KV cache memory: 1.62GiB; torch peak memory increase: 0.40GiB; non-torch forward increase "
            "memory: 0.05GiB; weights memory: 0.93GiB.\n"
            "INFO Available KV cache memory: 18.20 GiB\n"
            "INFO GPU KV cache size: 1,590,272 tokens\n")
SGLANG_LOG = ("[2026-11-20 10:00:00] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=21.20 GB, "
              "mem usage=0.98 GB.\n"
              "[2026-11-20 10:00:01] KV Cache is allocated. #tokens: 1500000, K size: 8.50 GB, V size: 8.50 GB\n")
OOM_LOG = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
REFUSED_LOG = ("ValueError: Free memory on device cuda:0 (23.4/24.0 GiB) on startup is less than desired GPU memory "
               "utilization ({fraction}, 23.76 GiB). Decrease GPU memory utilization or reduce GPU memory used by "
               "other processes.\n")


def main() -> None:
    argv = sys.argv[1:]
    engine = argv[argv.index("--fake-engine") + 1]

    def value(flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None

    port = int(value("--port"))
    fraction = value("--gpu-memory-utilization") or value("--mem-fraction-static")
    fail = [f for f in os.environ.get("FAKE_FAIL", "").split(",") if f]
    if fraction and float(fraction) > 0.95:
        sys.stdout.write(REFUSED_LOG.format(fraction=fraction))
        sys.stdout.flush()
        sys.exit(1)
    if "1024" in argv and "32768" in argv:
        sys.stdout.write(OOM_LOG)
        sys.stdout.flush()
        sys.exit(1)
    for word in fail:
        if any(word in a for a in argv):
            sys.stdout.write(f"ValueError: unsupported setting {word}\n")
            sys.stdout.flush()
            sys.exit(1)
    reports = "--enable-prompt-tokens-details" in argv or "--enable-cache-report" in argv
    speculative = "--speculative-config" in argv or "--speculative-algorithm" in argv
    seen = set()
    counters = {"queries": 0, "hits": 0, "accepted": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/metrics" and engine == "vllm":
                lines = [f'vllm:prefix_cache_queries_total{{engine="0"}} {counters["queries"]}',
                         f'vllm:prefix_cache_hits_total{{engine="0"}} {counters["hits"]}',
                         'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 10']
                if speculative:
                    lines.append(f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {counters["accepted"]}')
                self._send(200, "\n".join(lines) + "\n", "text/plain")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            tokens = int(req.get("max_tokens", 4))
            if speculative:
                counters["accepted"] += tokens // 2
            usage = {"prompt_tokens": 10, "completion_tokens": tokens}
            if self.path == "/v1/chat/completions":
                prompt = json.dumps(req.get("messages"))
                counters["queries"] += 400
                if prompt in seen:
                    counters["hits"] += 384
                if reports:
                    usage["prompt_tokens_details"] = {"cached_tokens": 384 if prompt in seen else 0}
                seen.add(prompt)
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": "x " * tokens}}],
                                 "usage": usage})
            elif self.path == "/v1/completions":
                self._send(200, {"choices": [{"text": "x " * tokens}], "usage": usage})
            else:
                self._send(404, {"error": "not found"})

    sys.stdout.write(VLLM_LOG if engine == "vllm" else SGLANG_LOG)
    sys.stdout.flush()
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
