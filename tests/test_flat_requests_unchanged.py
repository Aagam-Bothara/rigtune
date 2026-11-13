"""Flat prompt files and presets send exactly the requests they sent before session files existed.

Session replay is opt-in by file format: a file without sessions must keep its bytes and its workload name, or
every profile cached for it would be served for traffic it was not measured on (or silently recalibrated). The
digests below were recorded from the code before session support was added.
"""

from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.measure import run_trial
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload, get_workload, workload_from_file
from tests.test_measure_and_proxy import _Server


def _recording_engine() -> FastAPI:
    app = FastAPI()
    app.state.bodies = []

    @app.post("/v1/completions")
    @app.post("/v1/chat/completions")
    async def completions(req: Request):
        app.state.bodies.append(await req.body())

        async def gen():
            yield f"data: {json.dumps({'choices': [{'text': 'x '}]})}\n\n"
            yield f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 1}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _digest(bodies) -> str:
    # Requests within a level race each other, so their order is not part of what is compared.
    return hashlib.sha256(b"\n".join(sorted(bodies))).hexdigest()[:16]


def _sent(wl: Workload) -> str:
    app = _recording_engine()
    with _Server(app) as s:
        m = run_trial(s.url, LlmtraceHooks(model_name="m"), wl, warmup=True, counter=TokenCounter())
    assert m.ok
    return _digest(app.state.bodies)


def _flat_file(tmp_path):
    """Every flat form, including the two a session file could be mistaken for: a multi-turn conversation without a
    "session" key, and a "session" key beside a prompt. Both are read as flat, as they always were."""
    p = tmp_path / "flat.jsonl"
    lines = ['"a plain string"', '{"prompt": "a prompt"}', '{"text": "a text"}',
             '{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]}',
             json.dumps({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q1"},
                                      {"role": "assistant", "content": "a1"}, {"role": "user", "content": "why?"}]}),
             json.dumps({"prompt": "a prompt that names a session", "session": "u1"})]
    lines += [json.dumps({"prompt": f"question number {i}"}) for i in range(40)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_a_flat_file_keeps_its_name_and_its_requests(tmp_path):
    p = _flat_file(tmp_path)
    wl = workload_from_file(p, get_workload("chat"))
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    assert wl.name == f"file-flat-{digest}" and not wl.sessions
    assert _sent(workload_from_file(p, get_workload("chat"))) == "1bead5ed33b01094"


# Launch arguments of flat configurations, recorded from the code before session support (the interpreter path,
# which depends on the machine, left out). A session file adds the engines' cache-report flags; nothing else may.
BASE_ARGS = {
    ("vllm", 0): ["-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Llama-3.2-3B-Instruct",
                  "--served-model-name", "meta-llama/Llama-3.2-3B-Instruct", "--host", "127.0.0.1", "--port", "8000",
                  "--max-model-len", "4096", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.90",
                  "--dtype", "bfloat16"],
    ("vllm", 1): ["-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Llama-3.2-3B-Instruct",
                  "--served-model-name", "meta-llama/Llama-3.2-3B-Instruct", "--host", "127.0.0.1", "--port", "8000",
                  "--max-model-len", "8192", "--max-num-seqs", "256", "--gpu-memory-utilization", "0.85",
                  "--max-num-batched-tokens", "8192", "--quantization", "fp8", "--kv-cache-dtype", "fp8_e5m2",
                  "--no-enable-prefix-caching"],
    ("sglang", 0): ["-m", "sglang.launch_server", "--model-path", "meta-llama/Llama-3.2-3B-Instruct",
                    "--served-model-name", "meta-llama/Llama-3.2-3B-Instruct", "--host", "127.0.0.1", "--port", "8000",
                    "--context-length", "4096", "--max-running-requests", "64", "--mem-fraction-static", "0.90",
                    "--dtype", "bfloat16"],
    ("sglang", 1): ["-m", "sglang.launch_server", "--model-path", "meta-llama/Llama-3.2-3B-Instruct",
                    "--served-model-name", "meta-llama/Llama-3.2-3B-Instruct", "--host", "127.0.0.1", "--port", "8000",
                    "--context-length", "8192", "--max-running-requests", "256", "--mem-fraction-static", "0.85",
                    "--chunked-prefill-size", "8192", "--quantization", "fp8", "--kv-cache-dtype", "fp8_e5m2",
                    "--disable-radix-cache"],
}


def test_flat_launch_arguments_are_unchanged(prepared_vllm, monkeypatch):
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    from rigtune.backends import get_backend
    from rigtune.models import Config

    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    for (backend, i), want in BASE_ARGS.items():
        cfg = (Config(backend=backend, quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9),
               Config(backend=backend, quant="fp8", ctx=8192, batch=256, gpu_memory_utilization=0.85,
                      prefill_budget=8192, kv_dtype="fp8_e5m2", prefix_cache=False))[i]
        assert get_backend(backend).launch_spec(cfg, prepared_vllm, 8000).args[1:] == want, (backend, i)


def test_a_preset_keeps_its_requests():
    wl = replace_levels(get_workload("chat-system"), (1, 2))
    assert wl.name == "chat-system"
    assert _sent(wl) == "01a2c8c0b4e33026"


def test_request_keys_beside_a_prompt_leave_a_file_flat(tmp_path):
    """Only a "requests" list, or "request" with "session" and "messages", gives requests whole. These lines hold a
    prompt, so they were read as flat before explicit requests existed and still are: same bytes, same name. The
    digest was recorded from the code before explicit requests were added."""
    from rigtune.calibrate.datasets import read_sessions

    p = tmp_path / "flat-keys.jsonl"
    lines = [{"prompt": "a prompt with a request number", "request": 3},
             {"prompt": "a prompt with a session and a request", "session": "u1", "request": 1},
             {"text": "a text with requests", "requests": [{"messages": [{"role": "user", "content": "x"}]}]},
             {"prompt": "a prompt with requests and a session", "session": "u2", "requests": []}]
    lines += [{"prompt": f"question number {i}", "request": i} for i in range(40)]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    assert read_sessions(p) == []
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.name == f"file-flat-keys-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}" and not wl.sessions
    assert _sent(workload_from_file(p, get_workload("chat"))) == "3ee1ecceb60c93e2"


def test_a_session_file_keeps_its_requests(tmp_path):
    """A turn-by-turn session file sends what it sent before explicit requests existed. One level of one slot, so the
    turns sent do not depend on which request finished first."""
    from dataclasses import replace

    p = tmp_path / "chats.jsonl"
    lines = [{"session": f"c{i}", "system": "Be brief.",
              "messages": [{"role": "user", "content": f"c{i} first"}, {"role": "assistant", "content": "ok"},
                           {"role": "user", "content": f"c{i} second"}]} for i in range(30)]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.name == f"file-chats-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}-sessions"
    assert wl.spec()["history"] == "generated"
    assert _sent(replace(wl, concurrencies=(1,))) == "adf08bdbd76dd9e3"


def _conversations(tmp_path, name, **meta):
    p = tmp_path / f"{name}.jsonl"
    lines = [{"session": f"s{i}", **{k: (v(i) if callable(v) else v) for k, v in meta.items()},
              "messages": [{"role": "user", "content": f"s{i} first"}, {"role": "assistant", "content": "ok"},
                           {"role": "user", "content": f"s{i} second"}]} for i in range(30)]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return p


# A "request" or "requests" key of any shape but the explicit ones is metadata on a conversation, as it was before
# explicit requests existed: same requests, same name, same turn budgets. The digest was recorded from the code
# before explicit requests were added; the three files send the same bodies, since metadata is never sent.
METADATA = {"request-id": {"request": lambda i: f"req-{i}"}, "empty-requests": {"requests": []},
            "requests-of-strings": {"requests": ["r1", "r2"]}}
METADATA_DIGEST = "46cda0a9779f99a9"


def test_request_metadata_on_a_conversation_changes_nothing(tmp_path):
    from dataclasses import replace

    for name, meta in METADATA.items():
        digest = METADATA_DIGEST
        p = _conversations(tmp_path, name, **meta)
        wl = workload_from_file(p, get_workload("chat"))
        assert wl.name == f"file-{name}-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}-sessions", name
        assert wl.history == "generated" and [wl.turn_budget(c) for c in wl.concurrencies] == [8, 24, 48], name
        assert _sent(replace(wl, concurrencies=(1,))) == digest, name


def test_a_whole_number_request_is_explicit_and_named_apart(tmp_path):
    """{"session", "request": 3, "messages"} was a conversation before and is now one explicit request: the name
    carries "-explicit", so a profile cached for the old reading can never be served for the new one."""
    p = _conversations(tmp_path, "numbered", request=lambda i: i)
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.name == f"file-numbered-{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}-sessions-explicit"
    assert wl.history == "explicit"


def replace_levels(wl: Workload, levels) -> Workload:
    from dataclasses import replace

    return replace(wl, n_prompts=4, concurrencies=tuple(levels), prompts=[], fitted=False)
