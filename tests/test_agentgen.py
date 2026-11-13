"""The MemGPT-like agent trace generator: deterministic bytes, and requests that rewrite where the design says."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rigtune.calibrate import agentgen, datasets
from rigtune.calibrate.agentgen import AgentShape, generate, to_jsonl

# sha256 of to_jsonl(generate(AgentShape(), 3, 0)), recorded when the generator was written. A change here changes
# every generated file, and so the name of every profile calibrated on one: change it only on purpose.
GOLDEN = "99a4a284e76eee174e9404dfb2df06a936c01e49bbe1b65a6a632307233ae77d"

_spec = importlib.util.spec_from_file_location(
    "make_agent_trace", Path(__file__).resolve().parents[1] / "benchmarks" / "make_agent_trace.py")
mat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mat)


def _sessions(tmp_path, shape=AgentShape(), n=6, seed=0, name="agent.jsonl"):
    p = tmp_path / name
    with p.open("w", encoding="ascii", newline="\n") as fh:
        fh.write(to_jsonl(generate(shape, n, seed)))
    return datasets.read_sessions(p)


def test_the_default_output_is_pinned():
    text = to_jsonl(generate(AgentShape(), 3, 0))
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == GOLDEN


def test_the_same_seed_gives_the_same_bytes_and_another_does_not():
    assert to_jsonl(generate(AgentShape(), 2, 5)) == to_jsonl(generate(AgentShape(), 2, 5))
    assert to_jsonl(generate(AgentShape(), 2, 5)) != to_jsonl(generate(AgentShape(), 2, 6))
    # A session is drawn from its own index, so a longer file starts with the shorter one.
    assert to_jsonl(generate(AgentShape(), 3, 5)).startswith(to_jsonl(generate(AgentShape(), 2, 5)))


def test_the_script_writes_those_bytes_with_unix_newlines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert mat.main(["-n", "3"]) == 0
    written = tmp_path / "agent-memgpt-n3-s0.jsonl"
    assert hashlib.sha256(written.read_bytes()).hexdigest() == GOLDEN and b"\r" not in written.read_bytes()
    assert mat.main(["-n", "2", "--layout", "append", "--window-tokens", "9000", "--meta-refresh"]) == 0
    assert (tmp_path / "agent-append-n2-s0-windowtokens9000-metarefreshTrue.jsonl").exists()
    with pytest.raises(SystemExit):
        mat.main(["--layout", "sideways"])


def test_every_line_says_how_it_was_made_and_reads_as_explicit_requests(tmp_path):
    lines = generate(AgentShape(window_tokens=7000), 2, 3)
    gen = dict(lines[1]["gen"])
    assert 5.5 < gen.pop("chars_per_token") < 6.5  # about 6 characters per estimated token
    assert "Letta" in gen.pop("evict_frac_source") and "not a figure from the MemGPT paper" in agentgen.EVICT_FRAC_SOURCE
    assert gen == {**{f: v for f, v in vars(AgentShape(window_tokens=7000)).items()}, "seed": 3, "index": 1}
    ss = _sessions(tmp_path)
    assert len(ss) == 6 and all(step.explicit for s in ss for step in s.steps)
    assert all(st.max_tokens == 160 for s in ss for st in s.steps)


def test_no_request_passes_the_window_and_roles_alternate(tmp_path):
    """The bound is counted here, not with the generator's own estimate: 1.25 tokens per word, rounded up per message,
    plus 4 per message, which is what the shape promises."""
    shape = AgentShape(window_tokens=5000)
    longest = 0
    for line in generate(shape, 6, 1):
        for req in line["requests"]:
            msgs = req["messages"]
            words = [len(m["content"].split()) for m in msgs]
            tokens = sum(-(-5 * w // 4) + 4 for w in words)
            assert tokens <= shape.window_tokens
            longest = max(longest, tokens)
            roles = [m["role"] for m in msgs]
            assert roles[0] == "system" and roles[-1] == "user" and "system" not in roles[1:]
            assert all(a != b for a, b in zip(roles[1:], roles[2:]))  # user and assistant alternate
    assert longest > 0.9 * shape.window_tokens  # the window binds: requests reach it before evicting


def test_chains_are_heartbeats_and_follow_their_function_result(tmp_path):
    for s in _sessions(tmp_path):
        for before, step in zip(s.steps, s.steps[1:]):
            if before.chain:
                assert step.tag in ("heartbeat", "core_edit", "retrieval", "evict")
                assert step.messages[-1][1].startswith("[function result]")
        assert s.steps[0].tag == "first"


def test_append_never_rewrites_except_to_evict_and_memgpt_does(tmp_path):
    memgpt, append = _sessions(tmp_path), _sessions(tmp_path, AgentShape(layout="append"), name="append.jsonl")
    assert not any(st.kind == "rewrite_head" for s in append for st in s.steps)
    assert all(st.tag == "evict" for s in append for st in s.steps if st.kind.startswith("rewrite"))
    assert any(st.kind == "rewrite_head" and st.tag == "core_edit" for s in memgpt for st in s.steps)
    wide = AgentShape(window_tokens=100_000, layout="append")  # nothing ever evicted: nothing ever rewritten
    assert {st.kind for s in _sessions(tmp_path, wide, name="wide.jsonl") for st in s.steps} == {"first", "append"}


def test_both_layouts_render_the_same_events(tmp_path):
    """At the default window: the same requests, chained the same way, and the same tags wherever neither request
    was the one that evicted (the appending layout's queue grows faster, so it evicts at other points)."""
    a, b = _sessions(tmp_path, name="a.jsonl"), _sessions(tmp_path, AgentShape(layout="append"), name="b.jsonl")
    assert [s.turns for s in a] == [s.turns for s in b]
    assert [[st.chain for st in s.steps] for s in a] == [[st.chain for st in s.steps] for s in b]
    pairs = [(x.tag, y.tag) for s, t in zip(a, b) for x, y in zip(s.steps, t.steps)]
    assert all(x == y for x, y in pairs if "evict" not in (x, y)) and any(x == "evict" for x, _ in pairs)
    assert a[0].steps[0].messages == b[0].steps[0].messages  # nothing has happened yet


def test_refill_behind_edits_is_the_layouts_doing(tmp_path):
    def avoidable(ss):
        steps = [st for s in ss for st in s.steps]
        return sum(1 - st.lcp - st.new for st in steps) / len(steps)

    wide = 100_000
    appended = avoidable(_sessions(tmp_path, AgentShape(window_tokens=wide, layout="append"), name="x.jsonl"))
    assert appended == pytest.approx(0.0, abs=1e-9)  # nothing behind a change, up to float rounding
    assert avoidable(_sessions(tmp_path, AgentShape(window_tokens=wide), name="y.jsonl")) > 0.05


def test_an_eviction_keeps_only_the_instructions_and_working_context_ahead_of_it(tmp_path):
    first = later = 0
    for s in _sessions(tmp_path, n=8):
        for before, step in zip(s.steps, s.steps[1:]):
            if step.tag == "evict" and before.messages[0] == step.messages[0]:
                head = len(datasets.rendered(step.messages[:1]) + "\x00user\x01")
                shared = round(step.lcp * len(datasets.rendered(step.messages)))
                assert step.messages[1][1].startswith(agentgen.SUMMARY)
                if before.messages[1][1].startswith(agentgen.SUMMARY):  # a summary replaced: its opening survives
                    assert head + len(agentgen.SUMMARY) <= shared < head + len(step.messages[1][1])
                    later += 1
                else:  # the first: what it replaced shares a few characters with it, at most
                    assert head <= shared <= head + 12
                    first += 1
    assert first and later


def test_retrieval_at_the_head_and_a_status_line_rewrite_the_first_message(tmp_path):
    head = _sessions(tmp_path, AgentShape(retrieval_at="head", window_tokens=100_000), name="h.jsonl")
    found = [st for s in head for st in s.steps if st.tag == "retrieval"]
    assert found and all(st.kind == "rewrite_head" for st in found)
    meta = _sessions(tmp_path, AgentShape(meta_refresh=True), name="m.jsonl")
    assert all(st.kind == "rewrite_head" for s in meta for st in s.steps[1:])


def test_bad_shapes_are_refused():
    for bad in ({"layout": "x"}, {"retrieval_at": "middle"}, {"heartbeat_p": 1.5}, {"window_tokens": 0}):
        with pytest.raises(ValueError):
            AgentShape(**bad)
    with pytest.raises(ValueError, match="window_tokens"):
        generate(AgentShape(window_tokens=1500), 1, 0)  # the instructions and memory alone fill it


def test_a_generated_file_is_a_session_workload_named_by_its_hash(tmp_path):
    from rigtune.calibrate.workload import get_workload, workload_from_file

    p = tmp_path / "agent.jsonl"
    p.write_bytes(to_jsonl(generate(AgentShape(), 20, 0)).encode("ascii"))
    wl = workload_from_file(p, get_workload("chat"))
    assert wl.sessions and wl.history == "explicit" and wl.name.startswith("file-agent-")
    assert json.loads(p.read_text(encoding="ascii").splitlines()[0])["session"] == "agent0-0"


def test_a_generated_file_is_sized_at_its_own_characters_per_token_so_8k_contexts_survive(tmp_path):
    """At 4 characters per token a 6000-token window was declared as 9472 tokens, which dropped every 8k context."""
    from rigtune.calibrate.measure import _LiveSession
    from rigtune.calibrate.tokens import TokenCounter
    from rigtune.calibrate.workload import get_workload, workload_from_file

    p = tmp_path / "agent.jsonl"
    p.write_bytes(to_jsonl(generate(AgentShape(), 20, 0)).encode("ascii"))
    wl = workload_from_file(p, get_workload("chat"))
    assert 5.5 < wl.chars_per_token < 6.5 and wl.min_ctx <= 8192 and wl.prefill_tokens >= 6000
    # Without a tokenizer, replay estimates each request at the same ratio, so none is cut for not fitting.
    for s in datasets.read_sessions(p):
        for k in range(s.turns):
            state = _LiveSession(s, step=k)
            assert state.next_prompt_tokens(state.request(), TokenCounter(), wl.chars_per_token) <= wl.prefill_tokens


def test_sessions_of_a_generated_file_share_their_instructions(tmp_path):
    from rigtune.calibrate.measure import _shared_system
    from rigtune.calibrate.workload import get_workload, workload_from_file

    p = tmp_path / "agent.jsonl"
    p.write_bytes(to_jsonl(generate(AgentShape(), 20, 0)).encode("ascii"))
    wl = workload_from_file(p, get_workload("chat")).ensure_prompts()
    shared = _shared_system(wl)
    instructions = generate(AgentShape(), 1, 0)[0]["requests"][0]["messages"][0]["content"].split("\n\nCore memory:")[0]
    assert shared.startswith(instructions) and len(shared) >= len(instructions)
