#!/usr/bin/env python
"""Write a synthetic agent trace: explicit-request sessions shaped like a MemGPT agent (rigtune/calibrate/agentgen.py).

    python benchmarks/make_agent_trace.py -n 200                         # agent-memgpt-n200-s0.jsonl
    python benchmarks/make_agent_trace.py -n 200 --layout append         # the same events, every edit appended
    rigtune bench <model> --workload chat --workload-file agent-memgpt-n200-s0.jsonl

A model of MemGPT's published design (Packer et al. 2023), not a recording of any framework. The file is
deterministic: the same parameters and seed give the same bytes on every platform, so its hash, which names the
profile, says what was measured. The parameters are in each line's "gen" key and, where not the defaults, in the
file name. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from rigtune.calibrate.agentgen import AgentShape, default_name, generate, to_jsonl


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-n", "--sessions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: a name that holds the parameters")
    default = AgentShape()
    for f in fields(AgentShape):
        value = getattr(default, f.name)
        if isinstance(value, bool):
            ap.add_argument(_flag(f.name), action="store_true", default=value)
        else:
            ap.add_argument(_flag(f.name), type=type(value), default=value)
    args = ap.parse_args(argv)
    try:
        shape = AgentShape(**{f.name: getattr(args, f.name) for f in fields(AgentShape)})
    except ValueError as exc:
        ap.error(str(exc))
    out = args.out or Path(default_name(shape, args.sessions, args.seed))
    lines = generate(shape, args.sessions, args.seed)
    with out.open("w", encoding="ascii", newline="\n") as fh:
        fh.write(to_jsonl(lines))
    print(f"wrote {len(lines)} agent sessions, {sum(len(x['requests']) for x in lines)} requests, to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
