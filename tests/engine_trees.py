"""Fake engine source trees, shaped like each release rigtune is checked against.

rigtune tells what an installed engine offers by reading its source text (importing vLLM or SGLang costs seconds,
and SGLang often lives in another environment). These trees hold, at the paths each release has them, only the
lines those checks look for, copied from the release tag (file and line in each comment), so a test can point
rigtune at "vLLM 0.33.0" with no engine installed. Paths are relative to the vllm or sglang package.

    with installed("vllm", "0.33.0", tmp_path):
        ...  # rigtune's feature checks now read that tree
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from typing import Dict, Iterator
from unittest import mock

VERSIONS = ("vllm-0.29.0", "vllm-0.33.0", "sglang-0.5.19", "sglang-0.5.23")
SGLANG_PYTHON = "/opt/sglang/bin/python"  # what $SGLANG_PYTHON points at, in launch commands built from a tree

_VLLM_COMMON = {
    # engine/arg_utils.py: line 38 (both tags) imports AttentionConfig; EngineArgs.language_model_only is line 613
    # at v0.29.0 and 651 at v0.33.0.
    "engine/arg_utils.py": "    AttentionConfig,\n"
                           "    language_model_only: bool = MultiModalConfig.language_model_only\n",
    # distributed/kv_transfer/kv_connector/v1/offloading_connector.py: lines 49 / 53.
    "distributed/kv_transfer/kv_connector/v1/offloading_connector.py":
        "class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):\n",
    # v1/kv_offload/base.py: lines 598-599 / 729-730.
    "v1/kv_offload/base.py": "        self.offload_prompt_only: bool = bool(\n"
                             "            self.extra_config.get(\"offload_prompt_only\", True)\n",
    # v1/kv_offload/cpu/shared_offload_region.py: lines 98 / 128.
    "v1/kv_offload/cpu/shared_offload_region.py":
        "        self.mmap_path = f\"/dev/shm/vllm_offload_{engine_id}.mmap\"\n",
}

_SGLANG_ENVIRON = ("    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)\n"
                   "    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)\n")
_SGLANG_PROTOCOL = "    return_cached_tokens_details: bool = False\n"

TREES: Dict[str, Dict[str, str]] = {
    "vllm-0.29.0": dict(_VLLM_COMMON, **{
        # The server's own arguments: entrypoints/launchers/cli_args.py, BaseFrontendArgs, line 132.
        "entrypoints/launchers/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
    }),
    "vllm-0.33.0": dict(_VLLM_COMMON, **{
        # Moved to entrypoints/common/cli_args.py, BaseFrontendArgs, line 156; launchers/cli_args.py is gone.
        "entrypoints/common/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
    }),
    "sglang-0.5.19": {
        # Every argument is defined in srt/server_args.py (lines 1353, 2072, 2324, 2719, 2735, 2770).
        "srt/server_args.py": (
            "    enable_cache_report: A[\n"
            "        \"Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK. Or any "
            "name registered via `SpeculativeAlgorithm.register`.\",\n"
            "    speculative_ngram_max_bfs_breadth: A[\n"
            "    enable_hierarchical_cache: A[bool, \"Enable hierarchical cache\", NS(\"memory\")] = (\n"
            "    hicache_size: A[\n"
            "    hicache_storage_backend: A[\n"),
        "srt/arg_groups/speculative_hook.py": "        cfg.speculative_algorithm == \"STANDALONE\"\n",  # line 645
        "srt/environ.py": _SGLANG_ENVIRON,  # lines 717, 719
        "srt/entrypoints/openai/protocol.py": _SGLANG_PROTOCOL,  # line 355
    },
    "sglang-0.5.23": {
        # Arguments moved to srt/arg_groups/fields/*.py; server_args.py keeps add_cli_args (line 351). rigtune reads
        # server_args.py and arg_groups/*.py, not fields/, and still finds every name through
        # arg_groups/field_order.py's POSITIONAL_FIELD_ORDER (lines 146, 300, 359, 362, 366) and the hooks.
        "srt/server_args.py": "    def add_cli_args(parser: argparse.ArgumentParser):\n",
        "srt/arg_groups/field_order.py": (
            "    \"enable_cache_report\",\n"
            "    \"speculative_ngram_max_bfs_breadth\",\n"
            "    \"enable_hierarchical_cache\",\n"
            "    \"hicache_size\",\n"
            "    \"hicache_storage_backend\",\n"),
        # line 1078
        "srt/arg_groups/speculative_hook.py":
            "    if cfg.speculative_algorithm == \"STANDALONE\" and attn_dp_enabled_of(\n",
        # line 480
        "srt/arg_groups/attention_hook.py": "            if _algo not in (\"DSPARK\", \"DFLASH\") or verify not in (\n",
        "srt/arg_groups/fields/serving.py": "    enable_cache_report: A[\n",  # line 196, not read by rigtune
        "srt/environ.py": _SGLANG_ENVIRON,  # lines 828, 830
        "srt/entrypoints/openai/protocol.py": _SGLANG_PROTOCOL,  # line 392
    },
}


def build(engine: str, version: str, root: Path) -> Path:
    """Write the tree for `engine` `version` under `root`; returns the package directory."""
    pkg = root / engine
    for rel, text in TREES[f"{engine}-{version}"].items():
        path = pkg.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return pkg


def _clear() -> None:
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl

    for fn in (vl._vllm_package, vl._serve_cli, vl._attention_backend_flag, vl._prompt_tokens_details_flag,
               vl._language_model_only_flag, vl._offload_features, sg.sglang_spec_methods, sg._cache_report_flag,
               sg._hicache_features, sg._tier_details_flag):
        fn.cache_clear()


def _version_tuple(version: str):
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


@contextlib.contextmanager
def installed(engine: str, version: str, root: Path, flashinfer: bool = True,
              arctic_inference: bool = True) -> Iterator[Path]:
    """rigtune as it sees `engine` `version` installed from the tree under `root` (built if missing), with the
    optional packages rigtune looks for present or not. Yields the package directory."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    import rigtune.speculative as speculative

    pkg = root / engine
    if not pkg.exists():
        build(engine, version, root)
    present = {"flashinfer": flashinfer, "arctic_inference": arctic_inference}
    real_find_spec = importlib.util.find_spec

    class _Spec:
        submodule_search_locations = [str(pkg)]

    def find_spec(name, *args, **kwargs):
        if name == engine:
            return _Spec()
        if name in present:
            return object() if present[name] else None
        return real_find_spec(name, *args, **kwargs)

    _clear()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(importlib.util, "find_spec", find_spec))
            if engine == "vllm":
                stack.enter_context(mock.patch.object(vl, "_serve_cli", lambda: "vllm"))
                stack.enter_context(mock.patch.object(vl, "_vllm_package", lambda: pkg))
                stack.enter_context(mock.patch.object(vl, "_vllm_version", lambda: _version_tuple(version)))
                stack.enter_context(mock.patch.object(speculative, "_vllm_version", lambda: version))
            else:
                stack.enter_context(mock.patch.object(sg, "sglang_python", lambda: SGLANG_PYTHON))
                stack.enter_context(mock.patch.object(sg, "_sglang_package_dir", lambda python: pkg))
            yield pkg
    finally:
        _clear()  # every answer cached while the tree stood in for the engine
