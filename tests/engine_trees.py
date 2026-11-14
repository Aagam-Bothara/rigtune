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

VERSIONS = ("vllm-0.29.0", "vllm-0.33.0")

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

TREES: Dict[str, Dict[str, str]] = {
    "vllm-0.29.0": dict(_VLLM_COMMON, **{
        # The server's own arguments: entrypoints/launchers/cli_args.py, BaseFrontendArgs, line 132.
        "entrypoints/launchers/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
    }),
    "vllm-0.33.0": dict(_VLLM_COMMON, **{
        # Moved to entrypoints/common/cli_args.py, BaseFrontendArgs, line 156; launchers/cli_args.py is gone.
        "entrypoints/common/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
    }),
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
    import rigtune.backends.vllm as vl

    for fn in (vl._vllm_package, vl._serve_cli, vl._attention_backend_flag, vl._prompt_tokens_details_flag,
               vl._language_model_only_flag, vl._offload_features):
        fn.cache_clear()


def _version_tuple(version: str):
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


@contextlib.contextmanager
def installed(engine: str, version: str, root: Path, flashinfer: bool = True,
              arctic_inference: bool = True) -> Iterator[Path]:
    """rigtune as it sees `engine` `version` installed from the tree under `root` (built if missing), with the
    optional packages rigtune looks for present or not. Yields the package directory."""
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
            stack.enter_context(mock.patch.object(vl, "_serve_cli", lambda: "vllm"))
            stack.enter_context(mock.patch.object(vl, "_vllm_package", lambda: pkg))
            stack.enter_context(mock.patch.object(vl, "_vllm_version", lambda: _version_tuple(version)))
            stack.enter_context(mock.patch.object(speculative, "_vllm_version", lambda: version))
            yield pkg
    finally:
        _clear()  # every answer cached while the tree stood in for the engine
