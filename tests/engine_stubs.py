"""Stub vllm and sglang packages for running rigtune's engine probes for real, with no engine installed.

Each stub has, at the module paths the probe imports, only what it asks for, shaped as the release has it (a nested
Literal of methods, a parser built through a subcommand, a registry), and a dist-info so importlib.metadata finds a
version. A test puts the directory on PYTHONPATH and runs the probe script in a subprocess, as rigtune does:

    root = engine_stubs.build("vllm", tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(root))
    facts = run_probe("vllm", sys.executable)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

VLLM = {
    "vllm/__init__.py": "",
    "vllm/utils/__init__.py": "",
    "vllm/utils/argparse_utils.py": (
        "import argparse\n"
        "class FlexibleArgumentParser(argparse.ArgumentParser):\n"
        "    pass\n"),
    "vllm/entrypoints/__init__.py": "",
    "vllm/entrypoints/cli/__init__.py": "",
    "vllm/entrypoints/cli/serve.py": (
        "import argparse\n"
        "class ServeSubcommand:\n"
        "    name = 'serve'\n"
        "    def subparser_init(self, subparsers):\n"
        "        p = subparsers.add_parser(self.name)\n"
        "        p.add_argument('model_tag', nargs='?')\n"
        "        p.add_argument('--max-num-seqs', type=int, default=None)\n"
        "        p.add_argument('--kv-cache-dtype', choices=['auto', 'fp8', 'fp8_e5m2'], default='auto')\n"
        "        p.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=None)\n"
        "        p.add_argument('--attention-backend', default=None)\n"
        "        p.add_argument('--enable-prompt-tokens-details', action='store_true')\n"
        "        p.add_argument('--language-model-only', action=argparse.BooleanOptionalAction, default=False)\n"
        "        p.add_argument('--speculative-config', '-sc', type=str, default=object())\n"
        "        return p\n"),
    "vllm/config/__init__.py": "",
    "vllm/config/speculative.py": (
        "from typing import Literal\n"
        "MTPModelTypes = Literal['deepseek_mtp', 'qwen3_5_mtp', 'mtp']\n"
        "EagleModelTypes = Literal['eagle', 'eagle3', MTPModelTypes]\n"
        "SpeculativeMethod = Literal['ngram', 'draft_model', 'suffix', EagleModelTypes, Literal['ngram_gpu']]\n"),
    "vllm/config/cache.py": (
        "from typing import Literal\n"
        "CacheDType = Literal['auto', 'fp8', 'fp8_e5m2', 'int8_per_token_head']\n"),
    "vllm/distributed/__init__.py": "",
    "vllm/distributed/kv_transfer/__init__.py": "",
    "vllm/distributed/kv_transfer/kv_connector/__init__.py": "",
    "vllm/distributed/kv_transfer/kv_connector/v1/__init__.py": "",
    "vllm/distributed/kv_transfer/kv_connector/v1/base.py": (
        "class KVConnectorBase_V1:\n    pass\n"
        "class SupportsHMA:\n    pass\n"),
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py": (
        "from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, SupportsHMA\n"
        "class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):\n    pass\n"),
    "vllm/v1/__init__.py": "",
    "vllm/v1/kv_offload/__init__.py": "",
    "vllm/v1/kv_offload/base.py": "offload_prompt_only = True\n",
    "vllm/v1/kv_offload/cpu/__init__.py": "",
    "vllm/v1/kv_offload/cpu/shared_offload_region.py": "MMAP = '/dev/shm/vllm_offload_{engine_id}.mmap'\n",
    "vllm/model_executor/__init__.py": "",
    "vllm/model_executor/models/__init__.py": (
        "class _Registry:\n"
        "    models = {'LlamaForCausalLM': None, 'Qwen2ForCausalLM': None, 'GptOssForCausalLM': None}\n"
        "    def get_supported_archs(self):\n"
        "        return self.models.keys()\n"
        "ModelRegistry = _Registry()\n"),
}

SGLANG = {
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/server_args.py": (
        "class ServerArgs:\n"
        "    @staticmethod\n"
        "    def add_cli_args(parser):\n"
        "        parser.add_argument('--model-path', type=str)\n"
        "        parser.add_argument('--enable-cache-report', action='store_true')\n"
        "        parser.add_argument('--enable-hierarchical-cache', action='store_true')\n"
        "        parser.add_argument('--hicache-size', type=int, default=0)\n"
        "        parser.add_argument('--hicache-storage-backend', choices=['file', 'mooncake'], default=None)\n"
        "        parser.add_argument('--tensor-parallel-size', '--tp-size', type=int, default=1)\n"),
    "sglang/srt/speculative/__init__.py": "",
    "sglang/srt/speculative/spec_info.py": (
        "from enum import Enum, auto\n"
        "class SpeculativeAlgorithm(Enum):\n"
        "    DSPARK = auto()\n    EAGLE = auto()\n    EAGLE3 = auto()\n    STANDALONE = auto()\n"
        "    NGRAM = auto()\n    NONE = auto()\n"),
    "sglang/srt/speculative/spec_registry.py": "_RESERVED_ALIASES = frozenset({'NEXTN'})\n",
    "sglang/srt/entrypoints/__init__.py": "",
    "sglang/srt/entrypoints/openai/__init__.py": "",
    "sglang/srt/entrypoints/openai/protocol.py": (
        "class ChatCompletionRequest:\n"
        "    model_fields = {'messages': None, 'model': None, 'return_cached_tokens_details': None}\n"),
    "sglang/srt/environ.py": (
        "class EnvStr:\n"
        "    def __init__(self, default):\n        self.default = default\n"
        "class Envs:\n"
        "    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)\n"
        "    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)\n"
        "envs = Envs()\n"),
    "sglang/srt/arg_groups/__init__.py": "",
    "sglang/srt/arg_groups/overrides.py": "_MAMBA_EXTRA_BUFFER_ARCHS = frozenset({'Qwen3_5ForCausalLM'})\n",
    "sglang/srt/configs/__init__.py": "",
    "sglang/srt/configs/model_config.py": "SWA_SINK_ARCHS = frozenset({'GptOssForCausalLM'})\n",
    "sglang/srt/models/__init__.py": "",
    "sglang/srt/models/registry.py": (
        "class _Registry:\n"
        "    models = {'LlamaForCausalLM': None, 'GptOssForCausalLM': None}\n"
        "    def get_supported_archs(self):\n"
        "        return self.models.keys()\n"
        "ModelRegistry = _Registry()\n"),
}

STUBS = {"vllm": (VLLM, "0.33.0"), "sglang": (SGLANG, "0.5.23")}


def build(engine: str, root: Path, replace: Optional[Dict[str, str]] = None, version: Optional[str] = None) -> Path:
    """Write the `engine` stub under `root` (files in `replace` swapped in, or dropped when None); returns `root`."""
    files, default_version = STUBS[engine]
    files = dict(files)
    for rel, text in (replace or {}).items():
        if text is None:
            files.pop(rel, None)
        else:
            files[rel] = text
    for rel, text in files.items():
        path = root.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    dist = root / f"{engine}-{version or default_version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {engine}\nVersion: {version or default_version}\n",
                                   encoding="utf-8")
    return root
