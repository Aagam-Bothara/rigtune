"""Ask the vLLM in this Python what it offers. Run by rigtune as `<vllm's python> vllm_probe.py`; standard library only.

Prints one line per answer, `RIGTUNE-PROBE {"key": ..., "value": ...}` or `{"key": ..., "error": ...}`, as soon as it
has it, so an answer survives a later one that hangs (rigtune stops the probe after a timeout) and the engine's own log
lines, which vLLM writes to stdout too, are told apart. Each answer is asked separately: one that fails costs only
itself. The architectures come last: they are the slowest to list.

What is asked, and where each release keeps it (read at v0.29.0 and v0.33.0):
  version       importlib.metadata.version("vllm").
  libs          the libraries rigtune records beside an engine (rigtune.hardware.ENGINE_LIBS).
  spec_methods  config/speculative.py SpeculativeMethod, a Literal of Literals (line 71 / 74), flattened.
  kv_dtypes     config/cache.py CacheDType (line 39 / 44).
  kernel        the graph and compile settings --compilation-config takes, which the parser cannot show:
                config/compilation.py CompilationMode (line 37 both), CUDAGraphMode (53 both), PassConfig
                (107 / 114) and CompilationConfig (398 / 383); config/vllm.py OptimizationLevel (110 / 132);
                v1/attention/backends/registry.py AttentionBackendEnum (34 both). Names only: what a card supports
                cannot be asked with no GPU visible.
  flags         the parser `vllm serve` builds: entrypoints/cli/main.py makes a FlexibleArgumentParser and calls
                ServeSubcommand().subparser_init on its subparsers (entrypoints/cli/serve.py, line 158 / 161).
  features      the KV offloading tier: OffloadingConnector and whether it is SupportsHMA
                (distributed/kv_transfer/kv_connector/v1/offloading_connector.py, line 49 / 53), and the
                offload_prompt_only setting and the /dev/shm region, in the source of v1/kv_offload/base.py and
                v1/kv_offload/cpu/shared_offload_region.py as this Python imports them.
  archs         model_executor/models ModelRegistry.get_supported_archs() (registry.py, line 1095 / 1173).
"""

import importlib
import importlib.metadata
import inspect
import json
import sys
import typing

MARK = "RIGTUNE-PROBE "
# rigtune.hardware.ENGINE_LIBS: name shown -> distribution name.
LIBS = {"torch": "torch", "transformers": "transformers", "flashinfer": "flashinfer-python",
        "tokenizers": "tokenizers", "arctic-inference": "arctic_inference"}


def emit(key, ask):
    try:
        line = {"key": key, "value": ask()}
        text = json.dumps(line)
    except BaseException as exc:  # SystemExit from a parser, an ImportError, anything: this answer only
        text = json.dumps({"key": key, "error": (type(exc).__name__ + ": " + str(exc))[:300]})
    sys.stdout.write("\n" + MARK + text + "\n")
    sys.stdout.flush()


def libs():
    out = {}
    for name, dist in LIBS.items():
        try:
            out[name] = importlib.metadata.version(dist)
        except Exception:
            pass
    return out


def literal_values(tp):
    """Every value of a Literal, through nested Literals and Unions."""
    out = []
    for arg in typing.get_args(tp):
        if isinstance(arg, str):
            out.append(arg)
        else:
            out += literal_values(arg)
    return out


def plain(value):
    """A default as JSON can hold it: itself when it can, else its text."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def parser_flags(parser):
    """Every option string the parser takes -> its choices and default."""
    out = {}
    for action in parser._actions:
        choices = None if action.choices is None else [plain(c) for c in action.choices]
        for option in action.option_strings:
            out[option] = {"choices": choices, "default": plain(action.default)}
    return out


def flags():
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    subparsers = FlexibleArgumentParser().add_subparsers()
    return parser_flags(ServeSubcommand().subparser_init(subparsers))


def spec_methods():
    from vllm.config.speculative import SpeculativeMethod

    return sorted(set(literal_values(SpeculativeMethod)))


def spec_fields():
    from dataclasses import fields, is_dataclass
    from vllm.config.speculative import SpeculativeConfig

    if is_dataclass(SpeculativeConfig):
        return sorted(f.name for f in fields(SpeculativeConfig))
    if hasattr(SpeculativeConfig, "model_fields"):
        return sorted(SpeculativeConfig.model_fields)
    raise TypeError("SpeculativeConfig fields could not be read")


def kv_dtypes():
    from vllm.config.cache import CacheDType

    return sorted(set(literal_values(CacheDType)))


def field_names(cls):
    from dataclasses import fields, is_dataclass

    if is_dataclass(cls):
        return sorted(f.name for f in fields(cls))
    if hasattr(cls, "model_fields"):
        return sorted(cls.model_fields)
    raise TypeError(cls.__name__ + " fields could not be read")


def kernel():
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode, PassConfig
    from vllm.config.vllm import OptimizationLevel
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    return {"cudagraph_modes": list(CUDAGraphMode.__members__),
            "compilation_modes": list(CompilationMode.__members__),
            "compilation_fields": field_names(CompilationConfig),
            "pass_fields": field_names(PassConfig),
            "attention_backends": list(AttentionBackendEnum.__members__),
            "opt_levels": list(OptimizationLevel.__members__)}


def module_source(name):
    """A module's source as this Python imports it. A module that fails to import fails the whole answer, which then
    says why: read as empty, it would answer False for what it holds, and that False would beat the source check."""
    return inspect.getsource(importlib.import_module(name))


def features():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector

    return {"offloading_connector": True,
            "supports_hma": issubclass(OffloadingConnector, SupportsHMA),
            "offload_prompt_only": "offload_prompt_only" in module_source("vllm.v1.kv_offload.base"),
            "shm_region": "/dev/shm" in module_source("vllm.v1.kv_offload.cpu.shared_offload_region")}


def archs():
    from vllm.model_executor.models import ModelRegistry

    return sorted(ModelRegistry.get_supported_archs())


def main():
    emit("version", lambda: importlib.metadata.version("vllm"))
    emit("libs", libs)
    emit("spec_methods", spec_methods)
    emit("spec_fields", spec_fields)
    emit("kv_dtypes", kv_dtypes)
    emit("kernel", kernel)
    emit("flags", flags)
    emit("features", features)
    emit("archs", archs)
    emit("done", lambda: True)


if __name__ == "__main__":
    main()
