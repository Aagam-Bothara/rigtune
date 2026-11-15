"""Ask the SGLang in this Python what it offers. Run by rigtune as `<sglang's python> sglang_probe.py`; standard library
only.

Prints one line per answer, `RIGTUNE-PROBE {"key": ..., "value": ...}` or `{"key": ..., "error": ...}`, as soon as it
has it (see vllm_probe.py for why). The architectures come last: listing them imports every model module.

What is asked, and where each release keeps it (paths under python/sglang/srt/, read at v0.5.19 and v0.5.23):
  version         importlib.metadata.version("sglang").
  libs            the libraries rigtune records beside an engine (rigtune.hardware.ENGINE_LIBS).
  spec_methods    speculative/spec_info.py SpeculativeAlgorithm's members (line 32), and the names
                  speculative/spec_registry.py _RESERVED_ALIASES accepts besides them (NEXTN; line 191 / 183).
  flags           the parser `sglang serve` builds: an argparse.ArgumentParser given to ServerArgs.add_cli_args
                  (server_args.py prepare_server_args; add_cli_args is line 3783 / 351).
  request_fields  entrypoints/openai/protocol.py ChatCompletionRequest's fields (line 823 / 909).
  env_vars        the variables environ.py's Envs declares (class at line 257 / 251).
  arch_sets       arg_groups/overrides.py _MAMBA_EXTRA_BUFFER_ARCHS (line 519 / 403) and configs/model_config.py
                  SWA_SINK_ARCHS (line 54 / 88). Recorded; rigtune's planner does not read them yet.
  archs           models/registry.py ModelRegistry.get_supported_archs() (line 38 / 47). The registry skips a model
                  module that fails to import (import_model_classes logs it and goes on), so this list can miss an
                  architecture whose module needs a GPU at import: rigtune only ever adds from it.
"""

import argparse
import importlib.metadata
import json
import sys

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
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)
    return parser_flags(parser)


def spec_methods():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    names = set(SpeculativeAlgorithm.__members__)
    try:
        from sglang.srt.speculative.spec_registry import _RESERVED_ALIASES

        names |= set(_RESERVED_ALIASES)
    except Exception:
        pass
    return sorted(names)


def request_fields():
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    return sorted(ChatCompletionRequest.model_fields)


def env_vars():
    from sglang.srt.environ import envs

    return sorted(name for name in dir(type(envs)) if name.isupper())


def arch_sets():
    from sglang.srt.arg_groups import overrides
    from sglang.srt.configs import model_config

    return {"mamba_extra_buffer": sorted(overrides._MAMBA_EXTRA_BUFFER_ARCHS),
            "swa_sink": sorted(model_config.SWA_SINK_ARCHS)}


def archs():
    from sglang.srt.models.registry import ModelRegistry

    return sorted(ModelRegistry.get_supported_archs())


def main():
    emit("version", lambda: importlib.metadata.version("sglang"))
    emit("libs", libs)
    emit("spec_methods", spec_methods)
    emit("flags", flags)
    emit("request_fields", request_fields)
    emit("env_vars", env_vars)
    emit("arch_sets", arch_sets)
    emit("archs", archs)
    emit("done", lambda: True)


if __name__ == "__main__":
    main()
