"""Ask the vLLM in this Python which MTP draft it would build from a model's own checkpoint. Run by rigtune as
`<vllm's python> vllm_mtp_probe.py <model> [<revision>]`; standard library only, besides vLLM.

vLLM decides it in one place: config/speculative.py SpeculativeConfig.hf_config_override rewrites the target's config
into its MTP draft's, with a branch for each family that has one (v0.33.0 from line 711: Qwen3.5, Inkling, DeepSeek,
GLM, Kimi, MiMo, Ernie and more; v0.29.0 from line 643). With method "mtp" and no "model" the draft is the target's own
checkpoint (SpeculativeConfig.__post_init__), so applying the rewrite to the target's config is what a launch does.
Two steps, each with its own answer, printed as vllm_probe.py prints:
  target_arch  architectures[0] of the config vLLM loads (transformers_utils/config.py get_config, line 879 at v0.33.0,
               690 at v0.29.0, with trust_remote_code off, so no code from the repository runs). get_config fills in
               architectures a config lacks (lines 939-949), so this is the architecture vLLM serves, not config.json's.
               Every failure to load turns into a ValueError there ("Invalid repository ID or local directory", lines
               889-923): a bad revision, a gated repository, an unreachable Hub. That is this key's error, never a
               refusal of the head.
  draft_arch   architectures[0] of what the rewrite returns (Inkling's branch returns its text_config, line 1090): the
               draft's class, or the target's own when vLLM has no rule for it. A ValueError raised by the rewrite
               itself (Inkling's for a checkpoint with no MTP depths, line 1091) is vLLM refusing the head.
  layers       n_predict on what the rewrite returns: the MTP layers vLLM takes from the checkpoint, or none.
  libs         the versions of transformers and huggingface_hub here, which load the config.
Whether the class is registered is vllm_probe.py's `archs`.
"""

import copy
import json
import sys

MARK = "RIGTUNE-PROBE "


def emit(key, ask):
    try:
        text = json.dumps({"key": key, "value": ask()})
    except BaseException as exc:  # an ImportError, the rewrite's own ValueError, anything: this answer only
        text = json.dumps({"key": key, "error": (type(exc).__name__ + ": " + str(exc))[:300]})
    sys.stdout.write("\n" + MARK + text + "\n")
    sys.stdout.flush()


def first_arch(config):
    return list(getattr(config, "architectures", None) or [None])[0]


def libs():
    """The libraries that read the config, wherever this Python has them (own_head.CONFIG_LIBS)."""
    import importlib.metadata

    return {name: importlib.metadata.version(name) for name in ("transformers", "huggingface_hub")}


def main():
    model = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    state = {}

    def target_arch():
        # Both imports here, not in draft_arch: loading vllm.config can raise a ValueError of its own (an environment
        # variable outside its choices), which must not read as the rewrite refusing the head.
        from vllm.config.speculative import SpeculativeConfig
        from vllm.transformers_utils.config import get_config

        state["rewrite"] = SpeculativeConfig.hf_config_override
        state["config"] = get_config(model, trust_remote_code=False, revision=revision)
        return first_arch(state["config"])

    def draft_arch():
        state["draft"] = state["rewrite"](copy.deepcopy(state["config"]))
        return first_arch(state["draft"])

    emit("libs", libs)
    emit("target_arch", target_arch)
    if "config" in state:
        emit("draft_arch", draft_arch)
    if "draft" in state:
        emit("layers", lambda: getattr(state["draft"], "n_predict", None))
    emit("done", lambda: True)


if __name__ == "__main__":
    main()
