"""Ask the SGLang in this Python which MTP draft it would build from a model's own checkpoint. Run by rigtune as
`<sglang's python> sglang_mtp_probe.py <model> [<revision>]`; standard library only, besides SGLang.

SGLang decides it where it builds the draft's model config: srt/configs/model_config.py ModelConfig._config_draft_model
(v0.5.23 line 1010, called from __init__ at line 720; v0.5.19 line 651) renames architectures[0] to the draft's class,
with a branch for each family that has one (DeepSeek, GLM, MiMo, Inkling, Qwen3-Next, Qwen3.5 and more). With NEXTN,
which the launch resolves to EAGLE before it builds the draft's config (speculative_hook.py, lines 109-116 at v0.5.23),
and the target's own path, which rigtune passes, the draft is the target's own checkpoint. The probe builds that config
through a subclass whose _config_draft_model records the architecture before and after SGLang's own, so the answer
stands even when the rest of __init__ (quantization checks, shape derivation, generation config) raises afterwards.
Printed as vllm_probe.py prints:
  target_arch  architectures[0] when the rename starts: after the renames __init__ makes first (Qwen3_5Model and
               others, lines 627-647), so the architecture SGLang serves
  draft_arch   architectures[0] after it: the draft's class, or the target's own when SGLang has no rule for it
  layers       hf_text_config.num_nextn_predict_layers after it, the field SGLang reads (line 1602). Many rules force it
               to 1 whatever the checkpoint holds, so it says what SGLang builds, not what the checkpoint has.
  banded       whether the draft's text config has mtp_local_layer_ids (Inkling's banded head), which SGLang runs only
               on its multi-layer EAGLE worker (--enable-multi-layer-eagle).
  libs         the versions of transformers and huggingface_hub here, which load the config.
The rename does not read the weights: a rule for the architecture is all a yes says. ModelConfig is built with
trust_remote_code off (its own default is on), so no code from the repository runs. Whether the class is registered is
sglang_probe.py's `archs`.
"""

import json
import sys

MARK = "RIGTUNE-PROBE "


def say(key, value=None, error=None):
    line = {"key": key, "error": error} if error is not None else {"key": key, "value": value}
    sys.stdout.write("\n" + MARK + json.dumps(line) + "\n")
    sys.stdout.flush()


def first_arch(config):
    return list(getattr(config, "architectures", None) or [None])[0]


def main():
    model = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    seen = {}
    try:
        import importlib.metadata

        say("libs", {name: importlib.metadata.version(name) for name in ("transformers", "huggingface_hub")})
    except BaseException as exc:
        say("libs", error=(type(exc).__name__ + ": " + str(exc))[:300])
    try:
        from sglang.srt.configs.model_config import ModelConfig

        class Probe(ModelConfig):
            def _config_draft_model(self):
                seen["target_arch"] = first_arch(self.hf_config)
                super()._config_draft_model()
                seen["draft_arch"] = first_arch(self.hf_config)
                seen["layers"] = getattr(self.hf_text_config, "num_nextn_predict_layers", None)
                # Inkling's banded head (mtp_config.local_layer_ids) only runs on the multi-layer EAGLE worker
                # (models/inkling.py InklingForConditionalGenerationMTP.__init__, lines 1792-1797 at v0.5.23).
                seen["banded"] = bool(getattr(self.hf_text_config, "mtp_local_layer_ids", None))

        Probe(model, trust_remote_code=False, revision=revision, is_draft_model=True, speculative_algorithm="EAGLE")
        error = "not reached"
    except BaseException as exc:  # an ImportError, a config that cannot be read, a check after the rename: say why
        error = (type(exc).__name__ + ": " + str(exc))[:300]
    for key in ("target_arch", "draft_arch", "layers", "banded"):
        if key in seen:
            say(key, seen[key])
        elif key not in ("layers", "banded"):
            say(key, error=error)
            break
    say("done", True)


if __name__ == "__main__":
    main()
