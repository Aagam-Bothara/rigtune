#!/bin/bash
# Pod setup for the Inkling-2-Mini session: vLLM's nightly wheel in a venv, rigtune (this tree) with its reasoning
# extra, and the model downloaded once. Logs to /root/setup.log; ends with SETUP DONE or SETUP FAILED.
set -o pipefail
exec > >(tee -a /root/setup.log) 2>&1
fail() { echo "SETUP FAILED: $*"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || fail "no GPU"
python3 -V
pip install -q -U uv --break-system-packages 2>/dev/null || pip install -q -U uv || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH) || fail "uv"
export PATH=$HOME/.local/bin:$PATH
uv venv /root/v --python 3.12 -q || uv venv /root/v -q || fail "venv"
. /root/v/bin/activate
# The recipe: "Use the vllm/vllm-openai:nightly image or a nightly wheel until it is in a vLLM release."
uv pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --torch-backend=auto \
  || fail "vllm nightly"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" \
  || fail "vllm import"
python -c "import vllm.models.inkling2" && echo "inkling2 present" || fail "this vLLM has no inkling2"
grep -c Inkling2DSparkConfig "$(python -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')/config/speculative.py"
uv pip install -q "/root/rigtune.tar.gz[reason]" 2>/dev/null || (mkdir -p /root/rigtune && tar xzf /root/rigtune.tar.gz -C /root/rigtune && uv pip install -q "/root/rigtune[reason]") || fail "rigtune"
python -c "import rigtune; print('rigtune', rigtune.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install -q hf_transfer huggingface_hub
python -c "from huggingface_hub import snapshot_download; p = snapshot_download('thinkingmachines/Inkling-2-Mini'); print('model at', p)" \
  || fail "download"
du -sh /root/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini
python -c "from rigtune.reasoning.problems import load_math; print(len(load_math((4, 5), 'dev')), 'dev problems cached')"   || fail "MATH"
echo "SETUP DONE"
