#!/usr/bin/env bash
set -euo pipefail

# Use netscratch for HF cache
export HF_HOME="${HF_HOME:-/netscratch/$USER/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"

python -m pip install --upgrade pip
pip install --upgrade --no-cache-dir -r /home/banwari/llm_energy/api_energy/requirements.txt

# Optional: login if private models (set HF_TOKEN or HUGGINGFACE_HUB_TOKEN)
if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  pip install -q --no-cache-dir "huggingface_hub>=0.23"
  huggingface-cli login --token "${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}" --add-to-git-credential --non-interactive
fi

python - <<'PY'
import transformers, tokenizers, sentencepiece, safetensors
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("safetensors:", safetensors.__version__)
PY