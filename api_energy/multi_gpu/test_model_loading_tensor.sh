#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/test_model_vllm_consistent.sh

export HF_TOKEN="Your_HF_Token"

srun -K \
  --output="test_vllm_consistent_%j.out" \
  --error="test_vllm_consistent_%j.err" \
  -p A100-40GB \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task=8 \
  --mem=80G \
  --time=1-00:00:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
  --container-workdir="$(pwd)" \
  --task-prolog="$(pwd)/install.sh" \
  bash -c "export HF_TOKEN='$HF_TOKEN' HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'; \
    python test_model_loading_tensor.py \
    --model mistralai/Mistral-7B-v0.3 \
    --csv /netscratch/banwari/api_gpu/synthetic_prompts.csv \
    --out_csv ./mistral_vllm_consistent_outputs.csv \
    --tensor_parallel_size 4 \
    --max_samples 100 \
    --temperature 0.7 \
    --repetition_penalty 1.15 \
    --seed 42"