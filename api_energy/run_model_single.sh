#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_model_single.sh

srun -K \
  --output=mistral_7b_parallel_%j.out \
  --error=mistral_7b_parallel_%j.err \
  --job-name="mistral_7b_parallel" \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task=4 \
  -p H200 \
  --mem=60GB \
  --time=240 \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "chmod +x $(pwd)/install.sh && \
           $(pwd)/install.sh && \
           python $(pwd)/mistral_models_parallelism.py \
             --model mistralai/Mistral-7B-Instruct-v0.3 \
             --csv /netscratch/banwari/api_gpu/synthetic_prompts.csv \
             --hf_token HF_TOKEN \
             --temperature 0.7 \
             --max_samples 100 \
             --max_input_tokens_short 2048 \
             --max_input_tokens_long 8192 \
             --max_new_tokens_short 2048 \
             --max_new_tokens_long 8192 \
             --batch_size 8 \
             --seed 42 \
             --out_csv outputs_mistral_7b_parallelism.csv"