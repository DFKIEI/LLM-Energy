#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_model_multi_gpu.sh

# Get current directory for container mounts
CURRENT_DIR=$(pwd)

srun -K \
  --output=mistral_7b_local_multigpu%j.out \
  --error=mistral_7b_local_multigpu%j.err \
  --job-name="mistral_7b_local_multigpu" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task=16 \
  --gpu-bind=none \
  -p A100-80GB \
  --mem=300GB \
  --time=120 \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$CURRENT_DIR:$CURRENT_DIR" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-workdir="$CURRENT_DIR" \
  bash -c "
    # Install packages
    echo \"Installing packages...\"
    chmod +x $CURRENT_DIR/install.sh
    $CURRENT_DIR/install.sh
    echo \"Installation complete\"
    
    echo \"Starting torchrun with 4 GPUs...\"
    echo \"Available GPUs: \$(nvidia-smi -L)\"
    
    # Set memory-optimized environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=1
    
    # Run with torchrun for proper distributed setup
    torchrun --nproc_per_node=4 --master_port=29500 \
      $CURRENT_DIR/mistral_models_multi_gpu.py \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --csv /netscratch/banwari/api_gpu/synthetic_prompts.csv \
      --hf_token 'HF_TOKEN' \
      --temperature 0.7 \
      --max_samples 500 \
      --seed 42 \
      --dtype fp16 \
      --out_csv outputs_large_mgpu.csv
  "