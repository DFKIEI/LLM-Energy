#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_model_single.sh

srun -K \
  --output=collate_results%j.out \
  --error=collate_results%j.err \
  --job-name="collate_results" \
  --ntasks=1 \
  --gpus-per-task=0 \
  --cpus-per-task=2 \
  --mem=40GB \
  --time=240 \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "chmod +x $(pwd)/install.sh && \
           $(pwd)/install.sh && \
           python $(pwd)/collate_results.py"