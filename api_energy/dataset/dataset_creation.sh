srun -K \
  --output=dataset_creation%j.out \
  --error=dataset_creation%j.err \
  --job-name="dataset_creation" \
  --ntasks=1 \
  --gpus-per-task=0 \
  --cpus-per-task=4 \
  --mem=60GB \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "chmod +x /home/banwari/llm_energy/api_energy/install.sh && \
           /home/banwari/llm_energy/api_energy/install.sh && \
           python /home/banwari/llm_energy/api_energy/create_dataset.py \
           --num_samples 100 \
           --output_csv /netscratch/banwari/api_gpu/synthetic_prompts_larger.csv \
           --validate"