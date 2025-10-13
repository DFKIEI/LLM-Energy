srun -K \
  --output=mistral_7b_api200_ind%j.out \
  --error=mistral_7b_api200_ind%j.err \
  --job-name="mistral_7b_api200_ind" \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task=4 \
  -p RTXA6000 \
  --mem=40GB \
  --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "chmod +x /home/banwari/llm_energy/api_energy/install.sh && \
           /home/banwari/llm_energy/api_energy/install.sh && \
           python /home/banwari/llm_energy/api_energy/mistral_models_api.py \
             --model open-mistral-7b \
             --csv /netscratch/banwari/api_gpu/synthetic_prompts.csv \
             --api_key 9ZxEWrXV1qF65bILI50KyjbrhNfog1JX \
             --temperature 0.7 \
             --out_csv outputs_larger_api200_ind.csv \
             --max_samples 200"