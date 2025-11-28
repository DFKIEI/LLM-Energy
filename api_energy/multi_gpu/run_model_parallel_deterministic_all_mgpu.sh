#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/run_model_parallel_deterministic_all_mgpu.sh

MODELS=(
    "mistralai/Mistral-7B-v0.3"
)

GPUS=(
    "RTXA6000"
)

GPUS_PER_JOB=4
NUM_RUNS=1
HF_TOKEN="HF_TOKEN"
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"

MAX_SAMPLES=100
BATCH_SIZE=8
MAX_INPUT_SHORT=2048
MAX_INPUT_LONG=8192
MAX_OUTPUT_SHORT=2048
MAX_OUTPUT_LONG=8192
TEMPERATURE=0.7
SEED=42

declare -A GPU_TIME
GPU_TIME["RTXA6000"]=300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/results_parallel_tokens_mgpu"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/all_jobs.log"
echo "Starting multi-GPU batch job submission at $(date)" > "$LOG_FILE"
echo "Configuration: ${GPUS_PER_JOB} GPUs per job (single node)" >> "$LOG_FILE"
echo "Script directory: $SCRIPT_DIR" >> "$LOG_FILE"

TOTAL_JOBS=0

for model in "${MODELS[@]}"; do
    model_name=$(echo "$model" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    for gpu in "${GPUS[@]}"; do
        time_limit="${GPU_TIME[$gpu]}"
        
        for run in $(seq 1 $NUM_RUNS); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            job_name="${model_name}_${gpu}_${GPUS_PER_JOB}gpu_run${run}"
            out_file="$OUTPUT_DIR/${job_name}_%j.out"
            err_file="$OUTPUT_DIR/${job_name}_%j.err"
            csv_out="$OUTPUT_DIR/${job_name}_output.csv"
            
            echo "Submitting job $TOTAL_JOBS: $job_name (${GPUS_PER_JOB} GPUs on 1 node)" | tee -a "$LOG_FILE"
            
            srun -K \
              --output="$out_file" \
              --error="$err_file" \
              --job-name="$job_name" \
              --nodes=1 \
              --ntasks=${GPUS_PER_JOB} \
              --ntasks-per-node=${GPUS_PER_JOB} \
              --gpus-per-task=1 \
              --gpu-bind=none \
              --cpus-per-task=10 \
              -p "$gpu" \
              --mem-per-cpu=6G \
              --time="$time_limit" \
              --container-mounts="/netscratch/$USER:/netscratch/$USER,$SCRIPT_DIR:$SCRIPT_DIR" \
              --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
              --container-workdir="$SCRIPT_DIR" \
              --task-prolog="$SCRIPT_DIR/install.sh" \
              bash <<EOFBASH
set -euo pipefail
export HF_TOKEN='$HF_TOKEN'
export HUGGINGFACE_HUB_TOKEN='$HF_TOKEN'

echo "=== Task \${SLURM_PROCID} on \$(hostname) starting at \$(date) ==="
echo "Environment:"
echo "  SLURM_JOB_ID=\${SLURM_JOB_ID}"
echo "  SLURM_PROCID=\${SLURM_PROCID}"
echo "  SLURM_LOCALID=\${SLURM_LOCALID}"
echo "  SLURM_NTASKS=\${SLURM_NTASKS}"
echo "  CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"

python $SCRIPT_DIR/mistral_models_parallelism_consistent_mgpu.py \
  --model $model \
  --csv $CSV_PATH \
  --hf_token $HF_TOKEN \
  --temperature $TEMPERATURE \
  --max_samples $MAX_SAMPLES \
  --max_input_tokens_short $MAX_INPUT_SHORT \
  --max_input_tokens_long $MAX_INPUT_LONG \
  --max_new_tokens_short $MAX_OUTPUT_SHORT \
  --max_new_tokens_long $MAX_OUTPUT_LONG \
  --batch_size $BATCH_SIZE \
  --seed $((SEED + run)) \
  --out_csv $csv_out

echo "=== Task \${SLURM_PROCID} completed at \$(date) ==="
EOFBASH
            
            sleep 2
        done
    done
done

wait

echo "" | tee -a "$LOG_FILE"
echo "All $TOTAL_JOBS jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Monitor: squeue -u $USER" | tee -a "$LOG_FILE"