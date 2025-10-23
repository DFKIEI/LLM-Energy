#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/rerun_failed_jobs.sh

# HuggingFace token
HF_TOKEN="HF_TOKEN"

# CSV file path
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"

# Common parameters
MAX_SAMPLES=100
BATCH_SIZE=8
MAX_INPUT_SHORT=2048
MAX_INPUT_LONG=8192
MAX_OUTPUT_SHORT=2048
MAX_OUTPUT_LONG=8192
TEMPERATURE=0.7
SEED=42

# Output directory
OUTPUT_DIR="$(pwd)/results_parallel_tokens_deterministic"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/rerun_failed_jobs.log"
echo "Starting rerun of failed jobs at $(date)" > "$LOG_FILE"

# Define failed jobs: model, gpu, run_number
declare -a FAILED_JOBS=(
    "mistralai/Mistral-Nemo-Instruct-2407:H100:5"
    "mistralai/Mistral-Nemo-Instruct-2407:H100:4"
    "mistralai/Mistral-Nemo-Base-2407:H200:2"
    "mistralai/Mistral-7B-Instruct-v0.3:H200:5"
)

# Memory requirements per GPU type
declare -A GPU_MEM
GPU_MEM["H100"]=80
GPU_MEM["H200"]=80

# Increased time limits for these jobs (in minutes)
declare -A GPU_TIME
GPU_TIME["H100"]=360  # Increased from 180 to 360 (6 hours)
GPU_TIME["H200"]=360  # Increased from 180 to 360 (6 hours)

SUBMITTED_JOBS=0

# Loop through failed jobs
for job_spec in "${FAILED_JOBS[@]}"; do
    # Parse job specification
    IFS=':' read -r model gpu run <<< "$job_spec"
    
    # Extract clean model name for filenames
    model_name=$(echo "$model" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    # Get memory and time for this GPU
    mem="${GPU_MEM[$gpu]}GB"
    time_limit="${GPU_TIME[$gpu]}"
    
    # Create unique job name
    job_name="${model_name}_${gpu}_run${run}"
    
    # Output files
    out_file="$OUTPUT_DIR/${job_name}_%j.out"
    err_file="$OUTPUT_DIR/${job_name}_%j.err"
    csv_out="$OUTPUT_DIR/${model_name}_${gpu}_run${run}_output.csv"
    
    echo "Submitting retry job: $job_name (timeout: ${time_limit}min)" | tee -a "$LOG_FILE"
    echo "  Model: $model" | tee -a "$LOG_FILE"
    echo "  GPU: $gpu" | tee -a "$LOG_FILE"
    echo "  Run: $run" | tee -a "$LOG_FILE"
    echo "  Memory: $mem" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Submit job
    srun -K \
      --output="$out_file" \
      --error="$err_file" \
      --job-name="$job_name" \
      --ntasks=1 \
      --gpus-per-task=1 \
      --cpus-per-task=4 \
      -p "$gpu" \
      --mem="$mem" \
      --time="$time_limit" \
      --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
      --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
      --container-workdir="$(pwd)" \
      bash -c "chmod +x $(pwd)/install.sh && \
               $(pwd)/install.sh && \
               python $(pwd)/mistral_models_parallelism_consistent.py \
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
                 --out_csv $csv_out" &
    
    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
    
    # Add a delay between submissions
    sleep 5
done

# Wait for all background jobs to complete
wait

echo "" | tee -a "$LOG_FILE"
echo "All $SUBMITTED_JOBS retry jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To monitor jobs, run: squeue -u $USER | grep retry" | tee -a "$LOG_FILE"
echo "To check results: ls -lh $OUTPUT_DIR/*retry*.csv" | tee -a "$LOG_FILE"