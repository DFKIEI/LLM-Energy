#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/run_llama_model_api_all.sh

# Define all API models (corresponding to open-source models)
API_MODELS=(
    "llama-3.1-8b-instant"
)

# Groq API key
API_KEY="Groq_API_Key"

# CSV file path
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"

# Common parameters
MAX_SAMPLES=100
MAX_INPUT_SHORT=2048
MAX_INPUT_LONG=8192
MAX_OUTPUT_SHORT=2048
MAX_OUTPUT_LONG=8192
TEMPERATURE=0.7
SEED=42

# Number of runs per model
NUM_RUNS=30

# Delay between job starts (in seconds)
DELAY_BETWEEN_JOBS=$((60 * 60))  # 1 hour = 3600 seconds

# Create output directory
OUTPUT_DIR="$(pwd)/results_llama_api"
mkdir -p "$OUTPUT_DIR"

# Log file for all jobs
LOG_FILE="$OUTPUT_DIR/all_llama_api_jobs.log"
echo "Starting Llama API job submission at $(date)" > "$LOG_FILE"
echo "Models: ${API_MODELS[@]}" >> "$LOG_FILE"
echo "Runs per model: $NUM_RUNS" >> "$LOG_FILE"
echo "Delay between jobs: $((DELAY_BETWEEN_JOBS / 60)) minutes" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Counter for total jobs
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# Track when each job should start
CURRENT_DELAY=0

# Loop through all API models
for model in "${API_MODELS[@]}"; do
    # Extract clean model name for filenames
    model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    for run in $(seq 21 $NUM_RUNS); do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        
        # Create unique job name
        job_name="${model_name}_api_run${run}"
        
        # Output files
        out_file="$OUTPUT_DIR/${job_name}_%j.out"
        err_file="$OUTPUT_DIR/${job_name}_%j.err"
        csv_out="$OUTPUT_DIR/${job_name}_output.csv"
        
        # Calculate start time for this job
        hours=$((CURRENT_DELAY / 3600))
        minutes=$(((CURRENT_DELAY % 3600) / 60))
        
        echo "Submitting job $TOTAL_JOBS: $job_name (will start in ${hours}h ${minutes}m)" | tee -a "$LOG_FILE"
        echo "  Model: $model" | tee -a "$LOG_FILE"
        echo "  Scheduled start: $(date -d "+${CURRENT_DELAY} seconds" '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        
        # Submit job with --begin flag to delay start
        srun -K \
          --output="$out_file" \
          --error="$err_file" \
          --job-name="$job_name" \
          --ntasks=1 \
          --gpus-per-task=0 \
          --cpus-per-task=4 \
          --mem=40GB \
          --time=180 \
          --begin=now+${CURRENT_DELAY}seconds \
          --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
          --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
          --container-workdir="$(pwd)" \
          bash -c "chmod +x $(pwd)/install.sh && \
                   $(pwd)/install.sh && \
                   python $(pwd)/llama_models_api.py \
                     --model $model \
                     --csv $CSV_PATH \
                     --api_key $API_KEY \
                     --temperature $TEMPERATURE \
                     --max_samples $MAX_SAMPLES \
                     --max_input_tokens_short $MAX_INPUT_SHORT \
                     --max_input_tokens_long $MAX_INPUT_LONG \
                     --max_new_tokens_short $MAX_OUTPUT_SHORT \
                     --max_new_tokens_long $MAX_OUTPUT_LONG \
                     --seed $((SEED + run)) \
                     --out_csv $csv_out" &
        
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
        
        # Increment delay for next job
        CURRENT_DELAY=$((CURRENT_DELAY + DELAY_BETWEEN_JOBS))
        
        # Small delay to avoid overwhelming the scheduler
        sleep 2
    done
done

# Wait for all background jobs to complete
wait

echo "" | tee -a "$LOG_FILE"
echo "All $TOTAL_JOBS Llama API jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Job Schedule:" | tee -a "$LOG_FILE"
echo "  First job starts: immediately" | tee -a "$LOG_FILE"
echo "  Last job starts: in $((CURRENT_DELAY / 3600)) hours" | tee -a "$LOG_FILE"
echo "  Total span: ~$(((CURRENT_DELAY + 180*60) / 3600)) hours (including 3h max runtime)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To monitor jobs, run: squeue -u $USER | grep llama" | tee -a "$LOG_FILE"
echo "To check scheduled jobs: squeue -u $USER --start" | tee -a "$LOG_FILE"
echo "To check results: ls -lh $OUTPUT_DIR/*.csv" | tee -a "$LOG_FILE"