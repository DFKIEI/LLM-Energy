#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_openai_model_api_all.sh

# Define all OpenAI API models
OPENAI_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-4-turbo"
    "gpt-3.5-turbo"
)

# For reasoning models (optional - they're more expensive)
# OPENAI_MODELS+=(
#     "o1-preview"
#     "o1-mini"
# )

# OpenAI API key
OPENAI_API_KEY="your_openai_key"

# CSV file path
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"

# Common parameters
MAX_SAMPLES=100
TEMPERATURE=0.7
SEED=42

# Number of runs per model
NUM_RUNS=5

# Create output directory
OUTPUT_DIR="$(pwd)/results_openai_api"
mkdir -p "$OUTPUT_DIR"

# Log file for all jobs
LOG_FILE="$OUTPUT_DIR/all_openai_api_jobs.log"
echo "Starting OpenAI API job submission at $(date)" > "$LOG_FILE"

# Counter for total jobs
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# Loop through all OpenAI API models
for model in "${OPENAI_MODELS[@]}"; do
    # Extract clean model name for filenames
    model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    for run in $(seq 2 $NUM_RUNS); do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        
        # Create unique job name
        job_name="${model_name}_openai_api_run${run}"
        
        # Output files
        out_file="$OUTPUT_DIR/${job_name}_%j.out"
        err_file="$OUTPUT_DIR/${job_name}_%j.err"
        csv_out="$OUTPUT_DIR/${job_name}_output.csv"
        
        echo "Submitting job $TOTAL_JOBS: $job_name" | tee -a "$LOG_FILE"
        
        # Submit job (no GPU needed for API calls)
        srun -K \
          --output="$out_file" \
          --error="$err_file" \
          --job-name="$job_name" \
          --ntasks=1 \
          --gpus-per-task=0 \
          --cpus-per-task=4 \
          --mem=40GB \
          --time=360 \
          --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
          --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
          --container-workdir="$(pwd)" \
          bash -c "chmod +x $(pwd)/install.sh && \
                   $(pwd)/install.sh && \
                   python $(pwd)/openai_models_api.py \
                     --model $model \
                     --csv $CSV_PATH \
                     --api_key $OPENAI_API_KEY \
                     --temperature $TEMPERATURE \
                     --max_samples $MAX_SAMPLES \
                     --use_batch \
                     --out_csv $csv_out" &
        
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
        
        # Add a small delay to avoid overwhelming the scheduler
        sleep 2
        
        # Optional: limit concurrent submissions
        if [ $((SUBMITTED_JOBS % 5)) -eq 0 ]; then
            echo "Submitted $SUBMITTED_JOBS jobs, pausing for 5 seconds..." | tee -a "$LOG_FILE"
            sleep 5
        fi
    done
done

# Wait for all background jobs to complete
wait

echo "" | tee -a "$LOG_FILE"
echo "All $TOTAL_JOBS OpenAI API jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To monitor jobs, run: squeue -u $USER" | tee -a "$LOG_FILE"
echo "To check results: ls -lh $OUTPUT_DIR/*.csv" | tee -a "$LOG_FILE"