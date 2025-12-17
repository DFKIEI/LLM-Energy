#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/run_model_api_all.sh

# Define all API models (corresponding to open-source models)
API_MODELS=(
    "open-mistral-7b"
    "open-mistral-nemo"
)

#API_MODELS=(
#    "open-mistral-7b"
#    "open-mistral-nemo"
#    "pixtral-12b-2409"
#    "ministral-8b-2410"
#    "ministral-3b-2410"
#)

# Mistral API key
API_KEY="Mistral_API_Key"

# CSV file path
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"

# Common parameters
MAX_SAMPLES=100
TEMPERATURE=0.7
SEED=42

# Number of runs per model
NUM_RUNS=10

# Create output directory
OUTPUT_DIR="$(pwd)/results_api"
mkdir -p "$OUTPUT_DIR"

# Log file for all jobs
LOG_FILE="$OUTPUT_DIR/all_api_jobs.log"
echo "Starting API job submission at $(date)" > "$LOG_FILE"
echo "Models: ${API_MODELS[@]}" >> "$LOG_FILE"
echo "Runs per model: $NUM_RUNS" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Counter for total jobs
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# Loop through all API models
for model in "${API_MODELS[@]}"; do
    # Extract clean model name for filenames
    model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    for run in $(seq 1 $NUM_RUNS); do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        
        # Create unique job name
        job_name="${model_name}_paid_api_run${run}"
        
        # Output files
        out_file="$OUTPUT_DIR/${job_name}_%j.out"
        err_file="$OUTPUT_DIR/${job_name}_%j.err"
        csv_out="$OUTPUT_DIR/${job_name}_output.csv"
        
        echo "Submitting job $TOTAL_JOBS: $job_name" | tee -a "$LOG_FILE"
        
        # Submit job immediately (no delay)
        srun -K \
          --output="$out_file" \
          --error="$err_file" \
          --job-name="$job_name" \
          --ntasks=1 \
          --gpus-per-task=0 \
          --cpus-per-task=4 \
          --mem=40GB \
          --time=180 \
          --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
          --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
          --container-workdir="$(pwd)" \
          bash -c "chmod +x $(pwd)/install.sh && \
                   $(pwd)/install.sh && \
                   python $(pwd)/mistral_models_api.py \
                     --model $model \
                     --csv $CSV_PATH \
                     --api_key $API_KEY \
                     --temperature $TEMPERATURE \
                     --max_samples $MAX_SAMPLES \
                     --seed $((SEED + run)) \
                     --out_csv $csv_out" &
        
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
        
        # Small delay to avoid overwhelming the scheduler
        sleep 2
    done
done

# Wait for all background jobs to complete
wait

echo "" | tee -a "$LOG_FILE"
echo "All $TOTAL_JOBS API jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "Job Submission Summary:" | tee -a "$LOG_FILE"
echo "  Total jobs submitted: $TOTAL_JOBS" | tee -a "$LOG_FILE"
echo "  All jobs started immediately" | tee -a "$LOG_FILE"
echo "  Expected runtime per job: ~3 hours max" | tee -a "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "To monitor jobs:" | tee -a "$LOG_FILE"
echo "  squeue -u $USER | grep mistral" | tee -a "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "To check results:" | tee -a "$LOG_FILE"
echo "  ls -lh $OUTPUT_DIR/*.csv" | tee -a "$LOG_FILE"