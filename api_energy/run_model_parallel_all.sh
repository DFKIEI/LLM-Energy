#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_model_parallel_all.sh

# Define all ACTUALLY AVAILABLE models on HuggingFace

#MODELS=(
#    "mistralai/Mistral-7B-v0.3"
#    "mistralai/Mistral-7B-Instruct-v0.3"
#    "mistralai/Mistral-Nemo-Base-2407"
#    "mistralai/Mistral-Nemo-Instruct-2407"
#    "mistralai/Pixtral-12B-2409"
#    "mistralai/Mistral-Small-Instruct-2409"
#    "mistralai/Ministral-8B-Instruct-2410"
#    "mistralai/Ministral-3B-Instruct-2410"
#)

MODELS=(
    "mistralai/Mistral-7B-v0.3"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mistral-Nemo-Base-2407"
    "mistralai/Mistral-Nemo-Instruct-2407"
)

# Define all GPU partitions
GPUS=(
    "A100-40GB"
    "A100-80GB"
    "A100-PCI"
    "H100"
    "H100-PCI"
    "H200"
    "RTXA6000"
)

# Number of runs per configuration
NUM_RUNS=5

# HuggingFace token
HF_TOKEN="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"

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

# Memory requirements per GPU type
declare -A GPU_MEM
GPU_MEM["A100-40GB"]=60
GPU_MEM["A100-80GB"]=70
GPU_MEM["A100-PCI"]=60
GPU_MEM["B200"]=100
GPU_MEM["H100"]=80
GPU_MEM["H100-PCI"]=80
GPU_MEM["H200"]=80
GPU_MEM["H200-PCI"]=80
GPU_MEM["RTXA6000"]=60

# Time limits per GPU type (in minutes)
declare -A GPU_TIME
GPU_TIME["A100-40GB"]=240
GPU_TIME["A100-80GB"]=240
GPU_TIME["A100-PCI"]=240
GPU_TIME["B200"]=180
GPU_TIME["H100"]=180
GPU_TIME["H100-PCI"]=180
GPU_TIME["H200"]=180
GPU_TIME["H200-PCI"]=180
GPU_TIME["RTXA6000"]=300

# Create output directory
OUTPUT_DIR="$(pwd)/results_parallel"
mkdir -p "$OUTPUT_DIR"

# Log file for all jobs
LOG_FILE="$OUTPUT_DIR/all_jobs.log"
echo "Starting batch job submission at $(date)" > "$LOG_FILE"

# Counter for total jobs
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# Loop through all combinations
for model in "${MODELS[@]}"; do
    # Extract clean model name for filenames
    model_name=$(echo "$model" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9._-]/_/g')
    
    for gpu in "${GPUS[@]}"; do
        # Get memory and time for this GPU
        mem="${GPU_MEM[$gpu]}GB"
        time_limit="${GPU_TIME[$gpu]}"
        
        for run in $(seq 2 $NUM_RUNS); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Create unique job name
            job_name="${model_name}_${gpu}_run${run}"
            
            # Output files
            out_file="$OUTPUT_DIR/${job_name}_%j.out"
            err_file="$OUTPUT_DIR/${job_name}_%j.err"
            csv_out="$OUTPUT_DIR/${job_name}_output.csv"
            
            echo "Submitting job $TOTAL_JOBS: $job_name" | tee -a "$LOG_FILE"
            
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
                       python $(pwd)/mistral_models_parallelism.py \
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
            
            # Add a small delay to avoid overwhelming the scheduler
            sleep 2
            
            # Optional: limit concurrent submissions
            if [ $((SUBMITTED_JOBS % 10)) -eq 0 ]; then
                echo "Submitted $SUBMITTED_JOBS jobs, pausing for 10 seconds..." | tee -a "$LOG_FILE"
                sleep 10
            fi
        done
    done
done

# Wait for all background jobs to complete
wait

echo "" | tee -a "$LOG_FILE"
echo "All $TOTAL_JOBS jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To monitor jobs, run: squeue -u $USER" | tee -a "$LOG_FILE"
echo "To check results: ls -lh $OUTPUT_DIR/*.csv" | tee -a "$LOG_FILE"