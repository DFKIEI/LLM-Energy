#!/bin/bash
# filepath: /home/banwari/llm_energy/api_energy/run_model.sh

# Define all available GPU partitions
GPU_PARTITIONS=(
    "A100-40GB" "A100-80GB" "A100-PCI"
    "H100" "H100-PCI" "H200" "H200-PCI" 
    "RTXA6000" "V100-32GB"
)

# Define Mistral models to test
MODELS=(
    "mistralai/Mistral-Nemo-Instruct-2407"          # 12B - multilingual
    "mistralai/Pixtral-12B-2409"                    # 12B - multimodal (text + images)
    "mistralai/Mistral-7B-Instruct-v0.3"           # 7B - classic instruction model
    "mistralai/Mistral-7B-Instruct-v0.2"           # 7B - previous version
    "mistralai/Codestral-22B-v0.1"                 # 22B - code specialized
)

# Model-specific configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS["mistralai/Mistral-Nemo-Instruct-2407"]="12B,60GB,128k"
MODEL_CONFIGS["mistralai/Pixtral-12B-2409"]="12B,60GB,128k"
MODEL_CONFIGS["mistralai/Mistral-7B-Instruct-v0.3"]="7B,40GB,32k"
MODEL_CONFIGS["mistralai/Mistral-7B-Instruct-v0.2"]="7B,40GB,32k"
MODEL_CONFIGS["mistralai/Codestral-22B-v0.1"]="22B,80GB,32k"

# Configuration
NUM_RUNS=1  # Number of runs per partition per model
MAX_SAMPLES=100  # Reduced for multiple model testing
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"
HF_TOKEN="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"
CURRENT_DIR=$(pwd)

echo "Submitting $NUM_RUNS runs for ${#MODELS[@]} models across ${#GPU_PARTITIONS[@]} GPU partitions..."
echo "Current directory: $CURRENT_DIR"
echo "Models to test:"
for model in "${MODELS[@]}"; do
    config=${MODEL_CONFIGS[$model]}
    IFS=',' read -r size min_mem context_len <<< "$config"
    echo "  - $model ($size, min $min_mem memory, $context_len context)"
done
echo ""

# Create jobs for each model and partition combination
for model in "${MODELS[@]}"; do
    # Get model configuration
    config=${MODEL_CONFIGS[$model]}
    IFS=',' read -r model_size min_memory context_length <<< "$config"
    
    # Create model name for filenames (replace / with _)
    model_name=$(basename "$model" | tr '/' '_')
    
    echo "Setting up jobs for $model ($model_size)..."
    
    for partition in "${GPU_PARTITIONS[@]}"; do
        
        # Adjust memory based on GPU type and model requirements
        case $partition in
            "V100-32GB"|"RTX3090") 
                if [[ "$min_memory" =~ ^[0-9]+GB$ ]] && [[ ${min_memory%GB} -gt 35 ]]; then
                    echo "  Skipping $partition (insufficient memory for $model)"
                    continue
                fi
                memory="30GB" 
                ;;
            "L40S") 
                if [[ "$min_memory" =~ ^[0-9]+GB$ ]] && [[ ${min_memory%GB} -gt 40 ]]; then
                    echo "  Skipping $partition (insufficient memory for $model)"
                    continue
                fi
                memory="40GB" 
                ;;
            "A100-40GB"|"RTXA6000") 
                if [[ "$min_memory" =~ ^[0-9]+GB$ ]] && [[ ${min_memory%GB} -gt 55 ]]; then
                    echo "  Skipping $partition (insufficient memory for $model)"
                    continue
                fi
                memory="60GB" 
                ;;
            "A100-80GB"|"A100-PCI") 
                memory="70GB" 
                ;;
            "H100"|"H100-PCI"|"H200"|"H200-PCI") 
                memory="80GB" 
                ;;
            *) 
                memory="60GB" 
                ;;
        esac
        
        # Submit multiple runs for this model-partition combination
        for run in $(seq 1 $NUM_RUNS); do
            
            # Create job script
            job_script="job_${model_name}_${partition}_run${run}.sh"
            cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${model_name}_${partition}_run${run}
#SBATCH --output=${model_name}_${partition}_run${run}_%j.out
#SBATCH --error=${model_name}_${partition}_run${run}_%j.err
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=${partition}
#SBATCH --mem=${memory}
#SBATCH --time=240
#SBATCH --container-mounts="/netscratch/\$USER:/netscratch/\$USER,${CURRENT_DIR}:${CURRENT_DIR}"
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh
#SBATCH --container-workdir="${CURRENT_DIR}"

echo "========================================="
echo "Starting ${model_name} run ${run} on ${partition}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Model: ${model}"
echo "Model size: ${model_size}"
echo "Context length: ${context_length}"
echo "Memory allocated: ${memory}"
echo "Timestamp: \$(date)"
echo "Working directory: ${CURRENT_DIR}"
echo "========================================="

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Install packages
echo "Installing packages..."
chmod +x ${CURRENT_DIR}/install.sh
${CURRENT_DIR}/install.sh

# Use API-EQUIVALENT token limits (matches mistral_models_api.py)
max_input_short=2048
max_input_long=8192
max_new_short=2048
max_new_long=8192

echo "Running inference with API-EQUIVALENT token limits:"
echo "  Max input tokens (short/long): \$max_input_short/\$max_input_long"
echo "  Max new tokens (short/long): \$max_new_short/\$max_new_long"
echo "  This matches mistral_models_api.py for fair comparison"
echo "  Expected long generation time with high GPU utilization"
echo ""

# Run the model with API-equivalent token limits
python ${CURRENT_DIR}/mistral_models.py \\
  --model "${model}" \\
  --csv ${CSV_PATH} \\
  --hf_token "${HF_TOKEN}" \\
  --temperature 0.7 \\
  --max_samples ${MAX_SAMPLES} \\
  --max_input_tokens_short \$max_input_short \\
  --max_input_tokens_long \$max_input_long \\
  --max_new_tokens_short \$max_new_short \\
  --max_new_tokens_long \$max_new_long \\
  --seed \$((42 + ${run})) \\
  --out_csv outputs_${model_name}_${partition}_run${run}.csv

echo ""
echo "========================================="
echo "Completed ${model_name} run ${run} on ${partition}"
echo "End timestamp: \$(date)"
echo "Output file: outputs_${model_name}_${partition}_run${run}.csv"
echo "========================================="
EOF
            
            # Submit the job
            echo "  Submitting run $run to $partition..."
            sbatch "$job_script"
            
            # Clean up job script
            rm "$job_script"
            
            # Small delay between submissions
            sleep 0.2
        done
        
        echo "  → Submitted $NUM_RUNS runs to $partition"
    done
    
    echo "Completed setup for $model"
    echo ""
done

echo ""
echo "========================================="
echo "SUBMISSION SUMMARY - API EQUIVALENT"
echo "========================================="
echo "- Models tested: ${#MODELS[@]}"
echo "- GPU partitions: ${#GPU_PARTITIONS[@]}"
echo "- Runs per model-partition: $NUM_RUNS"
echo "- Samples per run: $MAX_SAMPLES"
echo "- Job time limit: 4 hours (240 minutes)"
echo "- Estimated total jobs: $((${#MODELS[@]} * ${#GPU_PARTITIONS[@]} * NUM_RUNS))"
echo ""
echo "TOKEN CONFIGURATION (API-EQUIVALENT):"
echo "- Short input: 2048 tokens → 2048 new tokens"
echo "- Long input: 8192 tokens → 8192 new tokens"
echo "- This will generate VERY long responses"
echo "- High GPU utilization and energy consumption expected"
echo ""
echo "Models:"
for model in "${MODELS[@]}"; do
    echo "  - $(basename "$model")"
done
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Output files: outputs_<model>_<partition>_run<N>.csv"
echo "Log files: <model>_<partition>_run<N>_<jobid>.out/err"
echo "========================================="