#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/run_model_api_all_scheduled_free.sh

API_MODELS=(
    "open-mistral-7b"
    "open-mistral-nemo"
)

API_KEY="Your_Mistral_Key"
CSV_PATH="/netscratch/banwari/api_gpu/synthetic_prompts.csv"
MAX_SAMPLES=100
TEMPERATURE=0.7
SEED=42
BATCH_INTERVAL=30  # Minutes between batches
MAX_RUNTIME_HOURS=72  # 3 days

OUTPUT_DIR="$(pwd)/results_api_scheduled_free"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/scheduled_jobs.log"
echo "Starting scheduled API jobs at $(date)" > "$LOG_FILE"

for model in "${API_MODELS[@]}"; do
    model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
    job_name="${model_name}_scheduled_paid3"
    
    out_file="$OUTPUT_DIR/${job_name}_%j.out"
    err_file="$OUTPUT_DIR/${job_name}_%j.err"
    detailed_csv="$OUTPUT_DIR/${job_name}_detailed.csv"
    metrics_csv="$OUTPUT_DIR/${job_name}_metrics.csv"
    
    echo "Submitting ${job_name} (will run for ${MAX_RUNTIME_HOURS} hours)" | tee -a "$LOG_FILE"
    echo "  Detailed output: ${detailed_csv}" | tee -a "$LOG_FILE"
    echo "  Metrics output: ${metrics_csv}" | tee -a "$LOG_FILE"
    
    srun -K \
      --output="$out_file" \
      --error="$err_file" \
      --job-name="$job_name" \
      --ntasks=1 \
      --gpus-per-task=0 \
      --cpus-per-task=2 \
      --mem=40GB \
      --time=4320 \
      --container-mounts="/netscratch/$USER:/netscratch/$USER,$(pwd):$(pwd)" \
      --container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
      --container-workdir="$(pwd)" \
      bash -c "chmod +x $(pwd)/install.sh && \
               $(pwd)/install.sh && \
               python $(pwd)/mistral_models_api_scheduled.py \
                 --model $model \
                 --csv $CSV_PATH \
                 --api_key $API_KEY \
                 --temperature $TEMPERATURE \
                 --max_samples $MAX_SAMPLES \
                 --seed $SEED \
                 --batch_interval_minutes $BATCH_INTERVAL \
                 --max_runtime_hours $MAX_RUNTIME_HOURS \
                 --out_csv $detailed_csv \
                 --metrics_csv $metrics_csv \
                 --resume" &
    
    sleep 5
done

wait

echo "All scheduled jobs submitted at $(date)" | tee -a "$LOG_FILE"
echo "Jobs will complete in approximately ${MAX_RUNTIME_HOURS} hours (3 days)"
echo "Monitor with: tail -f $OUTPUT_DIR/*.out"
echo ""
echo "Output files will be:"
for model in "${API_MODELS[@]}"; do
    model_name=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
    echo "  ${model_name}_detailed.csv  (first run only)"
    echo "  ${model_name}_metrics.csv   (all ~123 runs)"
done