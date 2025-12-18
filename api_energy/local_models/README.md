# Local Models

Scripts for running local LLM inference.

## Files

| File | Description |
|------|-------------|
| `mistral_models_parallelism_consistent.py` | Run Mistral models locally |
| `llama_models_parallelism_consistent.py` | Run Llama models locally |
| `qwen_models_parallelism_consistent.py` | Run Qwen models locally |
| `requirements.txt` | Python dependencies |
| `install.sh` | Install dependencies |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `run_model_parallel_deterministic_all.sh` | Run all Mistral models with deterministic settings |
| `run_model_llama_parallel_deterministic_all.sh` | Run all Llama models with deterministic settings |
| `run_qwen_model_parallel_deterministic_all.sh` | Run all Qwen models with deterministic settings |

## Installation

```bash
bash install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

Example for Mistral models:

```bash
python mistral_models_parallelism_consistent.py \
  --model mistral-7b \
  --csv /path/to/queries.csv \
  --out_csv results.csv
```

Or run all models:

```bash
bash run_model_parallel_deterministic_all.sh
```