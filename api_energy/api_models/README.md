# API Models

Scripts for benchmarking LLM inference via cloud APIs (Mistral, OpenAI, Gemini, Llama/Groq).

## Files

| File | Description |
|------|-------------|
| `mistral_models_api.py` | Run Mistral API queries with per-sample tracking |
| `mistral_models_api_scheduled.py` | Scheduled Mistral API experiments over multiple days |
| `mistral_models_api_scheduled_simple.py` | Simplified scheduled Mistral API runner |
| `openai_models_api.py` | Run OpenAI API queries using Batch API |
| `gemini_models_api.py` | Run Google Gemini API queries |
| `llama_models_api.py` | Run Llama models via Groq API |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `run_model_api_all.sh` | Run all Mistral API models |
| `run_model_api_all_scheduled_free.sh` | Scheduled runs for free-tier Mistral models |
| `run_model_api_all_scheduled_paid.sh` | Scheduled runs for paid Mistral models |
| `run_openai_model_api_all.sh` | Run all OpenAI API models |
| `run_gemini_model_api_all.sh` | Run all Gemini API models |
| `run_llama_model_api_all.sh` | Run Llama models via Groq API |

## Installation

```bash
bash install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

Example for Mistral API:

```bash
python mistral_models_api.py \
  --model open-mistral-7b \
  --csv /path/to/queries.csv \
  --api_key YOUR_API_KEY \
  --out_csv results.csv
```

Example for OpenAI API:

```bash
python openai_models_api.py \
  --model gpt-4o \
  --csv /path/to/queries.csv \
  --api_key YOUR_API_KEY \
  --out_csv results.csv
```

## Input CSV Format

The input CSV should contain:
- `query`: The input prompt
- `query_type`: "short" or "long"
- `output_type`: "short" or "long"

## Configuration

Common parameters:
- `--max_input_tokens_short` / `--max_input_tokens_long`: Input token limits
- `--max_new_tokens_short` / `--max_new_tokens_long`: Output token limits
- `--temperature`: Sampling temperature (default: 0.7)
- `--seed`: Random seed for reproducibility
- `--max_samples`: Maximum queries to process