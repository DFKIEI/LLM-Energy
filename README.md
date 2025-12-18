# LLM-Energy

Benchmarking energy consumption and performance of LLM inference across local and API deployments.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `api_energy/api_models/` | Scripts for running inference via cloud APIs (Mistral, OpenAI, Gemini, Llama/Groq) |
| `api_energy/local_models/` | Scripts for running local inference with consistent parallelism |
| `api_energy/multi_gpu/` | Multi-GPU loading and parallelism testing utilities |
| `api_energy/dataset/` | Dataset creation scripts for experiment queries |
| `api_energy/compile_results/` | Scripts for collating and summarizing experiment results |
| `api_energy/results/` | Output data from experiments |

## Installation

```bash
cd api_energy
bash install.sh
```

## Usage

1. Create dataset:
```bash
cd api_energy/dataset
bash dataset_creation.sh
```

2. Run API experiments:
```bash
cd api_energy/api_models
bash run_model_api_all.sh
```

3. Run local model experiments:
```bash
cd api_energy/local_models
bash run_model_parallel_deterministic_all.sh
```

4. Compile results:
```bash
cd api_energy/compile_results
bash collate_results.sh
```

## Supported Models

- Mistral (API and local)
- OpenAI (API)
- Google Gemini (API)
- Llama (API via Groq and local)