# Multi GPU

Scripts for testing multi-GPU model loading and parallelism strategies.

## Files

| File | Description |
|------|-------------|
| `test_model_loading.py` | Test model loading (layer parallelism) across multiple GPUs |
| `test_model_loading_tensor.py` | Test tensor parallelism for model loading |
| `requirements.txt` | Python dependencies |
| `install.sh` | Install dependencies |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `test_model_loading_layer_parallel.sh` | Test layer parallelism loading |
| `test_model_loading_tensor.sh` | Test tensor parallelism loading |

## Installation

```bash
bash install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

Test tensor parallelism:

```bash
bash test_model_loading_tensor.sh
```

Test layer parallelism:

```bash
bash test_model_loading_layer_parallel.sh
```