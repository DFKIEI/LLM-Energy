# Dataset

Scripts for creating the query dataset used in experiments.

## Files

| File | Description |
|------|-------------|
| `create_dataset.py` | Generate query dataset with short/long input and output combinations |
| `dataset_creation.sh` | Shell script to run dataset creation |
| `requirements.txt` | Python dependencies |
| `install.sh` | Install dependencies |

## Installation

```bash
bash install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python create_dataset.py \
  --output_csv queries.csv \
  --num_samples 1000
```

Or use the shell script:

```bash
bash dataset_creation.sh
```

## Output Format

The generated CSV contains:
- `query`: The input prompt
- `query_type`: "short" or "long"
- `output_type`: "short" or "long"