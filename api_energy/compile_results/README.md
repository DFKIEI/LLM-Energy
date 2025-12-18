# Compile Results

Scripts for collating and summarizing API and local experiment results.

## Files

| File | Description |
|------|-------------|
| `collate_results.py` | Aggregate results from multiple experiment runs into a single CSV |
| `compile_tokens.py` | Compile token usage statistics from experiment outputs |
| `scheduled_api_jobs_summary.py` | Generate summary reports for scheduled API job runs |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `collate_results.sh` | Batch run collate_results.py for multiple models |
| `compile_token_summary.sh` | Batch compile token summaries |

## Usage

Collate results from experiment runs:

```bash
python collate_results.py \
  --input_dir /path/to/results \
  --output_csv collated_results.csv
```

Compile token statistics:

```bash
python compile_tokens.py \
  --input_dir /path/to/results \
  --output_csv token_summary.csv
```

Generate scheduled job summary:

```bash
python scheduled_api_jobs_summary.py \
  --input_dir /path/to/scheduled_results \
  --output_csv job_summary.csv
```