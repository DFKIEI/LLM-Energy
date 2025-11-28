#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/scheduled_api_jobs_summary.py

import re
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
from glob import glob
from datetime import datetime, timedelta


def extract_run_metrics(content: str) -> List[Dict[str, str]]:
    """Extract metrics from each completed run in the output file."""
    runs = []
    
    # Pattern to capture start time and metrics
    run_pattern = r'STARTING RUN #(\d+).*?Started at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?'
    run_pattern += r'RUN #\d+ COMPLETED.*?'
    run_pattern += r'Samples processed: (\d+)/\d+.*?'
    run_pattern += r'Failed calls: (\d+).*?'
    run_pattern += r'Total batch time: ([\d:]+).*?'
    run_pattern += r'Total API time: ([\d:]+).*?'
    run_pattern += r'Average API call: ([\d.]+)s.*?'
    run_pattern += r'Total input tokens:\s+([\d,]+).*?'
    run_pattern += r'Total output tokens:\s+([\d,]+).*?'
    run_pattern += r'Throughput: ([\d.]+) tokens/second'
    
    matches = re.finditer(run_pattern, content, re.DOTALL)
    
    for match in matches:
        run_number = match.group(1)
        start_datetime = match.group(2)
        samples_processed = match.group(3)
        failed_calls = match.group(4)
        total_batch_time = match.group(5)
        total_api_time = match.group(6)
        avg_api_call = match.group(7)
        total_input_tokens = match.group(8).replace(',', '')
        total_output_tokens = match.group(9).replace(',', '')
        throughput = match.group(10)
        
        # Convert time format HH:MM:SS to total seconds
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            return 0
        
        total_batch_seconds = time_to_seconds(total_batch_time)
        total_api_seconds = time_to_seconds(total_api_time)
        
        # Calculate end datetime (start + total_api_time)
        try:
            start_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
            end_dt = start_dt + timedelta(seconds=total_api_seconds)
            end_datetime = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            end_datetime = "N/A"
            print(f"Warning: Could not calculate end time for run {run_number}: {e}", file=sys.stderr)
        
        runs.append({
            'run_number': run_number,
            'start_datetime': start_datetime,
            'end_datetime': end_datetime,
            'samples_processed': samples_processed,
            'failed_calls': failed_calls,
            'total_batch_time_seconds': str(total_batch_seconds),
            'total_api_time_seconds': str(total_api_seconds),
            'avg_api_call_seconds': avg_api_call,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': str(int(total_input_tokens) + int(total_output_tokens)),
            'throughput_tokens_per_second': throughput
        })
    
    return runs


def extract_model_and_tier(filepath: str) -> tuple:
    """Extract model name and tier (free/paid) from filename."""
    filename = Path(filepath).stem
    
    # Examples:
    # open-mistral-7b_scheduled_2289083 -> (open-mistral-7b, free)
    # open-mistral-7b_scheduled2_2306547 -> (open-mistral-7b, free)
    # open-mistral-7b_scheduled3_2306548 -> (open-mistral-7b, free)
    # open-mistral-7b_scheduled_paid_2306547 -> (open-mistral-7b, paid)
    # open-mistral-7b_scheduled_paid2_2306548 -> (open-mistral-7b, paid)
    # open-mistral-7b_scheduled_paid3_2306549 -> (open-mistral-7b, paid)
    # open-mistral-nemo_scheduled_2289084 -> (open-mistral-nemo, free)
    # open-mistral-nemo_scheduled_paid_2306548 -> (open-mistral-nemo, paid)
    
    if 'open-mistral-7b' in filename:
        model = 'open-mistral-7b'
    elif 'open-mistral-nemo' in filename:
        model = 'open-mistral-nemo'
    else:
        model = 'unknown'
    
    # Check for 'paid' anywhere in the filename
    if 'paid' in filename:
        tier = 'paid'
    else:
        tier = 'free'
    
    return model, tier


def process_file(filepath: str) -> List[Dict[str, str]]:
    """Process a single output file and extract all run metrics."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        model, tier = extract_model_and_tier(filepath)
        runs = extract_run_metrics(content)
        
        # Add model name, tier, and source file to each run
        for run in runs:
            run['model'] = model
            run['tier'] = tier
            run['source_file'] = Path(filepath).name
        
        return runs
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def find_output_files(base_dir: str) -> Dict[str, List[str]]:
    """Find all relevant output files for the models."""
    # Use broader patterns to catch all variations
    patterns = {
        'open-mistral-7b_free': [
            f'{base_dir}/open-mistral-7b_scheduled_*.out',
            f'{base_dir}/open-mistral-7b_scheduled2_*.out',
            f'{base_dir}/open-mistral-7b_scheduled3_*.out',
        ],
        'open-mistral-7b_paid': [
            f'{base_dir}/open-mistral-7b_scheduled_paid_*.out',
            f'{base_dir}/open-mistral-7b_scheduled_paid2_*.out',
            f'{base_dir}/open-mistral-7b_scheduled_paid3_*.out',
        ],
        'open-mistral-nemo_free': [
            f'{base_dir}/open-mistral-nemo_scheduled_*.out',
            f'{base_dir}/open-mistral-nemo_scheduled2_*.out',
            f'{base_dir}/open-mistral-nemo_scheduled3_*.out',
        ],
        'open-mistral-nemo_paid': [
            f'{base_dir}/open-mistral-nemo_scheduled_paid_*.out',
            f'{base_dir}/open-mistral-nemo_scheduled_paid2_*.out',
            f'{base_dir}/open-mistral-nemo_scheduled_paid3_*.out',
        ],
    }
    
    found_files = {}
    for key, pattern_list in patterns.items():
        all_files = []
        for pattern in pattern_list:
            files = glob(pattern)
            all_files.extend(files)
        
        # Remove duplicates and sort
        all_files = sorted(set(all_files))
        
        # For free tier, explicitly exclude files with 'paid' in name
        if 'paid' not in key:
            all_files = [f for f in all_files if 'paid' not in f]
        
        found_files[key] = all_files
    
    return found_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract API metrics from scheduled job output files"
    )
    parser.add_argument(
        '--dir', 
        default='results_api_scheduled_free',
        help='Directory containing output files (default: results_api_scheduled_free)'
    )
    parser.add_argument(
        '--output', '-o', 
        default='api_run_metrics.csv',
        help='Output CSV file (default: api_run_metrics.csv)'
    )
    
    args = parser.parse_args()
    
    # Find all relevant files
    print(f"Searching for output files in: {args.dir}\n")
    file_groups = find_output_files(args.dir)
    
    # Display found files
    for key, files in file_groups.items():
        model, tier = key.rsplit('_', 1)
        print(f"{model} ({tier} tier): {len(files)} file(s)")
        for f in files:
            print(f"  - {Path(f).name}")
    
    print()
    
    # Collect all runs from all files
    all_runs = []
    for key, files in file_groups.items():
        for filepath in files:
            print(f"Processing: {Path(filepath).name}")
            runs = process_file(filepath)
            all_runs.extend(runs)
            print(f"  Found {len(runs)} completed runs")
    
    if not all_runs:
        print("\nNo runs found in any file!", file=sys.stderr)
        sys.exit(1)
    
    # Define CSV columns
    fieldnames = [
        'model',
        'tier',
        'run_number',
        'start_datetime',
        'end_datetime',
        'samples_processed',
        'failed_calls',
        'total_batch_time_seconds',
        'total_api_time_seconds',
        'avg_api_call_seconds',
        'total_input_tokens',
        'total_output_tokens',
        'total_tokens',
        'throughput_tokens_per_second',
        'source_file'
    ]
    
    # Write to CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_runs)
    
    print(f"\n✓ Extracted {len(all_runs)} runs to {args.output}")
    
    # Print summary statistics
    print("\nSummary by model and tier:")
    summary = {}
    for run in all_runs:
        key = (run['model'], run['tier'])
        if key not in summary:
            summary[key] = {
                'total_runs': 0, 
                'total_samples': 0, 
                'total_failures': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'first_run': None,
                'last_run': None
            }
        summary[key]['total_runs'] += 1
        summary[key]['total_samples'] += int(run['samples_processed'])
        summary[key]['total_failures'] += int(run['failed_calls'])
        summary[key]['total_input_tokens'] += int(run['total_input_tokens'])
        summary[key]['total_output_tokens'] += int(run['total_output_tokens'])
        
        # Track first and last run times
        if run['start_datetime'] != 'N/A':
            if summary[key]['first_run'] is None or run['start_datetime'] < summary[key]['first_run']:
                summary[key]['first_run'] = run['start_datetime']
            if summary[key]['last_run'] is None or run['end_datetime'] > summary[key]['last_run']:
                summary[key]['last_run'] = run['end_datetime']
    
    for (model, tier), stats in sorted(summary.items()):
        print(f"\n  {model} ({tier} tier):")
        print(f"    Completed runs: {stats['total_runs']}")
        print(f"    Total samples: {stats['total_samples']:,}")
        print(f"    Total failures: {stats['total_failures']}")
        print(f"    Total input tokens: {stats['total_input_tokens']:,}")
        print(f"    Total output tokens: {stats['total_output_tokens']:,}")
        print(f"    Avg samples/run: {stats['total_samples'] / stats['total_runs']:.1f}")
        if stats['first_run'] and stats['last_run']:
            print(f"    Time range: {stats['first_run']} to {stats['last_run']}")


if __name__ == "__main__":
    main()