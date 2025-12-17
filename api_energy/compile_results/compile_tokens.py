#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/compile_token_summary.py

import os
import sys
import re
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

def extract_model_info_from_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Extract model name, type (API/Local), GPU, and run number from filename.
    
    Examples:
    - 'open_mistral_nemo_api_run1_2203514.out' -> ('open-mistral-nemo', 'API', 'N/A', '1')
    - 'Mistral-7B-Instruct-v0.3_A100-40GB_run1_2223918.out' -> ('Mistral-7B-Instruct-v0.3', 'Local', 'A100-40GB', '1')
    - 'llama_3_1_8b_instant_api_run1_123456.out' -> ('llama-3.1-8b-instant', 'API', 'N/A', '1')
    """
    basename = os.path.basename(filename)
    
    # Remove .out extension
    basename = basename.replace('.out', '')
    
    # Remove job ID suffix (underscore followed by digits at the end)
    basename = re.sub(r'_\d+$', '', basename)
    
    # Check if it's an API model
    if '_api_run' in basename:
        # API model pattern: {model}_api_run{N}
        parts = basename.split('_api_run')
        model_name = parts[0].replace('_', '-')
        run_num = parts[1] if len(parts) > 1 else 'unknown'
        return model_name, 'API', 'N/A', run_num
    
    # Local model pattern: {model}_{GPU}_run{N}
    elif '_run' in basename:
        parts = basename.rsplit('_run', 1)
        run_num = parts[1] if len(parts) > 1 else 'unknown'
        
        # Split model and GPU (GPU is typically the last part before _run)
        model_gpu_parts = parts[0].rsplit('_', 1)
        if len(model_gpu_parts) == 2:
            model_name, gpu = model_gpu_parts
            # Check if GPU looks like a GPU name
            if any(g in gpu for g in ['A100', 'H100', 'H200', 'V100', 'A40', 'RTX']):
                return model_name, 'Local', gpu, run_num
        
        # Couldn't parse GPU, treat whole thing as model name
        return parts[0], 'Local', 'Unknown', run_num
    
    return 'Unknown', 'Unknown', 'Unknown', 'Unknown'

def parse_local_out_file(filepath: str) -> Optional[Dict]:
    """
    Parse token statistics from local model .out files.
    
    Looks for pattern:
    ======================================================================
    OVERALL RUN STATISTICS
    ======================================================================
    Total Queries Processed: 100
    Total Input Tokens: 9,810
    Total Output Tokens: 304,753
    Total Tokens: 314,563
    Avg Input Tokens/Query: 98.1
    Avg Output Tokens/Query: 3047.5
    Avg Total Tokens/Query: 3145.6
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for the statistics section
        stats_pattern = r"OVERALL RUN STATISTICS.*?Total Queries Processed:\s*([\d,]+).*?Total Input Tokens:\s*([\d,]+).*?Total Output Tokens:\s*([\d,]+).*?Total Tokens:\s*([\d,]+).*?Avg Input Tokens/Query:\s*([\d,.]+).*?Avg Output Tokens/Query:\s*([\d,.]+).*?Avg Total Tokens/Query:\s*([\d,.]+)"
        
        match = re.search(stats_pattern, content, re.DOTALL)
        
        if match:
            stats = {
                'num_queries': int(match.group(1).replace(',', '')),
                'total_input_tokens': int(match.group(2).replace(',', '')),
                'total_output_tokens': int(match.group(3).replace(',', '')),
                'total_tokens': int(match.group(4).replace(',', '')),
                'avg_input_tokens': float(match.group(5).replace(',', '')),
                'avg_output_tokens': float(match.group(6).replace(',', '')),
                'avg_total_tokens': float(match.group(7).replace(',', '')),
            }
            return stats
        
        return None
        
    except Exception as e:
        print(f"  ⚠️  Error parsing {os.path.basename(filepath)}: {e}")
        return None

def parse_api_out_file(filepath: str) -> Optional[Dict]:
    """
    Parse token statistics from API model .out files.
    
    Looks for patterns like:
    [Sample 98/100] Processing: short query, short output
      Max input tokens: 2048, Max output tokens: 2048
      ✓ Completed in 4.58s
      Tokens - Input: 36, Output: 667, Total: 703
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Pattern to find token information
        # Matches: "Tokens - Input: 36, Output: 667, Total: 703"
        pattern = r"Tokens\s+-\s+Input:\s+(\d+),\s+Output:\s+(\d+),\s+Total:\s+(\d+)"
        matches = re.findall(pattern, content)
        
        if not matches:
            # Try alternative pattern without "Tokens -" prefix
            pattern = r"Input:\s+(\d+),\s+Output:\s+(\d+),\s+Total:\s+(\d+)"
            matches = re.findall(pattern, content)
        
        if not matches:
            return None
        
        # Calculate statistics
        input_tokens = [int(m[0]) for m in matches]
        output_tokens = [int(m[1]) for m in matches]
        total_tokens = [int(m[2]) for m in matches]
        
        stats = {
            'num_queries': len(matches),
            'total_input_tokens': sum(input_tokens),
            'total_output_tokens': sum(output_tokens),
            'total_tokens': sum(total_tokens),
            'avg_input_tokens': sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            'avg_output_tokens': sum(output_tokens) / len(output_tokens) if output_tokens else 0,
            'avg_total_tokens': sum(total_tokens) / len(total_tokens) if total_tokens else 0,
            'min_input_tokens': min(input_tokens) if input_tokens else 0,
            'max_input_tokens': max(input_tokens) if input_tokens else 0,
            'min_output_tokens': min(output_tokens) if output_tokens else 0,
            'max_output_tokens': max(output_tokens) if output_tokens else 0,
        }
        
        return stats
        
    except Exception as e:
        print(f"  ⚠️  Error parsing {os.path.basename(filepath)}: {e}")
        return None

def compile_summaries(base_dirs: List[str]) -> pd.DataFrame:
    """Compile token summaries from all .out files."""
    
    all_results = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        
        print(f"\nScanning directory: {base_dir}")
        
        # Find all .out files
        out_files = glob.glob(os.path.join(base_dir, '*.out'))
        
        # Filter out non-model output files
        out_files = [f for f in out_files if not any(skip in os.path.basename(f).lower() 
                     for skip in ['collate', 'dataset', 'compile'])]
        
        print(f"Found {len(out_files)} model .out files")
        
        for out_file in out_files:
            filename = os.path.basename(out_file)
            print(f"  Processing: {filename}")
            
            # Extract metadata
            model_name, model_type, gpu, run_num = extract_model_info_from_filename(out_file)
            
            if model_type == 'Unknown':
                print(f"    ⚠️  Could not parse filename format")
                continue
            
            # Parse based on type
            if model_type == 'API':
                stats = parse_api_out_file(out_file)
            elif model_type == 'Local':
                stats = parse_local_out_file(out_file)
            else:
                print(f"    ⚠️  Unknown model type")
                continue
            
            if stats is None:
                print(f"    ⚠️  Could not parse statistics from file")
                continue
            
            # Combine metadata and stats
            result = {
                'model_name': model_name,
                'model_type': model_type,
                'gpu': gpu,
                'run': run_num,
                'filename': filename,
                **stats
            }
            
            all_results.append(result)
            print(f"    ✓ {stats['num_queries']} queries, {stats['total_tokens']:,} tokens")
    
    return pd.DataFrame(all_results)

def create_aggregated_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated summary by model and type."""
    
    # Group by model and type
    agg_funcs = {
        'num_queries': 'sum',
        'total_input_tokens': 'sum',
        'total_output_tokens': 'sum',
        'total_tokens': 'sum',
        'avg_input_tokens': 'mean',
        'avg_output_tokens': 'mean',
        'avg_total_tokens': 'mean',
        'run': 'count'  # Number of runs
    }
    
    # Add min/max if they exist
    if 'min_input_tokens' in df.columns:
        agg_funcs['min_input_tokens'] = 'min'
        agg_funcs['max_input_tokens'] = 'max'
        agg_funcs['min_output_tokens'] = 'min'
        agg_funcs['max_output_tokens'] = 'max'
    
    summary = df.groupby(['model_name', 'model_type', 'gpu']).agg(agg_funcs).reset_index()
    summary.rename(columns={'run': 'num_runs'}, inplace=True)
    
    # Calculate per-query averages across all runs
    summary['avg_input_per_query'] = summary['total_input_tokens'] / summary['num_queries']
    summary['avg_output_per_query'] = summary['total_output_tokens'] / summary['num_queries']
    summary['avg_total_per_query'] = summary['total_tokens'] / summary['num_queries']
    
    # Sort by total tokens
    summary = summary.sort_values('total_tokens', ascending=False)
    
    return summary

def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create API vs Local comparison table."""
    
    # Function to normalize model names for comparison
    def normalize_model_name(name):
        name = name.lower()
        # Remove version numbers and variants
        name = re.sub(r'-v\d+\.\d+', '', name)
        name = re.sub(r'-instruct', '', name)
        name = re.sub(r'-\d+b', '', name)
        name = re.sub(r'instant', '', name)
        # Extract base model name
        if 'mistral' in name:
            return 'mistral-7b'
        elif 'llama' in name and '8b' in name:
            return 'llama-8b'
        elif 'llama' in name:
            return 'llama'
        elif 'qwen' in name:
            return 'qwen'
        elif 'gpt' in name:
            return 'gpt'
        elif 'gemini' in name:
            return 'gemini'
        return name
    
    df_copy = df.copy()
    df_copy['base_model'] = df_copy['model_name'].apply(normalize_model_name)
    
    # Separate API and Local
    api_df = df_copy[df_copy['model_type'] == 'API'].groupby('base_model').agg({
        'total_tokens': 'sum',
        'num_queries': 'sum',
        'avg_total_tokens': 'mean',
        'model_name': 'first',
        'run': 'count'
    }).reset_index()
    api_df.columns = ['base_model', 'api_total_tokens', 'api_queries', 'api_avg_tokens', 'api_model_name', 'api_runs']
    
    local_df = df_copy[df_copy['model_type'] == 'Local'].groupby('base_model').agg({
        'total_tokens': 'sum',
        'num_queries': 'sum',
        'avg_total_tokens': 'mean',
        'model_name': 'first',
        'run': 'count'
    }).reset_index()
    local_df.columns = ['base_model', 'local_total_tokens', 'local_queries', 'local_avg_tokens', 'local_model_name', 'local_runs']
    
    # Merge
    comparison = pd.merge(api_df, local_df, on='base_model', how='outer')
    
    # Calculate differences
    comparison['token_diff_%'] = (
        (comparison['api_avg_tokens'] - comparison['local_avg_tokens']) / 
        comparison['local_avg_tokens'] * 100
    ).round(2)
    
    # Calculate absolute difference
    comparison['token_diff_absolute'] = (
        comparison['api_avg_tokens'] - comparison['local_avg_tokens']
    ).round(2)
    
    return comparison

def main():
    print("="*70)
    print("TOKEN SUMMARY COMPILATION SCRIPT")
    print("Extracting token counts from .out files only")
    print("="*70)
    
    # Define directories to scan
    base_dirs = [
        './results_api',
        './results_llama_api',
        './results_parallel_tokens_fixed_deterministic',
        './results_llama_parallel_tokens_deterministic',
    ]
    
    # Compile all results
    print("\n" + "="*70)
    print("COMPILING TOKEN STATISTICS FROM .OUT FILES")
    print("="*70)
    
    df = compile_summaries(base_dirs)
    
    if df.empty:
        print("\n❌ No data found! Check that .out files exist in the directories.")
        print("\nSearched directories:")
        for d in base_dirs:
            exists = "✓" if os.path.exists(d) else "✗"
            print(f"  {exists} {d}")
        sys.exit(1)
    
    print(f"\n✓ Compiled {len(df)} result files")
    
    # Save detailed results
    output_file = 'token_summary_detailed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed summary to: {output_file}")
    
    # Create aggregated summary
    print("\n" + "="*70)
    print("CREATING AGGREGATED SUMMARY")
    print("="*70)
    
    agg_df = create_aggregated_summary(df)
    agg_output = 'token_summary_aggregated.csv'
    agg_df.to_csv(agg_output, index=False)
    print(f"\n✓ Saved aggregated summary to: {agg_output}")
    
    # Create comparison table
    print("\n" + "="*70)
    print("CREATING API VS LOCAL COMPARISON")
    print("="*70)
    
    comparison_df = create_comparison_table(df)
    comparison_output = 'token_summary_comparison.csv'
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\n✓ Saved comparison to: {comparison_output}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal Files Analyzed: {len(df)}")
    print(f"Total Queries Processed: {df['num_queries'].sum():,}")
    print(f"Total Input Tokens: {df['total_input_tokens'].sum():,}")
    print(f"Total Output Tokens: {df['total_output_tokens'].sum():,}")
    print(f"Total Tokens: {df['total_tokens'].sum():,}")
    
    print(f"\n{'Model Type':<15} {'Files':<10} {'Queries':<15} {'Total Tokens':<20} {'Avg Tokens/Query':<20}")
    print("-" * 80)
    for model_type in sorted(df['model_type'].unique()):
        type_df = df[df['model_type'] == model_type]
        num_files = len(type_df)
        total_queries = type_df['num_queries'].sum()
        total_tokens = type_df['total_tokens'].sum()
        avg_tokens = total_tokens / total_queries if total_queries > 0 else 0
        print(f"{model_type:<15} {num_files:<10} {total_queries:<15,} {total_tokens:<20,} {avg_tokens:<20,.2f}")
    
    print(f"\nTop 20 Models by Total Tokens:")
    print(f"{'Model Name':<40} {'Type':<10} {'GPU':<15} {'Runs':<8} {'Total Tokens':<20}")
    print("-" * 100)
    model_summary = df.groupby(['model_name', 'model_type', 'gpu']).agg({
        'run': 'count',
        'total_tokens': 'sum'
    }).reset_index().sort_values('total_tokens', ascending=False)
    
    for _, row in model_summary.head(20).iterrows():
        print(f"{row['model_name']:<40} {row['model_type']:<10} {row['gpu']:<15} {row['run']:<8} {row['total_tokens']:<20,}")
    
    # Save JSON summary
    json_output = 'token_summary.json'
    summary_dict = {
        'total_files': len(df),
        'total_queries': int(df['num_queries'].sum()),
        'total_input_tokens': int(df['total_input_tokens'].sum()),
        'total_output_tokens': int(df['total_output_tokens'].sum()),
        'total_tokens': int(df['total_tokens'].sum()),
        'by_type': {
            model_type: {
                'files': len(type_df),
                'queries': int(type_df['num_queries'].sum()),
                'input_tokens': int(type_df['total_input_tokens'].sum()),
                'output_tokens': int(type_df['total_output_tokens'].sum()),
                'total_tokens': int(type_df['total_tokens'].sum())
            }
            for model_type, type_df in df.groupby('model_type')
        },
        'by_model': {
            model: {
                'queries': int(model_df['num_queries'].sum()),
                'input_tokens': int(model_df['total_input_tokens'].sum()),
                'output_tokens': int(model_df['total_output_tokens'].sum()),
                'total_tokens': int(model_df['total_tokens'].sum())
            }
            for model, model_df in df.groupby('model_name')
        },
    }
    
    with open(json_output, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n✓ Saved JSON summary to: {json_output}")
    
    print("\n" + "="*70)
    print("COMPILATION COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print(f"  1. {output_file} - Detailed per-file statistics")
    print(f"  2. {agg_output} - Aggregated by model/GPU")
    print(f"  3. {comparison_output} - API vs Local comparison")
    print(f"  4. {json_output} - JSON summary")
    print("="*70)

if __name__ == "__main__":
    main()