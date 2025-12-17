import os
import re
import pandas as pd
from tqdm import tqdm
import glob

# === Configuration ===
base_dir = "/home/banwari/llm_energy/LLM-Energy/api_energy"

# Define the main directories
dirs = {
    "local_mistral": os.path.join(base_dir, "results_parallel_tokens_fixed_deterministic"),
    "mistral_api": os.path.join(base_dir, "results_api"),  # Both free and paid API
    "openai_api": os.path.join(base_dir, "results_openai_api"),
    "local_llama": os.path.join(base_dir, "results_llama_parallel_tokens_deterministic"),
    "llama_api": os.path.join(base_dir, "results_llama_api")
}

# Patterns for extracting metrics from .out files
patterns = {
    "Energy": r"Energy:\s+([0-9.]+) kWh",
    "CO2eq": r"CO2eq:\s+([0-9.]+) g",
    "Time": r"Time:\s+([0-9]+):([0-9]+):([0-9]+)",
    "Distance": r"([0-9.]+) km travelled by car"
}

# File naming patterns for each type
file_patterns = {
    "local_mistral": re.compile(r"(?P<model>[^_]+)_(?P<gpu>[^_]+)_run(?P<run>\d+)_\d+\.out"),
    "mistral_api": re.compile(r"(?P<model>[^_]+)_api_run(?P<run>\d+)_\d+\.out"),
    "mistral_paid_api": re.compile(r"(?P<model>[^_]+)_paid_api_run(?P<run>\d+)_\d+\.out"),
    "openai_api": re.compile(r"(?P<model>[^_]+)_openai_api_run(?P<run>\d+)_\d+\.out"),
    "local_llama": re.compile(r"(?P<model>[^_]+)_(?P<gpu>[^_]+)_run(?P<run>\d+)_\d+\.out"),
    "llama_api": re.compile(r"(?P<model>[^_]+)_api_run(?P<run>\d+)_\d+\.out")
}

# Store rows for each source separately
data = {
    "local_mistral": [],
    "mistral_api": [],
    "mistral_paid_api": [],
    "openai_api": [],
    "local_llama": [],
    "llama_api": []
}

def extract_metrics(filepath):
    """Extract energy metrics from a .out file"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Check if monitoring finished
    if "CarbonTracker: Finished monitoring." not in text:
        return None

    result = {}
    for metric, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        if metric == "Time":
            h, m, s = map(int, match.groups())
            result[metric] = f"{h:02d}:{m:02d}:{s:02d}"
            result["Time_seconds"] = h * 3600 + m * 60 + s
        else:
            result[metric] = round(float(match.group(1)), 6)
    
    return result if result else None

def process_local_mistral():
    """Process results from local Mistral models (results_parallel_tokens_deterministic)"""
    print("\nProcessing Local Mistral Models (results_parallel_tokens_deterministic)...")
    
    dir_path = dirs["local_mistral"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    pattern = file_patterns["local_mistral"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    for file in tqdm(files, desc="Local Mistral"):
        match = pattern.match(file)
        if not match:
            continue
        
        model = match.group("model")
        gpu = match.group("gpu")
        run_id = int(match.group("run"))
        
        filepath = os.path.join(dir_path, file)
        result = extract_metrics(filepath)
        
        if result is None:
            continue
        
        for metric, value in result.items():
            if metric == "Time_seconds":
                continue
            data["local_mistral"].append({
                "Type": "Local",
                "Model": model,
                "GPU": gpu,
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })

def process_mistral_api():
    """Process results from Mistral API - both free and paid (results_api)"""
    print("\nProcessing Mistral API Models (results_api)...")
    
    dir_path = dirs["mistral_api"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    free_pattern = file_patterns["mistral_api"]
    paid_pattern = file_patterns["mistral_paid_api"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    free_count = 0
    paid_count = 0
    
    for file in tqdm(files, desc="Mistral API (Free + Paid)"):
        # Try paid API pattern first
        match = paid_pattern.match(file)
        if match:
            api_type = "mistral_paid_api"
            type_label = "Paid-API"
            paid_count += 1
        else:
            # Try free API pattern
            match = free_pattern.match(file)
            if match:
                api_type = "mistral_api"
                type_label = "API"
                free_count += 1
            else:
                continue
        
        model = match.group("model")
        run_id = int(match.group("run"))
        
        filepath = os.path.join(dir_path, file)
        result = extract_metrics(filepath)
        
        if result is None:
            continue
        
        for metric, value in result.items():
            if metric == "Time_seconds":
                continue
            data[api_type].append({
                "Type": type_label,
                "Model": model,
                "GPU": "N/A",
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })
    
    print(f"  Found {free_count} free API files, {paid_count} paid API files")

def process_openai_api():
    """Process results from OpenAI API (results_openai_api)"""
    print("\nProcessing OpenAI API Models (results_openai_api)...")
    
    dir_path = dirs["openai_api"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    pattern = file_patterns["openai_api"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    for file in tqdm(files, desc="OpenAI API"):
        match = pattern.match(file)
        if not match:
            continue
        
        model = match.group("model")
        run_id = int(match.group("run"))
        
        filepath = os.path.join(dir_path, file)
        result = extract_metrics(filepath)
        
        if result is None:
            continue
        
        for metric, value in result.items():
            if metric == "Time_seconds":
                continue
            data["openai_api"].append({
                "Type": "API",
                "Model": model,
                "GPU": "N/A",
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })

def process_local_llama():
    """Process results from local Llama models (results_llama_parallel_tokens_deterministic)"""
    print("\nProcessing Local Llama Models (results_llama_parallel_tokens_deterministic)...")
    
    dir_path = dirs["local_llama"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    pattern = file_patterns["local_llama"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    for file in tqdm(files, desc="Local Llama"):
        match = pattern.match(file)
        if not match:
            continue
        
        model = match.group("model")
        gpu = match.group("gpu")
        run_id = int(match.group("run"))
        
        filepath = os.path.join(dir_path, file)
        result = extract_metrics(filepath)
        
        if result is None:
            continue
        
        for metric, value in result.items():
            if metric == "Time_seconds":
                continue
            data["local_llama"].append({
                "Type": "Local",
                "Model": model,
                "GPU": gpu,
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })

def process_llama_api():
    """Process results from Llama API (results_llama_api)"""
    print("\nProcessing Llama API Models (results_llama_api)...")
    
    dir_path = dirs["llama_api"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    pattern = file_patterns["llama_api"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    for file in tqdm(files, desc="Llama API"):
        match = pattern.match(file)
        if not match:
            continue
        
        model = match.group("model")
        run_id = int(match.group("run"))
        
        filepath = os.path.join(dir_path, file)
        result = extract_metrics(filepath)
        
        if result is None:
            continue
        
        for metric, value in result.items():
            if metric == "Time_seconds":
                continue
            data["llama_api"].append({
                "Type": "API",
                "Model": model,
                "GPU": "N/A",
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })

def save_individual_files():
    """Save separate CSV files for each source"""
    print("\n" + "=" * 70)
    print("SAVING INDIVIDUAL CSV FILES")
    print("=" * 70)
    
    output_files = {}
    
    for source_name, rows in data.items():
        if not rows:
            print(f"No data for {source_name}")
            continue
        
        df = pd.DataFrame(rows)
        df.sort_values(by=["Model", "GPU", "Metric", "Run"], inplace=True)
        
        # Define output filename
        output_file = os.path.join(base_dir, f"{source_name}_energy_metrics.csv")
        df.to_csv(output_file, index=False)
        output_files[source_name] = output_file
        
        print(f"{source_name}: {output_file}")
        print(f"   Rows: {len(df)}, Models: {df['Model'].nunique()}, Runs: {df['Run'].nunique()}")
    
    return output_files

def create_combined_csv():
    """Create a single combined CSV with all data"""
    print("\n" + "=" * 70)
    print("CREATING COMBINED CSV FILE")
    print("=" * 70)
    
    all_rows = []
    for source_name, rows in data.items():
        for row in rows:
            row["Source"] = source_name
            all_rows.append(row)
    
    if not all_rows:
        print("No data to combine!")
        return None
    
    df_combined = pd.DataFrame(all_rows)
    df_combined.sort_values(by=["Source", "Type", "Model", "GPU", "Metric", "Run"], inplace=True)
    
    output_file = os.path.join(base_dir, "combined_energy_metrics.csv")
    df_combined.to_csv(output_file, index=False)
    
    print(f"Combined file: {output_file}")
    print(f"   Total rows: {len(df_combined)}")
    print(f"   Sources: {df_combined['Source'].nunique()}")
    print(f"   Models: {df_combined['Model'].nunique()}")
    print(f"   GPUs: {df_combined['GPU'].nunique()}")
    
    return output_file

def print_summary_statistics():
    """Print summary statistics for all data sources"""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for source_name, rows in data.items():
        if not rows:
            continue
        
        df = pd.DataFrame(rows)
        
        print(f"\n{source_name.upper().replace('_', ' ')}:")
        print(f"  Total data points: {len(df)}")
        print(f"  Unique models: {df['Model'].nunique()} ({', '.join(df['Model'].unique())})")
        print(f"  Unique GPUs: {df['GPU'].nunique()} ({', '.join(df['GPU'].unique())})")
        print(f"  Runs per model: {df['Run'].nunique()}")
        print(f"  Metrics tracked: {', '.join(df['Metric'].unique())}")
        
        # Calculate average energy if available
        energy_data = df[df['Metric'] == 'Energy']
        if not energy_data.empty:
            avg_energy = energy_data['Value'].mean()
            print(f"  Average Energy: {avg_energy:.6f} kWh")

def main():
    print("=" * 70)
    print("COLLATING ENERGY METRICS FROM ALL SOURCES")
    print("=" * 70)
    
    # Process all sources
    process_local_mistral()
    process_mistral_api()  # Handles both free and paid
    process_openai_api()
    process_local_llama()
    process_llama_api()
    
    # Save individual files
    individual_files = save_individual_files()
    
    # Create combined file
    combined_file = create_combined_csv()
    
    # Print summary statistics
    print_summary_statistics()
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print("\nOutput files created:")
    print("  1. local_mistral_energy_metrics.csv")
    print("  2. mistral_free_api_energy_metrics.csv")
    print("  3. mistral_paid_api_energy_metrics.csv")
    print("  4. openai_api_energy_metrics.csv")
    print("  5. local_llama_energy_metrics.csv")
    print("  6. llama_api_energy_metrics.csv")
    print("  7. combined_energy_metrics.csv (all data)")
    print("\nTo analyze:")
    print("  - Compare free vs paid Mistral API: mistral_free_api_* vs mistral_paid_api_*")
    print("  - Compare local vs API: local_*_energy_metrics.csv vs *_api_energy_metrics.csv")
    print("  - Compare Mistral vs Llama: local_mistral_* vs local_llama_*")
    print("  - Compare GPUs: Filter by GPU column in local files")
    print("  - Overall view: combined_energy_metrics.csv")

if __name__ == "__main__":
    main()