import os
import re
import pandas as pd
from tqdm import tqdm
import glob

# === Configuration ===
base_dir = "/home/banwari/llm_energy/LLM-Energy/api_energy"

# Define the three main directories
dirs = {
    "local_mistral": os.path.join(base_dir, "results_parallel_tokens_deterministic"),
    "mistral_api": os.path.join(base_dir, "results_api"),
    "openai_api": os.path.join(base_dir, "results_openai_api")
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
    "openai_api": re.compile(r"(?P<model>[^_]+)_openai_api_run(?P<run>\d+)_\d+\.out")
}

# Store rows for each source separately
data = {
    "local_mistral": [],
    "mistral_api": [],
    "openai_api": []
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
    """Process results from local Mistral models (results_parallel)"""
    print("\nProcessing Local Mistral Models (results_parallel)...")
    
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
    """Process results from Mistral API (results_api)"""
    print("\nProcessing Mistral API Models (results_api)...")
    
    dir_path = dirs["mistral_api"]
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    pattern = file_patterns["mistral_api"]
    files = [f for f in os.listdir(dir_path) if f.endswith(".out")]
    
    for file in tqdm(files, desc="Mistral API"):
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
            data["mistral_api"].append({
                "Type": "API",
                "Model": model,
                "GPU": "N/A",
                "Run": run_id,
                "Metric": metric,
                "Value": value
            })

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


def main():
    print("=" * 70)
    print("COLLATING ENERGY METRICS FROM ALL SOURCES")
    print("=" * 70)
    
    # Process all three sources
    process_local_mistral()
    process_mistral_api()
    process_openai_api()
    
    # Save individual files
    individual_files = save_individual_files()
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print("\nOutput files created:")
    print("  1. local_mistral_energy_metrics.csv")
    print("  2. mistral_api_energy_metrics.csv")
    print("  3. openai_api_energy_metrics.csv")

if __name__ == "__main__":
    main()