#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/api_energy/openai_models_api.py

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import List, Tuple

from openai import OpenAI

#################################### Import carbontracker and apply the fix
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
    original_log_components_info = CarbonTrackerThread._log_components_info
    
    def fixed_log_components_info(self):
        log = ["The following components were found:"]
        for comp in self.components:
            name = comp.name.upper()
            devices = [d.decode('utf-8') if isinstance(d, bytes) else d for d in comp.devices()]
            devices = ", ".join(devices)
            log.append(f"{name} with device(s) {devices}.")
        log_str = " ".join(log)
        print(log_str)
    
    CarbonTrackerThread._log_components_info = fixed_log_components_info
    CARBONTRACKER_AVAILABLE = True
    
except (ImportError, AttributeError) as e:
    CARBONTRACKER_AVAILABLE = False
##########################################################################################

logger = logging.getLogger(__name__)


def pick_lengths(query_type: str, output_type: str,
                 max_in_short: int, max_in_long: int,
                 max_out_short: int, max_out_long: int) -> Tuple[int, int]:
    """Determine max input and output tokens based on query/output types."""
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()

    if qt not in {"short", "long"} or ot not in {"short", "long"}:
        raise ValueError(f"query_type={query_type}, output_type={output_type} must be in {{short,long}}")

    max_input = max_in_short if qt == "short" else max_in_long
    max_new = max_out_short if ot == "short" else max_out_long
    return max_input, max_new


def create_batch_file(queries_data: List[dict], args, batch_file_path: str):
    """Create JSONL file for OpenAI Batch API."""
    with open(batch_file_path, "w") as f:
        for idx, row in enumerate(queries_data):
            query = (row.get(args.text_col) or "").strip()
            qtype = (row.get(args.query_type_col) or "").strip().lower()
            otype = (row.get(args.output_type_col) or "").strip().lower()
            
            if not query:
                continue
            
            try:
                max_in, max_out = pick_lengths(
                    qtype, otype, 
                    args.max_input_tokens_short, args.max_input_tokens_long,
                    args.max_new_tokens_short, args.max_new_tokens_long
                )
                
                # Create batch request format
                batch_request = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "user", "content": query}
                        ],
                        "max_tokens": max_out,
                    }
                }
                
                # Add temperature for non-o1 models
                if not args.model.startswith("o1-"):
                    batch_request["body"]["temperature"] = args.temperature
                
                f.write(json.dumps(batch_request) + "\n")
                
            except ValueError as e:
                print(f"Skipping row {idx}: {e}")
                continue


def submit_batch_job(client: OpenAI, batch_file_path: str) -> str:
    """Upload batch file and submit batch job."""
    print("Uploading batch file...")
    
    # Upload the batch file
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"Batch file uploaded: {batch_input_file.id}")
    
    # Create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch job created: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    
    return batch_job.id


def wait_for_batch_completion(client: OpenAI, batch_id: str, poll_interval: int = 60):
    """Poll batch job until completion."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        batch_job = client.batches.retrieve(batch_id)
        status = batch_job.status
        
        print(f"Status: {status} | Completed: {batch_job.request_counts.completed}/{batch_job.request_counts.total}")
        
        if status == "completed":
            print("Batch completed successfully!")
            return batch_job
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch failed with status: {status}")
            return None
        
        time.sleep(poll_interval)


def download_batch_results(client: OpenAI, batch_job, output_file: str):
    """Download and save batch results."""
    if not batch_job or not batch_job.output_file_id:
        print("No output file available")
        return None
    
    print(f"Downloading results from: {batch_job.output_file_id}")
    
    # Download the results
    result_content = client.files.content(batch_job.output_file_id)
    result_text = result_content.text
    
    # Save raw results
    raw_output = output_file.replace(".csv", "_raw.jsonl")
    with open(raw_output, "w") as f:
        f.write(result_text)
    
    print(f"Raw results saved to: {raw_output}")
    return result_text


def process_batch_results(result_text: str, queries_data: List[dict], args, output_csv: str):
    """Process batch results and save to CSV."""
    results = {}
    
    # Parse JSONL results
    for line in result_text.strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id", "")
        idx = int(custom_id.split("-")[1]) if custom_id else -1
        
        if idx >= 0:
            response_body = result.get("response", {}).get("body", {})
            content = response_body.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response_body.get("usage", {})
            
            results[idx] = {
                "response": content,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
    
    # Write to CSV
    out_fields = ["query", "query_type", "output_type", "response", 
                  "prompt_tokens", "completion_tokens", "total_tokens"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        for idx, row in enumerate(queries_data):
            query = (row.get(args.text_col) or "").strip()
            qtype = (row.get(args.query_type_col) or "").strip().lower()
            otype = (row.get(args.output_type_col) or "").strip().lower()
            
            if not query:
                continue
            
            result = results.get(idx, {})
            
            writer.writerow({
                "query": query,
                "query_type": qtype,
                "output_type": otype,
                "response": result.get("response", ""),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            })
    
    print(f"Results saved to: {output_csv}")


def process_sequential(client: OpenAI, queries_data: List[dict], args, output_csv: str):
    """Process queries sequentially using standard API."""
    out_fields = ["query", "query_type", "output_type", "response", 
                  "prompt_tokens", "completion_tokens", "total_tokens"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        total_processed = 0
        all_metadata = []
        
        for idx, row in enumerate(queries_data):
            query = (row.get(args.text_col) or "").strip()
            qtype = (row.get(args.query_type_col) or "").strip().lower()
            otype = (row.get(args.output_type_col) or "").strip().lower()
            
            if not query:
                continue
            
            try:
                max_in, max_out = pick_lengths(
                    qtype, otype,
                    args.max_input_tokens_short, args.max_input_tokens_long,
                    args.max_new_tokens_short, args.max_new_tokens_long
                )
                
                # Call API
                start_time = time.time()
                
                if args.model.startswith("o1-"):
                    # o1 models don't support temperature
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": query}],
                        max_completion_tokens=max_out
                    )
                else:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": query}],
                        max_tokens=max_out,
                        temperature=args.temperature
                    )
                
                end_time = time.time()
                
                response_text = response.choices[0].message.content
                usage = response.usage
                
                # Write result
                writer.writerow({
                    "query": query,
                    "query_type": qtype,
                    "output_type": otype,
                    "response": response_text,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                })
                
                # Track metadata
                all_metadata.append({
                    "latency": end_time - start_time,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                })
                
                total_processed += 1
                
                if total_processed % 10 == 0:
                    progress = (total_processed / len(queries_data)) * 100
                    print(f"  Processed {total_processed}/{len(queries_data)} ({progress:.1f}%)")
                
                # Rate limiting: small delay between requests
                time.sleep(0.1)
                
            except ValueError as e:
                print(f"Skipping row {idx}: {e}")
                continue
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                writer.writerow({
                    "query": query,
                    "query_type": qtype,
                    "output_type": otype,
                    "response": f"ERROR: {str(e)}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
    
    # Save metadata
    metadata_file = output_csv.replace(".csv", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump({
            "model": args.model,
            "total_samples": len(all_metadata),
            "samples": all_metadata
        }, f, indent=2)
    
    print(f"Results saved to: {output_csv}")
    print(f"Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI API models over a CSV using Batch API.")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model name")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--text_col", type=str, default="query")
    parser.add_argument("--query_type_col", type=str, default="query_type")
    parser.add_argument("--output_type_col", type=str, default="output_type")
    parser.add_argument("--out_csv", type=str, default="openai_outputs.csv")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--use_batch", action="store_true", default=True, 
                        help="Use OpenAI Batch API (50%% cost reduction)")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between polling for batch completion")

    # Length knobs
    parser.add_argument("--max_input_tokens_short", type=int, default=2048)
    parser.add_argument("--max_input_tokens_long", type=int, default=8192)
    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--max_new_tokens_long", type=int, default=8192)

    # Sampling knobs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_samples", type=int, default=0)

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)

    # Read data
    queries_data = []
    sample_count = 0
    skipped_count = 0
    
    with open(args.csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for col in [args.text_col, args.query_type_col, args.output_type_col]:
            if col not in reader.fieldnames:
                print(f"Missing column in CSV: {col}", file=sys.stderr)
                sys.exit(1)
        
        for row in reader:
            if skipped_count < args.skip_samples:
                skipped_count += 1
                continue
                
            queries_data.append(row)
            sample_count += 1
            
            if args.max_samples is not None and sample_count >= args.max_samples:
                break

    print(f"Processing {len(queries_data)} samples with model {args.model}")

    # Initialize carbon tracker
    if CARBONTRACKER_AVAILABLE:
        tracker = CarbonTracker(epochs=1)
        tracker.epoch_start()
    else:
        tracker = None

    start_time = time.time()

    if args.use_batch:
        print("Using OpenAI Batch API (50% cost reduction)")
        
        # Create batch file
        batch_file_path = args.out_csv.replace(".csv", "_batch_input.jsonl")
        create_batch_file(queries_data, args, batch_file_path)
        
        # Submit batch job
        batch_id = submit_batch_job(client, batch_file_path)
        
        # Wait for completion
        batch_job = wait_for_batch_completion(client, batch_id, args.poll_interval)
        
        if batch_job:
            # Download and process results
            result_text = download_batch_results(client, batch_job, args.out_csv)
            if result_text:
                process_batch_results(result_text, queries_data, args, args.out_csv)
    else:
        print("Using standard OpenAI API (sequential)")
        process_sequential(client, queries_data, args, args.out_csv)

    total_time = time.time() - start_time

    if tracker:
        tracker.epoch_end()
        tracker.stop()

    print(f"\nDONE: {len(queries_data)} samples in {total_time/60:.1f} mins")
    print(f"Results: {args.out_csv}")


if __name__ == "__main__":
    main()