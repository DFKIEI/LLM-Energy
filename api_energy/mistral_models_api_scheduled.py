#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/mistral_models_api_scheduled.py

import argparse
import csv
import os
import sys
import time
from typing import Tuple, List, Dict
from datetime import datetime, timedelta

from mistralai import Mistral

# Common API model names
VALID_API_MODELS = {
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
    "codestral-latest",
}

#################################### Import carbontracker and apply the fix
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
    # Fix for carbontracker device handling
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
    print("Successfully patched carbontracker device handling")
    CARBONTRACKER_AVAILABLE = True
    
except (ImportError, AttributeError) as e:
    print(f"CarbonTracker not available or failed to patch: {e}")
    CARBONTRACKER_AVAILABLE = False


def pick_lengths(query_type: str, output_type: str,
                max_in_short: int, max_in_long: int,
                max_out_short: int, max_out_long: int) -> Tuple[int, int]:
    """
    Given query/output types and the short/long knobs, return (max_tokens_in, max_tokens_out).
    """
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()

    if qt not in {"short", "long"} or ot not in {"short", "long"}:
        raise ValueError(f"query_type={query_type}, output_type={output_type} must be in {{short,long}}")

    max_input = max_in_short if qt == "short" else max_in_long
    max_new = max_out_short if ot == "short" else max_out_long
    return max_input, max_new


def format_duration(seconds):
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_batch(client, queries_data: List[Dict], args, run_number: int, 
                  tracker=None, save_detailed: bool = True):
    """Process all queries in one batch and return metrics"""
    
    batch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING RUN #{run_number} - Processing {len(queries_data)} samples")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if save_detailed:
        print(f"Saving detailed results: YES (first run)")
    else:
        print(f"Saving detailed results: NO (metrics only)")
    print(f"{'='*70}\n")
    
    # Start carbon tracking for this batch
    if tracker:
        tracker.epoch_start()
    
    results = [] if save_detailed else None  # Only store if saving
    total_input_tokens = 0
    total_output_tokens = 0
    total_api_time = 0
    successful_calls = 0
    failed_calls = 0
    
    for idx, row in enumerate(queries_data):
        query = (row.get(args.text_col) or "").strip()
        qtype = (row.get(args.query_type_col) or "").strip().lower()
        otype = (row.get(args.output_type_col) or "").strip().lower()

        if not query:
            print(f"  [{idx+1}/{len(queries_data)}] Skipping empty query")
            continue

        try:
            max_in, max_out = pick_lengths(
                qtype, otype,
                args.max_input_tokens_short, args.max_input_tokens_long,
                args.max_new_tokens_short, args.max_new_tokens_long
            )
        except ValueError as e:
            print(f"  [{idx+1}/{len(queries_data)}] Skipping invalid row: {e}")
            continue

        # Make API call
        api_start = time.time()
        
        try:
            response = client.chat.complete(
                model=args.model,
                messages=[{"role": "user", "content": query}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_out,
                random_seed=args.seed,
            )
            
            api_end = time.time()
            api_time = api_end - api_start
            total_api_time += api_time

            # Extract response and token counts
            answer = response.choices[0].message.content if response.choices else ""
            usage = response.usage if hasattr(response, 'usage') else None
            
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            successful_calls += 1

            # Progress indicator
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(queries_data)}] ✓ {api_time:.2f}s | "
                      f"In: {input_tokens} Out: {output_tokens} | "
                      f"Progress: {(idx+1)/len(queries_data)*100:.1f}%")

        except Exception as e:
            api_end = time.time()
            api_time = api_end - api_start
            
            print(f"  [{idx+1}/{len(queries_data)}] ✗ Error: {e}")
            answer = f"ERROR: {str(e)}"
            input_tokens = output_tokens = total_tokens = 0
            failed_calls += 1

        # Store result only if saving detailed results
        if save_detailed:
            results.append({
                "run_number": run_number,
                "sample_id": idx + 1,
                "query": query,
                "query_type": qtype,
                "output_type": otype,
                "response": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "api_call_time_seconds": round(api_time, 3),
                "query_preview": query[:50]
            })
    
    # Stop carbon tracking for this batch
    if tracker:
        tracker.epoch_end()
    
    batch_end = time.time()
    batch_time = batch_end - batch_start
    
    # Calculate batch metrics
    avg_input = total_input_tokens / successful_calls if successful_calls > 0 else 0
    avg_output = total_output_tokens / successful_calls if successful_calls > 0 else 0
    avg_api_time = total_api_time / successful_calls if successful_calls > 0 else 0
    
    metrics = {
        "run_number": run_number,
        "start_time": datetime.fromtimestamp(batch_start).strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": datetime.fromtimestamp(batch_end).strftime('%Y-%m-%d %H:%M:%S'),
        "total_samples": len(queries_data),
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "total_batch_time_seconds": round(batch_time, 3),
        "total_api_time_seconds": round(total_api_time, 3),
        "avg_api_time_seconds": round(avg_api_time, 3),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_tokens": round(avg_input, 2),
        "avg_output_tokens": round(avg_output, 2),
        "tokens_per_second": round(total_output_tokens / total_api_time, 2) if total_api_time > 0 else 0
    }
    
    # Print batch summary
    print(f"\n{'='*70}")
    print(f"RUN #{run_number} COMPLETED")
    print(f"{'='*70}")
    print(f"Samples processed: {successful_calls}/{len(queries_data)}")
    print(f"Failed calls: {failed_calls}")
    print(f"Total batch time: {format_duration(batch_time)}")
    print(f"Total API time: {format_duration(total_api_time)}")
    print(f"Average API call: {avg_api_time:.2f}s")
    print(f"")
    print(f"Token Statistics:")
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    print(f"  Average input:  {avg_input:.1f} tokens/query")
    print(f"  Average output: {avg_output:.1f} tokens/query")
    print(f"  Throughput: {total_output_tokens / total_api_time:.1f} tokens/second")
    print(f"{'='*70}\n")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Run Mistral API in scheduled batches with metrics tracking.")
    parser.add_argument("--model", type=str, required=True,
                        help="Mistral API model name")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with queries")
    parser.add_argument("--text_col", type=str, default="query",
                        help="CSV column for the input text")
    parser.add_argument("--query_type_col", type=str, default="query_type",
                        help='CSV column for query type: "short" | "long"')
    parser.add_argument("--output_type_col", type=str, default="output_type",
                        help='CSV column for output type: "short" | "long"')
    parser.add_argument("--out_csv", type=str, default="mistral_api_outputs.csv",
                        help="Where to save detailed outputs (CSV) - only for first run")
    parser.add_argument("--metrics_csv", type=str, default="mistral_api_metrics.csv",
                        help="Where to save batch metrics (CSV)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("MISTRAL_API_KEY", ""),
                        help="Mistral API key")

    # Length knobs
    parser.add_argument("--max_input_tokens_short", type=int, default=2048)
    parser.add_argument("--max_input_tokens_long", type=int, default=8192)
    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--max_new_tokens_long", type=int, default=8192)

    # Sampling knobs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Scheduling parameters
    parser.add_argument("--batch_interval_minutes", type=int, default=30,
                        help="Minutes to wait between batches (default: 30)")
    parser.add_argument("--num_runs", type=int, default=None,
                        help="Number of runs to complete (default: run until time limit)")
    parser.add_argument("--max_runtime_hours", type=int, default=72,
                        help="Maximum runtime in hours (default: 72 = 3 days)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed run")
    parser.add_argument("--save_all_detailed", action="store_true",
                        help="Save detailed results for ALL runs (default: only first run)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process per batch")
    parser.add_argument("--skip_samples", type=int, default=0,
                        help="Number of samples to skip from the beginning")
    
    args = parser.parse_args()

    # Validate model
    if args.model not in VALID_API_MODELS:
        print(f"Warning: '{args.model}' not in known API models: {VALID_API_MODELS}")
        print("Proceeding anyway, but verify the model name is correct.")

    # Check CSV exists
    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Initialize Mistral client
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("No Mistral API key provided. Set --api_key or MISTRAL_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    client = Mistral(api_key=api_key)
    print(f"Initialized Mistral API client for model: {args.model}")

    # Read CSV data (ONCE - same queries for all runs)
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

    print(f"Loaded {len(queries_data)} queries from CSV (same queries for all runs)")
    
    # Calculate scheduling info
    max_runtime_seconds = args.max_runtime_hours * 3600
    batch_interval_seconds = args.batch_interval_minutes * 60
    
    # Estimate number of runs if not specified
    if args.num_runs is None:
        # Rough estimate: assume each batch takes 5 minutes
        estimated_batch_time = 300  # 5 minutes in seconds
        time_per_cycle = estimated_batch_time + batch_interval_seconds
        estimated_runs = int(max_runtime_seconds / time_per_cycle)
        print(f"Estimated ~{estimated_runs} runs in {args.max_runtime_hours} hours")
        print(f"(Each cycle: ~5 min processing + {args.batch_interval_minutes} min wait)")
    
    print(f"\n{'='*70}")
    print(f"BATCH SCHEDULING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Samples per batch: {len(queries_data)}")
    print(f"Interval between batches: {args.batch_interval_minutes} minutes")
    print(f"Maximum runtime: {args.max_runtime_hours} hours")
    if args.num_runs:
        print(f"Target number of runs: {args.num_runs}")
        total_time = args.num_runs * (5 + args.batch_interval_minutes)  # rough estimate
        print(f"Estimated total time: {total_time/60:.1f} hours")
    print(f"")
    print(f"Detailed Results Policy:")
    if args.save_all_detailed:
        print(f"  ✓ Saving detailed results for ALL runs")
    else:
        print(f"  ✓ Saving detailed results for FIRST RUN ONLY")
        print(f"  ✓ Subsequent runs: metrics only")
    print(f"{'='*70}\n")

    # Check if resuming
    start_run = 1
    if args.resume and os.path.isfile(args.metrics_csv):
        with open(args.metrics_csv, "r", newline="", encoding="utf-8") as f:
            existing_runs = sum(1 for _ in csv.DictReader(f))
            start_run = existing_runs + 1
            print(f"Resuming from run #{start_run} (found {existing_runs} completed runs)")

    # Initialize carbon tracker (for batches)
    if CARBONTRACKER_AVAILABLE:
        batch_tracker = CarbonTracker(epochs=1)
        print("Carbon tracking initialized (will track each batch separately)")
    else:
        batch_tracker = None
        print("Warning: Carbon tracking not available")

    # Prepare output files
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    
    # CSV field names
    detail_fields = ["run_number", "sample_id", "query", "query_type", "output_type", 
                     "response", "input_tokens", "output_tokens", "total_tokens", 
                     "api_call_time_seconds", "query_preview"]
    
    metrics_fields = ["run_number", "start_time", "end_time", "total_samples", 
                      "successful_calls", "failed_calls", "total_batch_time_seconds",
                      "total_api_time_seconds", "avg_api_time_seconds",
                      "total_input_tokens", "total_output_tokens", "total_tokens",
                      "avg_input_tokens", "avg_output_tokens", "tokens_per_second"]
    
    # Start experiment
    experiment_start = time.time()
    run_number = start_run
    
    # Open metrics file (always append or create)
    metrics_mode = "a" if args.resume else "w"
    
    with open(args.metrics_csv, metrics_mode, newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        
        # Write header if new file
        if metrics_mode == "w":
            metrics_writer.writeheader()
        
        # Detailed file handle (only for first run or if save_all_detailed)
        detail_file = None
        detail_writer = None
        
        # Main loop
        while True:
            # Check if we should stop
            elapsed_time = time.time() - experiment_start
            if elapsed_time >= max_runtime_seconds:
                print(f"\nReached maximum runtime of {args.max_runtime_hours} hours. Stopping.")
                break
            
            if args.num_runs and run_number > args.num_runs:
                print(f"\nCompleted {args.num_runs} runs. Stopping.")
                break
            
            # Determine if we should save detailed results
            save_detailed = (run_number == 1) or args.save_all_detailed
            
            # Open detailed file if needed
            if save_detailed and detail_file is None:
                detail_mode = "a" if (args.resume and run_number > 1) else "w"
                detail_file = open(args.out_csv, detail_mode, newline="", encoding="utf-8")
                detail_writer = csv.DictWriter(detail_file, fieldnames=detail_fields)
                if detail_mode == "w":
                    detail_writer.writeheader()
            
            # Process one batch
            results, metrics = process_batch(
                client, 
                queries_data, 
                args, 
                run_number,
                batch_tracker,
                save_detailed=save_detailed
            )
            
            # Write detailed results if available
            if results and detail_writer:
                for result in results:
                    detail_writer.writerow(result)
                detail_file.flush()
            
            # Always write metrics
            metrics_writer.writerow(metrics)
            metrics_file.flush()
            
            # Close detailed file after first run (unless save_all_detailed)
            if detail_file and not args.save_all_detailed and run_number == 1:
                detail_file.close()
                detail_file = None
                detail_writer = None
                print(f"\n✓ Detailed results saved to: {args.out_csv}")
                print(f"  (Subsequent runs will only save metrics)")
            
            # Calculate remaining time
            elapsed_hours = elapsed_time / 3600
            remaining_hours = args.max_runtime_hours - elapsed_hours
            
            # Wait before next batch (unless this is the last run)
            should_continue = True
            if args.num_runs:
                should_continue = run_number < args.num_runs
            else:
                # Check if we have time for another cycle
                time_for_next_cycle = batch_interval_seconds + 600  # interval + ~10 min buffer
                should_continue = (elapsed_time + time_for_next_cycle) < max_runtime_seconds
            
            if should_continue:
                next_run_time = datetime.now() + timedelta(seconds=batch_interval_seconds)
                
                print(f"\n{'='*70}")
                print(f"WAITING {args.batch_interval_minutes} MINUTES BEFORE NEXT RUN")
                print(f"{'='*70}")
                print(f"Current run: #{run_number}")
                print(f"Next run: #{run_number + 1} at {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Elapsed: {format_duration(elapsed_time)} / {args.max_runtime_hours}h")
                print(f"Remaining: {format_duration(remaining_hours * 3600)}")
                print(f"{'='*70}\n")
                
                time.sleep(batch_interval_seconds)
            else:
                print(f"\nNo time for another complete cycle. Stopping.")
                break
            
            run_number += 1
        
        # Close detailed file if still open
        if detail_file:
            detail_file.close()
    
    # Final summary
    experiment_end = time.time()
    total_experiment_time = experiment_end - experiment_start
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED")
    print(f"{'='*70}")
    print(f"Total runs completed: {run_number - start_run}")
    print(f"Total experiment time: {format_duration(total_experiment_time)}")
    print(f"")
    print(f"Output files:")
    if start_run == 1 or args.save_all_detailed:
        print(f"  Detailed results: {args.out_csv}")
    print(f"  Batch metrics: {args.metrics_csv}")
    if batch_tracker:
        print(f"  Carbon logs: ./logs/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()