#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from mistralai import Mistral

# Common API model names
VALID_API_MODELS = [
    "open-mistral-7b",
    "open-mixtral-8x7b", 
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
    "codestral-latest",
]

def format_duration(seconds):
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def pick_lengths(query_type, output_type,
                 max_in_short, max_in_long,
                 max_out_short, max_out_long):
    """
    Given query/output types and the short/long knobs, return (max_tokens_in, max_tokens_out).
    """
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()

    if qt not in {"short", "long"} or ot not in {"short", "long"}:
        return max_in_short, max_out_short  # default to short

    max_input = max_in_short if qt == "short" else max_in_long
    max_new = max_out_short if ot == "short" else max_out_long
    return max_input, max_new


def run_experiment(model, api_key, csv_file, output_name, max_samples=100, 
                   runtime_days=7, temperature=0.7, top_p=0.95,
                   max_in_short=2048, max_in_long=8192,
                   max_out_short=2048, max_out_long=8192,
                   query_type_col="query_type", output_type_col="output_type",
                   query_col="query"):
    """Run continuous API calls for specified days"""
    
    # Initialize client
    client = Mistral(api_key=api_key)
    
    # Load queries (first max_samples)
    queries_data = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            queries_data.append({
                'query': row.get(query_col, "").strip(),
                'query_type': row.get(query_type_col, "short").strip(),
                'output_type': row.get(output_type_col, "short").strip(),
            })
    
    if not queries_data:
        print(f"No queries loaded from {csv_file}")
        return
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT STARTED: {model}")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Queries loaded: {len(queries_data)}")
    print(f"Runtime: {runtime_days} days")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Token limits: Input short={max_in_short}, long={max_in_long}")
    print(f"             Output short={max_out_short}, long={max_out_long}")
    print(f"Output name: {output_name}")
    print(f"End time: {(datetime.now() + timedelta(days=runtime_days)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Setup output files
    results_file = f"{output_name}_results.csv"
    metrics_file = f"{output_name}_metrics.csv"
    
    # Initialize results CSV
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "start_time", "end_time", "duration_seconds",
                        "query_type", "output_type", "max_input_tokens", "max_output_tokens",
                        "query", "response_preview", "input_tokens", "output_tokens", 
                        "total_tokens", "error"])
    
    # Initialize metrics CSV
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "queries_processed", "total_tokens", 
                        "avg_duration", "errors", "uptime_hours"])
    
    # Run for specified days
    end_time = time.time() + (runtime_days * 24 * 3600)
    query_count = 0
    total_tokens = 0
    total_duration = 0
    error_count = 0
    start_time = time.time()
    
    print(f"Starting query processing...\n")
    
    while time.time() < end_time:
        query_data = queries_data[query_count % len(queries_data)]
        query = query_data['query']
        query_type = query_data['query_type']
        output_type = query_data['output_type']
        
        # Determine max tokens based on query/output type
        max_input_tokens, max_output_tokens = pick_lengths(
            query_type, output_type,
            max_in_short, max_in_long,
            max_out_short, max_out_long
        )
        
        # Record start time
        call_start = datetime.now()
        call_start_ts = time.time()
        
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_output_tokens
            )
            
            # Record end time
            call_end = datetime.now()
            duration = time.time() - call_start_ts
            total_duration += duration
            
            # Extract data
            answer = response.choices[0].message.content if response.choices else ""
            usage = response.usage if hasattr(response, 'usage') else None
            
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            tokens = usage.total_tokens if usage else 0
            total_tokens += tokens
            
            error_msg = ""
            
        except Exception as e:
            call_end = datetime.now()
            duration = time.time() - call_start_ts
            total_duration += duration
            
            answer = ""
            input_tokens = output_tokens = tokens = 0
            error_msg = str(e)
            error_count += 1
            print(f"  Error on query {query_count}: {e}")
        
        # Save result
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                query_count,
                call_start.strftime('%Y-%m-%d %H:%M:%S'),
                call_end.strftime('%Y-%m-%d %H:%M:%S'),
                round(duration, 3),
                query_type,
                output_type,
                max_input_tokens,
                max_output_tokens,
                query[:100],
                answer[:100] if answer else "",
                input_tokens,
                output_tokens,
                tokens,
                error_msg
            ])
        
        # Progress update every 10 queries
        if query_count % 10 == 0:
            remaining_hours = (end_time - time.time()) / 3600
            uptime_hours = (time.time() - start_time) / 3600
            avg_dur = total_duration / (query_count + 1)
            cycle = query_count // len(queries_data) + 1
            
            print(f"Query {query_count} (Cycle {cycle}) | "
                  f"Type: {query_type}/{output_type} | "
                  f"Duration: {duration:.2f}s | Tokens: {tokens} | "
                  f"Remaining: {remaining_hours:.1f}h")
            
            # Save metrics snapshot
            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    query_count + 1,
                    total_tokens,
                    round(avg_dur, 3),
                    error_count,
                    round(uptime_hours, 2)
                ])
        
        query_count += 1
        time.sleep(1)  # 1 second between queries
    
    # Final summary
    total_time = time.time() - start_time
    avg_duration = total_duration / query_count if query_count > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED: {model}")
    print(f"{'='*70}")
    print(f"Total queries: {query_count}")
    print(f"Total cycles: {query_count // len(queries_data)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average duration: {avg_duration:.2f}s")
    print(f"Errors: {error_count}")
    print(f"Total runtime: {format_duration(total_time)}")
    print(f"Results: {results_file}")
    print(f"Metrics: {metrics_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Mistral API locally for 7 days with multiple models")
    
    # Required
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file with queries")
    
    # Model selection (can specify multiple)
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["open-mistral-7b"],
                        choices=VALID_API_MODELS,
                        help="Model(s) to test (space-separated)")
    
    # API key
    parser.add_argument("--api_key", type=str, 
                        default=os.environ.get("MISTRAL_API_KEY"),
                        help="Mistral API key")
    
    # Output name
    parser.add_argument("--output_name", type=str, default="mistral_experiment",
                        help="Base name for output files (will add _results.csv and _metrics.csv)")
    
    # CSV column names
    parser.add_argument("--query_col", type=str, default="query",
                        help="Column name for queries (default: query)")
    parser.add_argument("--query_type_col", type=str, default="query_type",
                        help="Column name for query type (short/long)")
    parser.add_argument("--output_type_col", type=str, default="output_type",
                        help="Column name for output type (short/long)")
    
    # Token limits
    parser.add_argument("--max_input_tokens_short", type=int, default=2048,
                        help="Max input tokens for short queries")
    parser.add_argument("--max_input_tokens_long", type=int, default=8192,
                        help="Max input tokens for long queries")
    parser.add_argument("--max_new_tokens_short", type=int, default=2048,
                        help="Max output tokens for short responses")
    parser.add_argument("--max_new_tokens_long", type=int, default=8192,
                        help="Max output tokens for long responses")
    
    # Parameters
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples to load from CSV (default: 100)")
    parser.add_argument("--runtime_days", type=int, default=7,
                        help="Runtime in days (default: 7)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)
    
    if not args.api_key:
        print("No API key provided. Use --api_key or set MISTRAL_API_KEY")
        sys.exit(1)
    
    # Run experiments for each model
    print(f"\n{'='*70}")
    print(f"STARTING LOCAL 7-DAY EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Total experiments: {len(args.models)}")
    print(f"Token limits configured:")
    print(f"  Input:  short={args.max_input_tokens_short}, long={args.max_input_tokens_long}")
    print(f"  Output: short={args.max_new_tokens_short}, long={args.max_new_tokens_long}")
    print(f"{'='*70}\n")
    
    for model in args.models:
        # Create output name with model
        output_name = f"{args.output_name}_{model.replace('-', '_')}"
        
        print(f"\n>>> Starting experiment: {model}")
        
        try:
            run_experiment(
                model=model,
                api_key=args.api_key,
                csv_file=args.csv,
                output_name=output_name,
                max_samples=args.max_samples,
                runtime_days=args.runtime_days,
                temperature=args.temperature,
                top_p=args.top_p,
                max_in_short=args.max_input_tokens_short,
                max_in_long=args.max_input_tokens_long,
                max_out_short=args.max_new_tokens_short,
                max_out_long=args.max_new_tokens_long,
                query_type_col=args.query_type_col,
                output_type_col=args.output_type_col,
                query_col=args.query_col
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Stopping all experiments.")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR in experiment {model}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()