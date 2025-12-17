#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time
from typing import Tuple

from groq import Groq

# Valid Groq API models
VALID_API_MODELS = {
    "llama-3.1-8b-instant",
    "meta-llama/llama-guard-4-12b",
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


def main():
    parser = argparse.ArgumentParser(description="Run Groq API (Llama models) over a CSV of queries with per-sample tracking.")
    parser.add_argument("--model", type=str, required=True,
                        help="Groq API model name (e.g., llama-3.1-8b-instant)")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with queries")
    parser.add_argument("--text_col", type=str, default="query",
                        help="CSV column for the input text")
    parser.add_argument("--query_type_col", type=str, default="query_type",
                        help='CSV column for query type: "short" | "long"')
    parser.add_argument("--output_type_col", type=str, default="output_type",
                        help='CSV column for output type: "short" | "long"')
    parser.add_argument("--out_csv", type=str, default="llama_api_outputs.csv",
                        help="Where to save outputs (CSV)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GROQ_API_KEY", ""),
                        help="Groq API key")

    # Length knobs
    parser.add_argument("--max_input_tokens_short", type=int, default=2048)
    parser.add_argument("--max_input_tokens_long", type=int, default=8192)
    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--max_new_tokens_long", type=int, default=8192)

    # Sampling knobs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--skip_samples", type=int, default=0,
                        help="Number of samples to skip from the beginning")
    
    args = parser.parse_args()

    # Validate model
    if args.model not in VALID_API_MODELS:
        print(f"Warning: '{args.model}' not in known Groq API models: {VALID_API_MODELS}")
        print("Proceeding anyway, but verify the model name is correct.")

    # Check CSV exists
    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Initialize Groq client
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No Groq API key provided. Set --api_key or GROQ_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"Initialized Groq API client for model: {args.model}")

    # Read CSV data
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

    print(f"Loaded {len(queries_data)} queries from CSV")
    print(f"Processing with per-sample carbon tracking...")

    # Initialize overall carbon tracker
    if CARBONTRACKER_AVAILABLE:
        overall_tracker = CarbonTracker(epochs=1, log_dir="./carbon_logs_overall")
        overall_tracker.epoch_start()
    else:
        overall_tracker = None
        print("Warning: Overall carbon tracking not available")

    # Process each query with individual tracking
    out_fields = ["sample_id", "query", "query_type", "output_type", "response", 
                  "input_tokens", "output_tokens", "total_tokens", 
                  "time_seconds", "carbon_kg_sample"]
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        for idx, row in enumerate(queries_data):
            query = (row.get(args.text_col) or "").strip()
            qtype = (row.get(args.query_type_col) or "").strip().lower()
            otype = (row.get(args.output_type_col) or "").strip().lower()

            if not query:
                print(f"Skipping empty query at index {idx}")
                continue

            try:
                max_in, max_out = pick_lengths(
                    qtype, otype,
                    args.max_input_tokens_short, args.max_input_tokens_long,
                    args.max_new_tokens_short, args.max_new_tokens_long
                )
            except ValueError as e:
                print(f"Skipping row {idx}: {e}")
                continue

            print(f"\n[Sample {idx+1}/{len(queries_data)}] Processing: {qtype} query, {otype} output")
            print(f"  Max input tokens: {max_in}, Max output tokens: {max_out}")

            # Start per-sample carbon tracking
            sample_tracker = None
            if CARBONTRACKER_AVAILABLE:
                try:
                    sample_tracker = CarbonTracker(
                        epochs=1, 
                        log_dir=f"./carbon_logs_sample_{idx+1}",
                        verbose=0  # Reduce verbosity for per-sample tracking
                    )
                    sample_tracker.epoch_start()
                except Exception as e:
                    print(f"  Warning: Could not start sample tracker: {e}")
                    sample_tracker = None

            # Make API call
            start_time = time.time()
            
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": query}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_out,
                    seed=args.seed,
                )
                
                end_time = time.time()
                elapsed = end_time - start_time

                # Extract response and token counts
                answer = response.choices[0].message.content if response.choices else ""
                usage = response.usage if hasattr(response, 'usage') else None
                
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

                print(f"  ✓ Completed in {elapsed:.2f}s")
                print(f"  Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")

            except Exception as e:
                print(f"  ✗ API error: {e}")
                answer = f"ERROR: {str(e)}"
                input_tokens = output_tokens = total_tokens = 0
                elapsed = time.time() - start_time

            # Stop per-sample carbon tracking
            carbon_kg_sample = 0.0
            if sample_tracker:
                try:
                    sample_tracker.epoch_end()
                    sample_tracker.stop()
                    
                    # Try to get the carbon consumption for this sample
                    # Note: CarbonTracker writes to logs, we'll need to parse or estimate
                    # For now, we'll mark it as tracked
                    carbon_kg_sample = -1.0  # Placeholder: indicates tracking was done
                    print(f"  Carbon tracking completed (see carbon_logs_sample_{idx+1}/)")
                except Exception as e:
                    print(f"  Warning: Could not complete sample tracker: {e}")

            # Write result
            writer.writerow({
                "sample_id": idx + 1,
                "query": query,
                "query_type": qtype,
                "output_type": otype,
                "response": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "time_seconds": elapsed,
                "carbon_kg_sample": carbon_kg_sample
            })
            
            # Flush after each sample
            fout.flush()

    # Stop overall carbon tracker
    if overall_tracker:
        overall_tracker.epoch_end()
        overall_tracker.stop()
        print("\nOverall carbon tracking completed (see carbon_logs_overall/)")

    print(f"\nDone. Wrote: {args.out_csv}")
    print(f"Total samples processed: {len(queries_data)}")
    print(f"Per-sample carbon logs: ./carbon_logs_sample_*/")
    print(f"Overall carbon logs: ./carbon_logs_overall/")


if __name__ == "__main__":
    main()