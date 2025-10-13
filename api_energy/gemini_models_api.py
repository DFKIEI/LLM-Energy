#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/api_energy/gemini_models_api.py

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import List, Tuple

from google import genai
from google.genai import types

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


def call_gemini_api(client, model_name: str, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, dict]:
    """Call Gemini API and return response with metadata."""
    try:
        start_time = time.time()
        
        # Configure generation parameters with safety settings disabled
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            # Disable thinking for faster responses
            thinking_config=types.ThinkingConfig(thinking_budget=0) if "2.5" in model_name else None,
            # Relax safety settings to avoid blocking
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
            ]
        )
        
        # Generate content
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        
        end_time = time.time()
        
        # Extract response text safely
        try:
            response_text = response.text if hasattr(response, 'text') and response.text else ""
        except Exception as e:
            # If text access fails, check candidates
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else "UNKNOWN"
                
                # Map finish_reason codes to names
                finish_reason_map = {
                    0: "FINISH_REASON_UNSPECIFIED",
                    1: "STOP",
                    2: "SAFETY",
                    3: "RECITATION",
                    4: "OTHER"
                }
                finish_reason_name = finish_reason_map.get(finish_reason, str(finish_reason))
                
                response_text = f"[BLOCKED: {finish_reason_name}]"
            else:
                response_text = "[NO_RESPONSE]"
        
        # Extract token usage if available
        try:
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
        except:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        
        # Get finish reason
        try:
            if response.candidates and len(response.candidates) > 0:
                finish_reason = response.candidates[0].finish_reason
                finish_reason_map = {0: "UNSPECIFIED", 1: "STOP", 2: "SAFETY", 3: "RECITATION", 4: "OTHER"}
                finish_reason_name = finish_reason_map.get(finish_reason, str(finish_reason))
            else:
                finish_reason_name = "NO_CANDIDATES"
        except:
            finish_reason_name = "UNKNOWN"
        
        metadata = {
            "latency": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason_name
        }
        
        return response_text, metadata
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"ERROR: {str(e)}", {"error": str(e), "latency": 0}


def process_sequential(client, model_name: str, queries_data: List[dict], args, output_csv: str):
    """Process queries sequentially using Gemini API."""
    out_fields = ["query", "query_type", "output_type", "response", 
                  "prompt_tokens", "completion_tokens", "total_tokens", "latency", "finish_reason"]
    
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
                
                # Call Gemini API
                response_text, metadata = call_gemini_api(
                    client, model_name, query, max_out, args.temperature
                )
                
                # Write result
                writer.writerow({
                    "query": query,
                    "query_type": qtype,
                    "output_type": otype,
                    "response": response_text,
                    "prompt_tokens": metadata.get("prompt_tokens", 0),
                    "completion_tokens": metadata.get("completion_tokens", 0),
                    "total_tokens": metadata.get("total_tokens", 0),
                    "latency": metadata.get("latency", 0),
                    "finish_reason": metadata.get("finish_reason", "UNKNOWN")
                })
                
                # Track metadata
                all_metadata.append(metadata)
                
                total_processed += 1
                
                if total_processed % 10 == 0:
                    progress = (total_processed / len(queries_data)) * 100
                    print(f"  Processed {total_processed}/{len(queries_data)} ({progress:.1f}%)")
                
                # Rate limiting: small delay between requests
                time.sleep(0.5)
                
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
                    "total_tokens": 0,
                    "latency": 0,
                    "finish_reason": "ERROR"
                })
                time.sleep(2)
    
    # Save metadata
    metadata_file = output_csv.replace(".csv", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump({
            "model": model_name,
            "total_samples": len(all_metadata),
            "samples": all_metadata
        }, f, indent=2)
    
    print(f"Results saved to: {output_csv}")
    print(f"Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Google Gemini API models over a CSV.")
    parser.add_argument("--model", type=str, required=True, help="Gemini model name")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--text_col", type=str, default="query")
    parser.add_argument("--query_type_col", type=str, default="query_type")
    parser.add_argument("--output_type_col", type=str, default="output_type")
    parser.add_argument("--out_csv", type=str, default="gemini_outputs.csv")
    parser.add_argument("--api_key", type=str, required=True, help="Google API key")

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

    # Initialize Gemini client (new API for Gemini 2.5)
    client = genai.Client(api_key=args.api_key)
    
    print(f"Using Gemini model: {args.model}")

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

    # Process sequentially
    print("Using Gemini API (sequential processing)")
    process_sequential(client, args.model, queries_data, args, args.out_csv)

    total_time = time.time() - start_time

    if tracker:
        tracker.epoch_end()
        tracker.stop()

    print(f"\nDONE: {len(queries_data)} samples in {total_time/60:.1f} mins")
    print(f"Results: {args.out_csv}")


if __name__ == "__main__":
    main()