#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    print("Successfully patched carbontracker device handling")
    CARBONTRACKER_AVAILABLE = True
    
except (ImportError, AttributeError) as e:
    print(f"Failed to set up carbontracker: {e}")
    CARBONTRACKER_AVAILABLE = False
##########################################################################################


def load_model(model_name: str, cache_dir: str = "/netscratch/banwari/.cache/huggingface", token: str = None):
    """Load model with automatic device mapping (layer parallelism)."""
    print(f"Loading model: {model_name}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token
    )
    
    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Set pad_token to eos_token")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatic layer parallelism across GPUs
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token
    )
    
    print(f"✓ Model loaded with device_map='auto' (layer parallelism)")
    print(f"\nModel device map:")
    print(model.hf_device_map)
    
    # Show distribution
    if hasattr(model, 'hf_device_map'):
        device_counts = {}
        for layer, device in model.hf_device_map.items():
            device_counts[device] = device_counts.get(device, 0) + 1
        
        print("\nLayer distribution:")
        for device, count in sorted(device_counts.items()):
            print(f"  {device}: {count} layers")
    
    return model, tokenizer


def pick_lengths(query_type: str, output_type: str,
                 max_in_short: int, max_in_long: int,
                 max_out_short: int, max_out_long: int) -> Tuple[int, int]:
    """Determine input and output token lengths based on query/output types."""
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()

    if qt not in {"short", "long"} or ot not in {"short", "long"}:
        raise ValueError(f"query_type={query_type}, output_type={output_type} must be in {{short,long}}")

    max_input = max_in_short if qt == "short" else max_in_long
    max_new = max_out_short if ot == "short" else max_out_long
    return max_input, max_new


def generate_response(model, tokenizer, query: str, max_new_tokens: int,
                     temperature: float, top_p: float, top_k: int,
                     repetition_penalty: float, min_new_tokens: int = None,
                     do_sample: bool = True):
    """Generate response for a single query."""
    
    # Tokenize input
    inputs = tokenizer(query, return_tensors="pt", max_length=8192, truncation=True)
    
    # Move to cuda:0 (first GPU where model starts)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens or int(max_new_tokens * 0.5),
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "top_k": top_k if do_sample else 50,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode (skip input tokens)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Run models with layer parallelism over CSV queries.")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3",
                        help="HF model id (default: mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with queries")
    parser.add_argument("--text_col", type=str, default="query",
                        help="CSV column for the input text")
    parser.add_argument("--query_type_col", type=str, default="query_type",
                        help='CSV column for query type: "short" | "long"')
    parser.add_argument("--output_type_col", type=str, default="output_type",
                        help='CSV column for output type: "short" | "long"')
    parser.add_argument("--out_csv", type=str, default="layer_parallel_outputs.csv",
                        help="Where to save outputs (CSV)")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""),
                        help="HF access token for gated models")

    # Length knobs
    parser.add_argument("--max_input_tokens_short", type=int, default=2048)
    parser.add_argument("--max_input_tokens_long", type=int, default=8192)
    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--max_new_tokens_long", type=int, default=8192)

    # Sampling knobs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Use sampling (default: True)")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--cache_dir", type=str, default="/netscratch/banwari/.cache/huggingface",
                        help="HuggingFace cache directory")

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Set token
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    # Load model with layer parallelism
    model, tokenizer = load_model(args.model, cache_dir=args.cache_dir, token=token)

    # Read queries
    queries_data = []
    with open(args.csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for col in [args.text_col, args.query_type_col, args.output_type_col]:
            if col not in reader.fieldnames:
                print(f"Missing column in CSV: {col}", file=sys.stderr)
                sys.exit(1)
        
        for i, row in enumerate(reader):
            if args.max_samples and i >= args.max_samples:
                break
            queries_data.append(row)
    
    print(f"\nProcessing {len(queries_data)} samples")

    # Prepare queries with their parameters
    queries_with_params = []
    for idx, row in enumerate(queries_data):
        query = (row.get(args.text_col) or "").strip()
        qtype = (row.get(args.query_type_col) or "").strip().lower()
        otype = (row.get(args.output_type_col) or "").strip().lower()
        
        if not query:
            continue
        
        max_in, max_out = pick_lengths(
            qtype, otype,
            args.max_input_tokens_short, args.max_input_tokens_long,
            args.max_new_tokens_short, args.max_new_tokens_long
        )
        
        queries_with_params.append((query, qtype, otype, max_out, idx))
    
    print(f"Valid queries to process: {len(queries_with_params)}")

    # Initialize CarbonTracker
    if CARBONTRACKER_AVAILABLE:
        tracker = CarbonTracker(epochs=1, components="gpu", log_dir="./carbontracker_layer_parallel_logs")
        tracker.epoch_start()
        print("✓ CarbonTracker started\n")

    # Store all results
    all_results = [None] * len(queries_data)
    start_time = time.time()
    
    # Process each query
    print("Starting generation...")
    for i, (query, qtype, otype, max_out, orig_idx) in enumerate(queries_with_params):
        query_start = time.time()
        
        response = generate_response(
            model, tokenizer, query,
            max_new_tokens=max_out,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample
        )
        
        query_time = time.time() - query_start
        num_tokens = len(tokenizer.encode(response))
        
        all_results[orig_idx] = {
            "query": query,
            "query_type": qtype,
            "output_type": otype,
            "response": response,
            "output_tokens": num_tokens
        }
        
        print(f"  [{i+1}/{len(queries_with_params)}] {qtype}/{otype} -> {num_tokens} tokens in {query_time:.1f}s")
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            print(f"  Progress: {i+1}/{len(queries_with_params)} | Avg: {avg_time:.1f}s/query | ETA: {avg_time * (len(queries_with_params) - i - 1) / 60:.1f} min\n")

    # Stop CarbonTracker
    if CARBONTRACKER_AVAILABLE:
        tracker.epoch_end()
        tracker.stop()
        print("\n✓ CarbonTracker stopped")

    total_time = time.time() - start_time

    # Write outputs in original order
    out_fields = ["query", "query_type", "output_type", "response", "output_tokens"]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    
    total_tokens = 0
    
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        for result in all_results:
            if result:
                writer.writerow(result)
                total_tokens += result["output_tokens"]

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"GPUs used: {torch.cuda.device_count()}")
    print(f"Queries processed: {len([r for r in all_results if r])}")
    print(f"Total output tokens: {total_tokens:,}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time per query: {total_time/len(queries_with_params):.1f}s")
    print(f"Throughput: {len([r for r in all_results if r])/total_time:.2f} queries/sec")
    print(f"Output file: {args.out_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()