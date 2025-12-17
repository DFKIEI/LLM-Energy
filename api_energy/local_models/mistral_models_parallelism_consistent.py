#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time
import random
import numpy as np
from typing import List, Tuple

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


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic (may slow down by 5-10%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For Python's hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Set all seeds to {seed} for reproducibility")


def load_model(model_name: str, dtype: str = "fp16", token: str = None):
    """Load model with optimizations for GPU efficiency."""
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"Loading model: {model_name} [{dtype}]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for batch generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    model.eval()
    
    # Enable TF32 for faster computation on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 for faster computation")
    
    # Enable optimizations
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Model loaded on device: {model.device}")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def pick_lengths(query_type: str, output_type: str,
                 max_in_short: int, max_in_long: int,
                 max_out_short: int, max_out_long: int) -> Tuple[int, int]:
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()

    if qt not in {"short", "long"} or ot not in {"short", "long"}:
        raise ValueError(f"query_type={query_type}, output_type={output_type} must be in {{short,long}}")

    max_input = max_in_short if qt == "short" else max_in_long
    max_new = max_out_short if ot == "short" else max_out_long
    return max_input, max_new


def generate_batch_parallel(model, tokenizer, prompts: List[str], max_input_tokens: int, 
                            max_new_tokens: int, temperature: float, top_p: float, 
                            top_k: int, repetition_penalty: float) -> Tuple[List[str], List[int], List[int]]:
    """
    Generate responses for multiple prompts in parallel using true batching.
    Returns: (responses, input_token_counts, output_token_counts)
    """
    if not prompts or all(not p for p in prompts):
        return [""] * len(prompts), [0] * len(prompts), [0] * len(prompts)
    
    # Filter out empty prompts but keep track of indices
    valid_prompts = [(i, p) for i, p in enumerate(prompts) if p]
    if not valid_prompts:
        return [""] * len(prompts), [0] * len(prompts), [0] * len(prompts)
    
    indices, valid_prompt_list = zip(*valid_prompts)
    
    # Tokenize all prompts at once with padding
    inputs = tokenizer(
        list(valid_prompt_list),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens
    )

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    
    # Track input lengths before moving to GPU
    input_lengths = [int((inputs['attention_mask'][i] == 1).sum()) for i in range(len(valid_prompt_list))]
    
    # Move to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with batching
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                do_sample=False,  # Greedy decoding
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )
    
    # Decode only the new tokens for each sequence
    responses = []
    output_lengths = []
    
    for i, output in enumerate(outputs):
        # Get only the generated part (after input)
        input_length = input_lengths[i]
        generated_tokens = output[input_length:]
        output_length = len(generated_tokens)
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
        output_lengths.append(output_length)
    
    # Reconstruct full response lists with zeros for skipped prompts
    full_responses = [""] * len(prompts)
    full_input_lengths = [0] * len(prompts)
    full_output_lengths = [0] * len(prompts)
    
    for idx, response, in_len, out_len in zip(indices, responses, input_lengths, output_lengths):
        full_responses[idx] = response
        full_input_lengths[idx] = in_len
        full_output_lengths[idx] = out_len
    
    return full_responses, full_input_lengths, full_output_lengths


def group_by_length_params(queries_data: List[dict], text_col: str, 
                           query_type_col: str, output_type_col: str,
                           max_in_short: int, max_in_long: int,
                           max_out_short: int, max_out_long: int) -> dict:
    """
    Group queries by their length parameters to enable efficient batching.
    Queries with same max_in/max_out can be processed together.
    """
    groups = {}
    
    for idx, row in enumerate(queries_data):
        query = (row.get(text_col) or "").strip()
        qtype = (row.get(query_type_col) or "").strip().lower()
        otype = (row.get(output_type_col) or "").strip().lower()
        
        if not query:
            continue
        
        try:
            max_in, max_out = pick_lengths(
                qtype, otype, max_in_short, max_in_long, max_out_short, max_out_long
            )
            
            key = (max_in, max_out)
            if key not in groups:
                groups[key] = []
            
            groups[key].append({
                'idx': idx,
                'query': query,
                'qtype': qtype,
                'otype': otype,
                'max_in': max_in,
                'max_out': max_out
            })
        except ValueError as e:
            print(f"Skipping row {idx}: {e}")
            continue
    
    return groups


def main():
    parser = argparse.ArgumentParser(description="Run Mistral models efficiently over a CSV of queries.")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id or local path")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file with queries")
    parser.add_argument("--text_col", type=str, default="query",
                        help="CSV column for the input text")
    parser.add_argument("--query_type_col", type=str, default="query_type",
                        help='CSV column for query type: "short" | "long"')
    parser.add_argument("--output_type_col", type=str, default="output_type",
                        help='CSV column for output type: "short" | "long"')
    parser.add_argument("--out_csv", type=str, default="mistral_outputs.csv",
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
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # Efficiency knobs
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for parallel generation (adjust based on GPU memory)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--skip_samples", type=int, default=0,
                        help="Number of samples to skip from the beginning")

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Set all seeds for reproducibility
    set_all_seeds(args.seed)

    # Load model
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model, tokenizer = load_model(args.model, args.dtype, token=token)

    # Read ALL data first
    all_queries_data = []
    with open(args.csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for col in [args.text_col, args.query_type_col, args.output_type_col]:
            if col not in reader.fieldnames:
                print(f"Missing column in CSV: {col}", file=sys.stderr)
                sys.exit(1)
        
        for row in reader:
            all_queries_data.append(row)
    
    print(f"\nTotal samples in CSV: {len(all_queries_data)}")
    
    # Skip samples if requested
    if args.skip_samples > 0:
        all_queries_data = all_queries_data[args.skip_samples:]
        print(f"Skipped first {args.skip_samples} samples")
    
    # Don't use random sampling - take first N queries like API does
    if args.max_samples is not None:
        queries_data = all_queries_data[:args.max_samples]
        print(f"✓ Processing first {len(queries_data)} queries (deterministic)")
    else:
        queries_data = all_queries_data
        print(f"✓ Processing all {len(queries_data)} queries")

    # Seed still used for generation, not query selection
    set_all_seeds(args.seed)

    print(f"\nProcessing with batch size {args.batch_size}")
    print(f"Optimizations: TF32, Mixed Precision, KV Cache, Parallel Batching")

    # Group queries by length parameters for efficient batching
    print("\nGrouping queries by length parameters for optimal batching...")
    query_groups = group_by_length_params(
        queries_data, args.text_col, args.query_type_col, args.output_type_col,
        args.max_input_tokens_short, args.max_input_tokens_long,
        args.max_new_tokens_short, args.max_new_tokens_long
    )
    
    print(f"Created {len(query_groups)} groups:")
    for (max_in, max_out), items in query_groups.items():
        print(f"  Group ({max_in} in, {max_out} out): {len(items)} queries")

    # Initialize carbon tracker
    if CARBONTRACKER_AVAILABLE:
        tracker = CarbonTracker(epochs=1, components="gpu")
        tracker.epoch_start()
    else:
        tracker = None

    # Initialize results array and token statistics
    results = [None] * len(queries_data)
    total_input_tokens = 0
    total_output_tokens = 0
    query_type_counts = {"short": 0, "long": 0}
    output_type_counts = {"short": 0, "long": 0}
    
    # Process each group with optimal batching
    total_processed = 0
    start_time = time.time()
    
    for group_idx, ((max_in, max_out), group_items) in enumerate(query_groups.items()):
        print(f"\n{'='*70}")
        print(f"Group {group_idx+1}/{len(query_groups)}: {max_in} in → {max_out} out ({len(group_items)} queries)")
        print(f"{'='*70}")
        
        group_start = time.time()
        
        # Process this group in batches
        for batch_start in range(0, len(group_items), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(group_items))
            batch_items = group_items[batch_start:batch_end]
            
            # Extract queries for this batch
            batch_queries = [item['query'] for item in batch_items]
            
            # Generate in parallel and get token counts
            batch_start_time = time.time()
            batch_responses, batch_in_tokens, batch_out_tokens = generate_batch_parallel(
                model, tokenizer, batch_queries, max_in, max_out,
                args.temperature, args.top_p, args.top_k, args.repetition_penalty
            )
            batch_time = time.time() - batch_start_time
            
            # Store results in correct positions and accumulate statistics
            for item, response, in_tokens, out_tokens in zip(batch_items, batch_responses, batch_in_tokens, batch_out_tokens):
                results[item['idx']] = {
                    'query': item['query'],
                    'query_type': item['qtype'],
                    'output_type': item['otype'],
                    'response': response,
                    'input_tokens': in_tokens,
                    'output_tokens': out_tokens
                }
                
                # Accumulate statistics
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                if item['qtype'] in query_type_counts:
                    query_type_counts[item['qtype']] += 1
                if item['otype'] in output_type_counts:
                    output_type_counts[item['otype']] += 1
            
            total_processed += len(batch_items)
            queries_per_sec = len(batch_items) / batch_time if batch_time > 0 else 0
            progress = (total_processed / len(queries_data)) * 100
            
            # Calculate tokens for this batch
            batch_total_tokens = sum(batch_in_tokens) + sum(batch_out_tokens)
            
            print(f"Batch {batch_start//args.batch_size + 1}: "
                  f"{len(batch_items)} queries in {batch_time:.2f}s "
                  f"({queries_per_sec:.2f} q/s, {batch_total_tokens} tokens) "
                  f"[{total_processed}/{len(queries_data)} = {progress:.1f}%]")
        
        group_time = time.time() - group_start
        print(f"Group completed in {group_time:.2f}s ({len(group_items)/group_time:.2f} q/s avg)")

    total_time = time.time() - start_time
    
    # Write results
    print(f"\nWriting results to {args.out_csv}...")
    out_fields = ["query", "query_type", "output_type", "response", "input_tokens", "output_tokens"]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        for result in results:
            if result is not None:
                writer.writerow(result)

    # Stop carbon tracker
    if tracker:
        tracker.epoch_end()
        tracker.stop()

    # Print overall statistics
    total_tokens = total_input_tokens + total_output_tokens
    print(f"\n{'='*70}")
    print(f"OVERALL RUN STATISTICS")
    print(f"{'='*70}")
    print(f"Total Queries Processed: {total_processed}")
    print(f"Total Input Tokens: {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Avg Input Tokens/Query: {total_input_tokens/total_processed:.1f}")
    print(f"Avg Output Tokens/Query: {total_output_tokens/total_processed:.1f}")
    print(f"Avg Total Tokens/Query: {total_tokens/total_processed:.1f}")
    print(f"\nQuery Type Distribution:")
    print(f"  Short: {query_type_counts['short']} ({100*query_type_counts['short']/total_processed:.1f}%)")
    print(f"  Long: {query_type_counts['long']} ({100*query_type_counts['long']/total_processed:.1f}%)")
    print(f"\nOutput Type Distribution:")
    print(f"  Short: {output_type_counts['short']} ({100*output_type_counts['short']/total_processed:.1f}%)")
    print(f"  Long: {output_type_counts['long']} ({100*output_type_counts['long']/total_processed:.1f}%)")
    print(f"{'='*70}")

    # Final benchmark statistics
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average throughput: {total_processed/total_time:.2f} queries/sec")
    print(f"Average per query: {total_time/total_processed:.2f}s")
    print(f"Token throughput: {total_tokens/total_time:.1f} tokens/sec")
    print(f"Output file: {args.out_csv}")
    
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    print(f"{'='*70}")
    
    # Save summary to file
    summary_file = args.out_csv.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OVERALL RUN STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Total Queries Processed: {total_processed}\n")
        f.write(f"Total Input Tokens: {total_input_tokens:,}\n")
        f.write(f"Total Output Tokens: {total_output_tokens:,}\n")
        f.write(f"Total Tokens: {total_tokens:,}\n")
        f.write(f"Avg Input Tokens/Query: {total_input_tokens/total_processed:.1f}\n")
        f.write(f"Avg Output Tokens/Query: {total_output_tokens/total_processed:.1f}\n")
        f.write(f"Avg Total Tokens/Query: {total_tokens/total_processed:.1f}\n")
        f.write(f"\nQuery Type Distribution:\n")
        f.write(f"  Short: {query_type_counts['short']} ({100*query_type_counts['short']/total_processed:.1f}%)\n")
        f.write(f"  Long: {query_type_counts['long']} ({100*query_type_counts['long']/total_processed:.1f}%)\n")
        f.write(f"\nOutput Type Distribution:\n")
        f.write(f"  Short: {output_type_counts['short']} ({100*output_type_counts['short']/total_processed:.1f}%)\n")
        f.write(f"  Long: {output_type_counts['long']} ({100*output_type_counts['long']/total_processed:.1f}%)\n")
        f.write(f"\nBenchmark:\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Throughput: {total_processed/total_time:.2f} queries/sec\n")
        f.write(f"Token throughput: {total_tokens/total_time:.1f} tokens/sec\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()