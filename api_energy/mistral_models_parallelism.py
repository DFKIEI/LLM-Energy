#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time
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
                            top_k: int, repetition_penalty: float) -> List[str]:
    """
    Generate responses for multiple prompts in parallel using true batching.
    This significantly improves GPU utilization.
    """
    if not prompts or all(not p for p in prompts):
        return [""] * len(prompts)
    
    # Filter out empty prompts but keep track of indices
    valid_prompts = [(i, p) for i, p in enumerate(prompts) if p]
    if not valid_prompts:
        return [""] * len(prompts)
    
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
    
    # Move to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with batching - this is where GPU parallelism happens!
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Mixed precision for efficiency
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for efficiency
                num_beams=1,  # No beam search for speed
            )
    
    # Decode only the new tokens for each sequence
    responses = []
    for i, output in enumerate(outputs):
        # Get only the generated part (after input)
        input_length = inputs["input_ids"][i].shape[0]
        generated_tokens = output[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    # Reconstruct full response list with empty strings for skipped prompts
    full_responses = [""] * len(prompts)
    for idx, response in zip(indices, responses):
        full_responses[idx] = response
    
    return full_responses


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

    # Efficiency knobs - THIS WAS MISSING!
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

    torch.manual_seed(args.seed)

    # Load model
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model, tokenizer = load_model(args.model, args.dtype, token=token)

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

    print(f"\nProcessing {len(queries_data)} samples with batch size {args.batch_size}")
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
        tracker = CarbonTracker(epochs=1)
        tracker.epoch_start()
    else:
        tracker = None

    # Initialize results array
    results = [None] * len(queries_data)
    
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
            
            # Generate in parallel
            batch_start_time = time.time()
            batch_responses = generate_batch_parallel(
                model, tokenizer, batch_queries, max_in, max_out,
                args.temperature, args.top_p, args.top_k, args.repetition_penalty
            )
            batch_time = time.time() - batch_start_time
            
            # Store results in correct positions
            for item, response in zip(batch_items, batch_responses):
                results[item['idx']] = {
                    'query': item['query'],
                    'query_type': item['qtype'],
                    'output_type': item['otype'],
                    'response': response
                }
            
            total_processed += len(batch_items)
            queries_per_sec = len(batch_items) / batch_time if batch_time > 0 else 0
            progress = (total_processed / len(queries_data)) * 100
            
            print(f"Batch {batch_start//args.batch_size + 1}: "
                  f"{len(batch_items)} queries in {batch_time:.2f}s "
                  f"({queries_per_sec:.2f} q/s) "
                  f"[{total_processed}/{len(queries_data)} = {progress:.1f}%]")
        
        group_time = time.time() - group_start
        print(f"Group completed in {group_time:.2f}s ({len(group_items)/group_time:.2f} q/s avg)")

    total_time = time.time() - start_time
    
    # Write results
    print(f"\nWriting results to {args.out_csv}...")
    out_fields = ["query", "query_type", "output_type", "response"]
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

    # Final statistics
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {total_processed}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average throughput: {total_processed/total_time:.2f} queries/sec")
    print(f"Average per query: {total_time/total_processed:.2f}s")
    print(f"Output file: {args.out_csv}")
    
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()