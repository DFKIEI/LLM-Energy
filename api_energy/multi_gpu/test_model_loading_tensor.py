#!/usr/bin/env python3
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/test_model_vllm_consistent.py
"""
vLLM with consistent output settings - uses greedy decoding for determinism
or controlled sampling with proper seeds.
"""
import argparse
import csv
import os
import sys
import time
from typing import Tuple, List

from vllm import LLM, SamplingParams

#################################### Import carbontracker and apply the fix
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
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
except (ImportError, AttributeError):
    CARBONTRACKER_AVAILABLE = False
##########################################################################################


def pick_lengths(query_type: str, output_type: str,
                 max_out_short: int, max_out_long: int) -> int:
    qt = (query_type or "").strip().lower()
    ot = (output_type or "").strip().lower()
    
    if ot == "short":
        return max_out_short
    return max_out_long


def create_sampling_params(max_tokens: int, min_tokens: int,
                           temperature: float, top_p: float, top_k: int,
                           repetition_penalty: float, seed: int,
                           use_greedy: bool = False) -> SamplingParams:
    """Create sampling params with options for greedy or controlled sampling."""
    
    if use_greedy:
        # Greedy decoding - fully deterministic
        return SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=0.0,  # Greedy
            top_p=1.0,
            top_k=-1,
            repetition_penalty=repetition_penalty,
            seed=seed,
            skip_special_tokens=True,
        )
    else:
        # Controlled sampling with seed for reproducibility
        return SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,  # Fixed seed for reproducibility
            skip_special_tokens=True,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="query")
    parser.add_argument("--query_type_col", type=str, default="query_type")
    parser.add_argument("--output_type_col", type=str, default="output_type")
    parser.add_argument("--out_csv", type=str, default="vllm_consistent_outputs.csv")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))

    parser.add_argument("--max_new_tokens_short", type=int, default=2048)
    parser.add_argument("--max_new_tokens_long", type=int, default=8192)

    # Sampling - use greedy for consistency, or set temperature
    parser.add_argument("--greedy", action="store_true", 
                        help="Use greedy decoding (fully deterministic)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)

    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for vLLM (handles continuous batching)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="/netscratch/banwari/.cache/huggingface")

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    # Load model
    print(f"Loading vLLM model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Mode: {'Greedy (deterministic)' if args.greedy else f'Sampling (temp={args.temperature})'}")
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.cache_dir,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        enforce_eager=False,  # Use CUDA graphs for speed
    )
    print("✓ Model loaded")

    # Read queries
    queries_data = []
    with open(args.csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            if args.max_samples and i >= args.max_samples:
                break
            queries_data.append(row)
    
    print(f"\nProcessing {len(queries_data)} samples")

    # Group queries by output length for efficient batching
    short_queries = []
    long_queries = []
    
    for idx, row in enumerate(queries_data):
        query = (row.get(args.text_col) or "").strip()
        qtype = (row.get(args.query_type_col) or "").strip().lower()
        otype = (row.get(args.output_type_col) or "").strip().lower()
        
        if not query:
            continue
        
        max_out = pick_lengths(qtype, otype, args.max_new_tokens_short, args.max_new_tokens_long)
        
        item = {
            "query": query,
            "query_type": qtype,
            "output_type": otype,
            "max_tokens": max_out,
            "orig_idx": idx
        }
        
        if otype == "short":
            short_queries.append(item)
        else:
            long_queries.append(item)
    
    print(f"Short output queries: {len(short_queries)}")
    print(f"Long output queries: {len(long_queries)}")

    if CARBONTRACKER_AVAILABLE:
        tracker = CarbonTracker(epochs=1, components="gpu", log_dir="./carbontracker_vllm_logs")
        tracker.epoch_start()
        print("✓ CarbonTracker started\n")

    all_results = [None] * len(queries_data)
    start_time = time.time()
    total_tokens = 0
    
    # Process short queries
    if short_queries:
        print(f"\n=== Processing {len(short_queries)} SHORT output queries ===")
        
        # Create sampling params for short outputs
        short_params = create_sampling_params(
            max_tokens=args.max_new_tokens_short,
            min_tokens=max(1, args.max_new_tokens_short // 2),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            use_greedy=args.greedy
        )
        
        prompts = [q["query"] for q in short_queries]
        
        batch_start = time.time()
        outputs = llm.generate(prompts, short_params)
        batch_time = time.time() - batch_start
        
        for i, output in enumerate(outputs):
            item = short_queries[i]
            response_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            total_tokens += num_tokens
            
            all_results[item["orig_idx"]] = {
                "query": item["query"],
                "query_type": item["query_type"],
                "output_type": item["output_type"],
                "response": response_text,
                "output_tokens": num_tokens
            }
        
        print(f"  Completed {len(short_queries)} queries in {batch_time:.1f}s")
        print(f"  Avg tokens: {sum(len(o.outputs[0].token_ids) for o in outputs)/len(outputs):.0f}")
    
    # Process long queries
    if long_queries:
        print(f"\n=== Processing {len(long_queries)} LONG output queries ===")
        
        # Create sampling params for long outputs
        long_params = create_sampling_params(
            max_tokens=args.max_new_tokens_long,
            min_tokens=max(1, args.max_new_tokens_long // 2),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            use_greedy=args.greedy
        )
        
        prompts = [q["query"] for q in long_queries]
        
        batch_start = time.time()
        outputs = llm.generate(prompts, long_params)
        batch_time = time.time() - batch_start
        
        for i, output in enumerate(outputs):
            item = long_queries[i]
            response_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            total_tokens += num_tokens
            
            all_results[item["orig_idx"]] = {
                "query": item["query"],
                "query_type": item["query_type"],
                "output_type": item["output_type"],
                "response": response_text,
                "output_tokens": num_tokens
            }
        
        print(f"  Completed {len(long_queries)} queries in {batch_time:.1f}s")
        print(f"  Avg tokens: {sum(len(o.outputs[0].token_ids) for o in outputs)/len(outputs):.0f}")

    if CARBONTRACKER_AVAILABLE:
        tracker.epoch_end()
        tracker.stop()
        print("\n✓ CarbonTracker stopped")

    total_time = time.time() - start_time

    # Write results
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["query", "query_type", "output_type", "response", "output_tokens"])
        writer.writeheader()
        for result in all_results:
            if result:
                writer.writerow(result)

    print(f"\n{'='*70}")
    print(f"VLLM CONSISTENT BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Greedy' if args.greedy else 'Sampling'}")
    print(f"Tensor Parallel: {args.tensor_parallel_size} GPUs")
    print(f"Queries: {len([r for r in all_results if r])}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Throughput: {total_tokens/total_time:.1f} tokens/sec")
    print(f"Output: {args.out_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()