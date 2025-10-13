#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


#################################### Import carbontracker and apply the fix ---- ONLY NEEDED IF RUNNING ON SLURM CLUSTER
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
    # Save original method
    original_log_components_info = CarbonTrackerThread._log_components_info
    
    # Create fixed method
    def fixed_log_components_info(self):
        log = ["The following components were found:"]
        for comp in self.components:
            name = comp.name.upper()
            # Fix here: decode byte strings in device names
            devices = [d.decode('utf-8') if isinstance(d, bytes) else d for d in comp.devices()]
            devices = ", ".join(devices)
            log.append(f"{name} with device(s) {devices}.")
        log_str = " ".join(log)
        print(log_str)
    
    # Apply the patch
    CarbonTrackerThread._log_components_info = fixed_log_components_info
    print("Successfully patched carbontracker device handling")
    
except (ImportError, AttributeError) as e:
    print(f"Failed to set up carbontracker: {e}")
##########################################################################################



def load_model(model_name: str, dtype: str = "fp16", token: str | None = None):
    #torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    #print(f"Loading model: {model_name} [{dtype}]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)  # pass token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,  # pass token
    )
    model.eval()
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


def generate(model, tokenizer, prompt: str, max_input_tokens: int, max_new_tokens: int,
             temperature: float, top_p: float, top_k: int, repetition_penalty: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Return only the generated continuation
    gen_ids = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Run Mistral models over a CSV of queries.")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id or local path (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
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
                        help="HF access token for gated models (or set HF_TOKEN env var)")

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

    # Misc
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to process (default: process all)")
    
    args = parser.parse_args()

    

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)

    # Use --hf_token arg (falls back to env if empty)
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model, tokenizer = load_model(args.model, args.dtype, token=token)

    # Read data with sample limit
    queries_data = []
    sample_count = 0
    with open(args.csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for col in [args.text_col, args.query_type_col, args.output_type_col]:
            if col not in reader.fieldnames:
                print(f"Missing column in CSV: {col}", file=sys.stderr)
                sys.exit(1)
        
        for row in reader:
            queries_data.append(row)
            sample_count += 1
            
            # Stop if we've reached the sample limit
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
    
    print(f"Processing {len(queries_data)} samples (from {sample_count} total)")
    

    # Initialize the carbon tracker
    tracker = CarbonTracker(epochs=1)
    tracker.epoch_start()

    # Process in batches for better GPU utilization
    batch_size = 32  # Adjust based on GPU memory
    out_fields = ["query", "query_type", "output_type", "response"]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        # Process in batches
        for i in range(0, len(queries_data), batch_size):
            batch = queries_data[i:i + batch_size]
            batch_queries = []
            batch_params = []
            
            for row in batch:
                query = (row.get(args.text_col) or "").strip()
                qtype = (row.get(args.query_type_col) or "").strip().lower()
                otype = (row.get(args.output_type_col) or "").strip().lower()
                
                if query:
                    max_in, max_out = pick_lengths(
                        qtype, otype,
                        args.max_input_tokens_short, args.max_input_tokens_long,
                        args.max_new_tokens_short, args.max_new_tokens_long
                    )
                    batch_queries.append(query)
                    batch_params.append((max_in, max_out, qtype, otype))
                else:
                    batch_queries.append("")
                    batch_params.append((0, 0, qtype, otype))
            
            # Generate batch responses
            batch_responses = generate_batch(
                model, tokenizer, batch_queries, batch_params,
                args.temperature, args.top_p, args.top_k, args.repetition_penalty
            )
            
            # Write batch results
            for j, (query, (_, _, qtype, otype), response) in enumerate(zip(batch_queries, batch_params, batch_responses)):
                writer.writerow({
                    "query": query,
                    "query_type": qtype,
                    "output_type": otype,
                    "response": response
                })
            
            print(f"Processed batch {i//batch_size + 1}/{(len(queries_data) + batch_size - 1)//batch_size}")

    tracker.epoch_end()
    tracker.stop()
    print(f"Done. Wrote: {args.out_csv}")

def generate_batch(model, tokenizer, prompts, params, temperature, top_p, top_k, repetition_penalty):
    """Generate responses for a batch of prompts."""
    responses = []
    
    # For now, process sequentially (can be optimized further with true batching)
    for prompt, (max_in, max_out, _, _) in zip(prompts, params):
        if not prompt:
            responses.append("")
            continue
            
        response = generate(
            model, tokenizer, prompt, max_in, max_out,
            temperature, top_p, top_k, repetition_penalty
        )
        responses.append(response)
    
    return responses
    


if __name__ == "__main__":
    main()