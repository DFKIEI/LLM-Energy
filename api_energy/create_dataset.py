#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
import random
from typing import List, Dict
from groq import Groq

# Prompt templates for different categories
PROMPT_CATEGORIES = {
    "technical": [
        "Write a comprehensive technical analysis with code examples for",
        "Develop a detailed implementation guide with performance benchmarks for",
        "Create an extensive architectural overview with trade-off analysis for",
        "Provide a multi-stage tutorial with debugging tips and optimization strategies for",
        "Design a complete system architecture with component interactions and scaling considerations for",
    ],
    "creative": [
        "Write a detailed story about",
        "Create an elaborate world-building description for",
        "Compose a comprehensive character analysis of",
        "Develop a complete plot outline for",
        "Write an in-depth review and analysis of",
    ],
    "educational": [
        "Provide a thorough explanation of",
        "Create a complete lesson plan for teaching",
        "Write a comprehensive study guide covering",
        "Develop a detailed curriculum outline for",
        "Explain with examples and case studies",
    ],
    "business": [
        "Write a detailed business plan for",
        "Analyze the market opportunity and strategy for",
        "Create a comprehensive marketing strategy for",
        "Develop a complete operational framework for",
        "Provide an in-depth competitive analysis of",
    ]
}

TOPICS = {
    "technical": [
        "machine learning algorithms", "distributed systems", "blockchain technology",
        "quantum computing", "cybersecurity frameworks", "cloud architecture",
        "microservices design", "database optimization", "API development",
        "containerization strategies", "neural networks", "data pipelines"
    ],
    "creative": [
        "space exploration", "medieval fantasy worlds", "dystopian futures",
        "underwater civilizations", "time travel paradoxes", "artificial consciousness",
        "parallel dimensions", "magical realism", "post-apocalyptic societies",
        "alien first contact", "virtual reality worlds", "genetic engineering"
    ],
    "educational": [
        "advanced mathematics", "historical events", "scientific principles",
        "philosophical concepts", "economic theories", "psychological phenomena",
        "environmental science", "linguistics", "anthropology", "astronomy",
        "biochemistry", "political science"
    ],
    "business": [
        "sustainable energy startups", "fintech innovation", "e-commerce platforms",
        "healthcare technology", "educational technology", "food delivery services",
        "remote work solutions", "AI consulting services", "green technology",
        "social media platforms", "subscription services", "logistics optimization"
    ]
}

def create_prompt(category: str, target_words: int) -> str:
    """Create a synthetic prompt of target length."""
    template = random.choice(PROMPT_CATEGORIES[category])
    topic = random.choice(TOPICS[category])
    
    # Base prompt
    prompt = f"{template} {topic}."
    
    # Add complexity based on target length
    if target_words >= 8000:  # Long prompt - increased from 5000
        extensions = [
            f" Include detailed examples, case studies, and real-world applications across at least 5 different scenarios.",
            f" Provide step-by-step explanations with technical specifications and implementation details for each component.",
            f" Compare and contrast at least 7 different approaches and methodologies, with pros and cons of each.",
            f" Discuss historical context, current trends, and make detailed predictions about future implications.",
            f" Analyze at least 8 potential challenges, their limitations, and propose multiple solutions for each.",
            f" Provide a comprehensive cost-benefit analysis including economic, social, environmental, and technical impacts.",
            f" Include references from at least 10 different sources and cite relevant academic research.",
            f" Create a multi-phase implementation plan with timelines, resource requirements, and risk assessments.",
            f" Discuss regulatory frameworks across different jurisdictions and compliance considerations.",
            f" Provide examples of code or pseudocode implementations where relevant, with detailed comments.",
            f" Include performance benchmarks and comparative analysis against alternative solutions.",
            f" Address common misconceptions and provide detailed clarifications with evidence.",
        ]
        
        # Add more extensions for extra complexity
        selected_extensions = random.sample(extensions, min(len(extensions), random.randint(8, 10)))
        prompt += "".join(selected_extensions)
        
        # Add specific requirements to reach target length
        prompt += f" Structure your response with at least 12 main sections, each with 3-5 subsections. Include a table of contents, executive summary, detailed analysis, and conclusion. Ensure your response is comprehensive and detailed, covering all aspects thoroughly with examples and explanations. The response should be approximately {target_words} words and formatted with proper headings, bullet points, tables, and diagrams where appropriate."
            
    else:  # Short prompt (now 2000 words instead of 500)
        extensions = [
            f" Provide at least 5 clear examples and practical applications.",
            f" Compare at least 3 different approaches with their benefits and drawbacks.",
            f" Explain the core concepts, principles, and technical requirements.",
            f" Include specific implementation details with references to best practices.",
            f" Provide a structured analysis with supporting evidence and case studies.",
        ]
        selected_extensions = random.sample(extensions, min(len(extensions), random.randint(3, 4)))
        prompt += "".join(selected_extensions)
        prompt += f" Structure your response with an introduction, 4-6 main sections, and a conclusion. Your response should be detailed yet focused, approximately {target_words} words."
    
    return prompt

def generate_synthetic_dataset(client: Groq, model: str, num_samples: int, output_csv: str):
    """Generate synthetic prompts using Groq API."""
    
    out_fields = ["query", "query_type", "output_type", "estimated_words", "category", "topic"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        
        for i in range(num_samples):
            # Randomly choose query and output types
            query_type = random.choice(["short", "long"])
            output_type = random.choice(["short", "long"])
            category = random.choice(list(PROMPT_CATEGORIES.keys()))
            
            # Set target words based on types - INCREASED
            if query_type == "short":
                target_words = 2000  # Increased from 500
            else:
                target_words = 8000  # Increased from 5000
                
            # Create the synthetic prompt
            prompt = create_prompt(category, target_words)
            
            # Estimate processing time (30 seconds - 1 minute)
            estimated_processing_time = random.uniform(30.0, 60.0)
            
            print(f"Generated sample {i+1}/{num_samples}: {query_type} query, {output_type} output, {category} category")
            
            writer.writerow({
                "query": prompt,
                "query_type": query_type,
                "output_type": output_type,
                "estimated_words": target_words,
                "category": category,
                "topic": random.choice(TOPICS[category])
            })
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
    
    print(f"Generated {num_samples} synthetic prompts in {output_csv}")

def validate_prompts_with_groq(client: Groq, model: str, csv_file: str, sample_size: int = 5):
    """Validate a few prompts with Groq to estimate actual processing time."""
    
    print(f"\nValidating {sample_size} random prompts with {model}...")
    
    with open(csv_file, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        
        # Sample random prompts
        sample_rows = random.sample(rows, min(sample_size, len(rows)))
        
        for i, row in enumerate(sample_rows):
            prompt = row["query"]
            query_type = row["query_type"]
            output_type = row["output_type"]
            
            print(f"\nTesting prompt {i+1}: {query_type} -> {output_type}")
            
            start_time = time.time()
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.7,
                    max_tokens=4000 if output_type == "short" else 16000,
                )
                end_time = time.time()
                
                processing_time = end_time - start_time
                response_text = response.choices[0].message.content
                word_count = len(response_text.split())
                
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Response word count: {word_count}")
                print(f"Prompt length: {len(prompt.split())} words")
                
            except Exception as e:
                print(f"Error testing prompt: {e}")
                
            # Rate limiting
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic prompts for LLM benchmarking")
    parser.add_argument("--groq_api_key", type=str, default="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",
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--model", type=str, default="llama3-70b-8192",
                        help="Groq model to use (default: llama3-70b-8192)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of synthetic prompts to generate")
    parser.add_argument("--output_csv", type=str, default="synthetic_prompts.csv",
                        help="Output CSV file for synthetic prompts")
    parser.add_argument("--validate", action="store_true",
                        help="Validate generated prompts with Groq API")
    parser.add_argument("--validation_samples", type=int, default=5,
                        help="Number of prompts to validate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not args.groq_api_key:
        print("Missing Groq API key. Set --groq_api_key or GROQ_API_KEY env var.", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize Groq client
    client = Groq(api_key=args.groq_api_key)
    
    # Generate synthetic dataset
    print(f"Generating {args.num_samples} synthetic prompts...")
    generate_synthetic_dataset(client, args.model, args.num_samples, args.output_csv)
    
    # Validate if requested
    if args.validate:
        validate_prompts_with_groq(client, args.model, args.output_csv, args.validation_samples)
    
    print(f"\nDataset generation complete!")
    print(f"Use the generated {args.output_csv} with your mistral_models.py or mistral_models_api.py scripts.")

if __name__ == "__main__":
    main()