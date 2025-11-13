#!/bin/bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/compile_token_summary.sh

echo "=========================================="
echo "TOKEN SUMMARY COMPILATION"
echo "Extracting from .out files only"
echo "=========================================="
echo ""

# Check if pandas is installed
python3 -c "import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pandas..."
    pip install pandas
fi

# Run the compilation script
python3 compile_tokens.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    ls -lh token_summary*.csv token_summary.json 2>/dev/null
    echo ""
    
    # Show preview of aggregated summary
    if [ -f "token_summary_aggregated.csv" ]; then
        echo "=========================================="
        echo "PREVIEW: Aggregated Summary (Top 10)"
        echo "=========================================="
        head -11 token_summary_aggregated.csv | column -t -s,
        echo ""
    fi
    
    # Show preview of comparison
    if [ -f "token_summary_comparison.csv" ]; then
        echo "=========================================="
        echo "PREVIEW: API vs Local Comparison"
        echo "=========================================="
        head -11 token_summary_comparison.csv | column -t -s,
        echo ""
    fi
    
    # Show JSON summary
    if [ -f "token_summary.json" ]; then
        echo "=========================================="
        echo "QUICK STATS FROM JSON"
        echo "=========================================="
        python3 -c "
import json
with open('token_summary.json', 'r') as f:
    data = json.load(f)
    print(f\"Total Files: {data['total_files']:,}\")
    print(f\"Total Queries: {data['total_queries']:,}\")
    print(f\"Total Input Tokens: {data['total_input_tokens']:,}\")
    print(f\"Total Output Tokens: {data['total_output_tokens']:,}\")
    print(f\"Total Tokens: {data['total_tokens']:,}\")
    print(f\"\\nBy Type:\")
    for mtype, stats in data['by_type'].items():
        print(f\"  {mtype}: {stats['total_tokens']:,} tokens ({stats['files']} files, {stats['queries']:,} queries)\")
"
        echo ""
    fi
    
    echo "=========================================="
    echo "All summaries saved!"
    echo "=========================================="
    
else
    echo ""
    echo "❌ Compilation failed. Check error messages above."
    exit 1
fi