#!/bin/bash

# Run 500-claim experiments using PALM moderator with all three helper types
# This script will run 3 times per helper type per claim (4500 total debates)

echo "Starting PALM Moderator Experiments"
echo "==================================="
echo "Claims: 500 (0-499)"
echo "Helper types: No_Helper, Vanilla_Helper, Fallacy_Helper"
echo "Runs per helper type: 3"
echo "Total debates: 4500"
echo "Max rounds per debate: 10"
echo "Moderator: PALM (with GPT-4 fallback)"
echo "Bad runs: Filtered out in statistics (off-topic, greeting loops)"
echo ""

# Make sure you have your API keys set in API_keys file
if [ ! -f "API_keys" ]; then
    echo "ERROR: API_keys file not found!"
    echo "Please create API_keys file with:"
    echo "OpenAI_API_key: your_openai_key_here"
    echo "Google_API_key: your_google_key_here"
    exit 1
fi

echo "✅ Found API_keys file, starting experiment..."
echo ""

# Run the experiment (API keys will be read automatically from API_keys file)
python3 main.py \
    --claim_number 499 \
    --num_runs 3 \
    --run_all_helpers \
    --data_path "./claims/all-claim-not-claim.csv" \
    --log_html_path "./debates_palm_experiments/" \
    --persuader_instruction "persuader_claim_reason_instruction" \
    --debater_instruction "debater_claim_reason_instruction"

echo ""
echo "PALM Moderator Experiments Completed!"
echo "Results saved to: ./debates_palm_experiments/"
echo "Summary statistics: ./debates_palm_experiments/comprehensive_statistics.xlsx"
echo ""
echo "📊 Statistics include both:"
echo "   - All runs (including bad runs)"
echo "   - Valid runs only (excluding off-topic/greeting loops)" 