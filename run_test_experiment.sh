#!/bin/bash

# Test run with 1 claim to verify the system works correctly
# This script will run 3 times per helper type per claim (9 total debates)

echo "Starting Test Experiment"
echo "======================="
echo "Claims: 1 (0)"
echo "Helper types: No_Helper, Vanilla_Helper, Fallacy_Helper"
echo "Runs per helper type: 3"
echo "Total debates: 9"
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

echo "✅ Found API_keys file, starting test experiment..."
echo ""

# Run the test experiment (API keys read automatically from API_keys file)
python3 main.py \
    --claim_number 1 \
    --num_runs 3 \
    --run_all_helpers \
    --data_path "./claims/all-claim-not-claim.csv" \
    --log_html_path "./debates_test/" \
    --persuader_instruction "persuader_claim_reason_instruction" \
    --debater_instruction "debater_claim_reason_instruction"

echo ""
echo "Test Experiment Completed!"
echo "Results saved to: ./debates_test/"
echo "Summary statistics: ./debates_test/comprehensive_statistics.xlsx"
echo ""
echo "📊 Statistics include both:"
echo "   - All runs (including bad runs)"
echo "   - Valid runs only (excluding off-topic/greeting loops)"
echo ""
echo "If this test works correctly, you can run the full experiments with:"
echo "./run_palm_experiments.sh"
echo "./run_gpt4_experiments.sh" 