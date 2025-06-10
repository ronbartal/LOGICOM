# Debate Experiment System - Research Paper Replication

## Overview
This system has been enhanced to properly replicate research paper experiments with multiple helper types, multiple runs per claim, comprehensive statistics tracking, and proper filtering of invalid runs.

## Key Improvements Made

### 🐛 **Bugs Fixed**
1. **Excel Export Bug**: Fixed column mismatch when adding new topics to existing Excel files
2. **Iteration Logic Bug**: Fixed off-by-one error (was running 501 claims instead of 500)
3. **Multiple Runs Support**: Added support for running each claim multiple times per helper type
4. **Round Limit**: Reduced from 12 to 10 rounds maximum per debate
5. **API Key Automation**: Automatically reads API keys from API_keys file (no command line required)

### ✨ **New Features**
1. **Batch Processing**: Automatically run all three helper types in sequence
2. **Multiple Runs**: Run each claim 3 times per helper type (configurable)
3. **Moderator Tracking**: Track PALM vs GPT-4 usage and PALM failure rates
4. **Statistics Generation**: Comprehensive Excel reports with success rates and analysis
5. **Forced GPT-4 Mode**: Option to start with GPT-4 moderator instead of PALM
6. **Progress Tracking**: Real-time progress updates during long experiments
7. **Bad Run Filtering**: Automatic detection and filtering of invalid runs (off-topic, greeting loops)
8. **Dual Statistics**: Show both filtered (valid runs only) and unfiltered success rates

### 📊 **Advanced Statistics & Reporting**
- **Individual debate results** with moderator used, finish reason, validity status
- **Success rates by helper type** - both including and excluding bad runs
- **PALM failure rates** and automatic GPT-4 fallback tracking
- **Run validity analysis** with common invalid run reasons
- **Average rounds per debate** for valid vs all runs
- **Comprehensive Excel reports** with multiple analysis sheets
- **Bad run identification**: Off-topic conversations, greeting loops, very short runs

### 🔍 **Run Validity Classification**

**Valid Runs** (included in filtered statistics):
- Normal debate completion (persuasion success/failure)
- Safety stops at 10 rounds (proper debate that reached time limit)
- Runs with ≥3 rounds of substantive conversation

**Invalid Runs** (excluded from filtered statistics):
- Off-topic conversations (`<OFF-TOPIC>` signal)
- Greeting loops (agents just greeting without engaging topic)
- Very short runs (<3 rounds) indicating failure to engage
- System errors or API failures

## Helper Types
1. **No_Helper**: No assistance provided
2. **Vanilla_Helper**: Legitimate argumentative improvements
3. **Fallacy_Helper**: Introduces logical fallacies to help "win" arguments

## How to Run Experiments

### 1. Prerequisites
Create an `API_keys` file in the project root:
```
OpenAI_API_key: your_openai_key_here
Google_API_key: your_google_key_here
```
**Note**: API keys are now read automatically - no need to specify them in command line!

### 2. Test Run (Recommended First)
Run a small test with 2 claims to verify everything works:
```bash
./run_test_experiment.sh
```
This runs 18 debates total (2 claims × 3 helper types × 3 runs each)

### 3. Full PALM Experiments
Run 500 claims with PALM moderator (with GPT-4 fallback):
```bash
./run_palm_experiments.sh
```
This runs 4,500 debates total (500 claims × 3 helper types × 3 runs each)

### 4. Full GPT-4 Experiments  
Run 500 claims with forced GPT-4 moderator:
```bash
./run_gpt4_experiments.sh
```
This runs 4,500 debates total (500 claims × 3 helper types × 3 runs each)

### 5. Custom Experiments
For custom configurations, run manually:
```bash
python main.py \
    --claim_number 499 \
    --num_runs 3 \
    --run_all_helpers \
    --force_gpt4_moderator \
    --data_path "./claims/all-claim-not-claim.csv" \
    --log_html_path "./custom_experiment/"
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--claim_number` | Last claim index (0-based) | 499 (= 500 claims) |
| `--num_runs` | Runs per helper type per claim | 3 |
| `--run_all_helpers` | Run all three helper types | False |
| `--force_gpt4_moderator` | Start with GPT-4 instead of PALM | False |
| `--helper_prompt_instruction` | Single helper type if not running all | No_Helper |
| `--data_path` | Path to claims CSV file | ./claims/all-claim-not-claim.csv |
| `--log_html_path` | Output directory | ./debates/ |
| `--api_key_openai` | OpenAI API key (optional) | Auto-read from API_keys file |
| `--api_key_palm` | Google API key (optional) | Auto-read from API_keys file |

## Output Files

### Individual Debate Files
- **HTML**: `{output_dir}/{topic_id}/{helper_type}/{chat_id}.html`
- **JSON**: `{output_dir}/{topic_id}/{helper_type}/{chat_id}.json`
- **TXT**: `{output_dir}/{topic_id}/{helper_type}/{chat_id}.txt`

### Summary Files
- **All Results**: `{output_dir}/all.xlsx` - Raw data for all debates
- **Statistics**: `{output_dir}/comprehensive_statistics.xlsx` - Analysis with multiple sheets:
  - **Overall_Summary**: Experiment totals with filtered/unfiltered rates
  - **Helper_Statistics**: Success rates by helper type (both filtered and unfiltered)
  - **Moderator_Statistics**: PALM vs GPT-4 usage and failure rates
  - **Run_Validity_Analysis**: Invalid run analysis and common reasons
  - **All_Debate_Results**: Complete raw data with validity flags
  - **Valid_Runs_Only**: Filtered dataset excluding bad runs
  - **Invalid_Runs_Analysis**: Analysis of problematic runs
  - **Success_by_Claim_All**: Pivot table for all runs
  - **Success_by_Claim_Valid**: Pivot table for valid runs only

## PALM Failure Handling

When PALM moderator returns `None` (API failure), the system:
1. Automatically switches to GPT-4 moderator
2. Logs this as "GPT-4 (PALM_Failed)" in moderator tracking
3. Continues the debate seamlessly
4. Tracks PALM failure rates in statistics

## Expected Results Structure

For replicating research paper conditions:
- **500 claims** (indices 0-499)
- **3 helper types** (No_Helper, Vanilla_Helper, Fallacy_Helper)  
- **3 runs each** = 4,500 total debates per moderator type
- **10 rounds maximum** per debate
- **Comprehensive tracking** of success rates and moderator usage
- **Bad run filtering** to match research paper methodology

## Example Statistics Output

```
OVERALL EXPERIMENT SUMMARY
============================================================
Total debates run: 4500
Valid debates: 3847
Invalid debates: 653 (14.5%)
Success rate (ALL runs): 24.67%
Success rate (VALID runs only): 28.85%
Unique claims processed: 500
Average rounds per debate (all): 6.4
Average rounds per debate (valid): 7.2

SUCCESS RATES BY HELPER TYPE
============================================================
Helper Type     All Runs        Valid Only      Invalid %
------------------------------------------------------------
No_Helper       15.23% (228/1500) 17.89% (198/1107)  26.2%
Vanilla_Helper  28.67% (430/1500) 32.14% (412/1282)  14.5%
Fallacy_Helper  38.40% (576/1500) 41.25% (578/1401)   6.6%

MODERATOR USAGE STATISTICS
============================================================
PALM successful usage: 3456 debates
GPT-4 direct usage: 0 debates  
PALM failures (switched to GPT-4): 1044 debates
Overall PALM failure rate: 23.20%

RUN VALIDITY ANALYSIS
============================================================
Total runs: 4500
Valid runs: 3847 (85.5%)
Invalid runs: 653 (14.5%)

Most common invalid run reasons:
  <TERMINATE>: 387 runs (greeting loops)
  safety_stop_max_rounds: 184 runs (short conversations)
  <OFF-TOPIC>: 82 runs (off-topic discussions)
```

## Research Paper Replication

This system now properly supports the experimental setup described in research papers:
- ✅ Multiple runs per condition (3x each)
- ✅ All three helper types tested
- ✅ Comprehensive statistics tracking
- ✅ PALM vs GPT-4 moderator comparison
- ✅ Success rate analysis by helper type
- ✅ Controlled debate length (10 rounds max)
- ✅ Proper error handling and fallback mechanisms
- ✅ **Bad run filtering** (like research papers exclude problematic runs)
- ✅ **Dual statistics** (filtered and unfiltered success rates)
- ✅ **Automated API key management**

## Key Research Insights

The dual statistics approach allows you to:
1. **Compare with research papers** that filtered out bad runs
2. **Understand real-world performance** including all attempted runs  
3. **Analyze system robustness** by examining invalid run patterns
4. **Track moderator reliability** through PALM failure rates

This gives you both the clean academic metrics and the practical deployment metrics needed for comprehensive research analysis.

## Troubleshooting

1. **API Key Issues**: Ensure `API_keys` file exists with correct format in project root
2. **Memory Issues**: Large experiments may need adequate RAM for progress tracking
3. **PALM Failures**: High failure rates are normal; system automatically handles with GPT-4 fallback
4. **Invalid Runs**: High invalid percentages may indicate prompt tuning needed
5. **Incomplete Experiments**: Check logs for specific error messages; experiments can be resumed by running again 