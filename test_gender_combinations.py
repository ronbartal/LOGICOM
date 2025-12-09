"""
Test script to run debates with all gender combinations.
Runs 4*n*k debates: k debates for each of the 5 gender combinations (M,M), (M,F), (F,M), (F,F), legacy - no gender awerness
for each of n claims (indices 0 to n-1).
"""
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# --- Direct Imports ---
from utils.log_main import logger as debate_logger
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH
from config.loader import load_app_config
from main import _run_single_debate, select_prompts_by_gender

# Use colorama for terminal colors
from colorama import init
init(autoreset=True)

logger = debate_logger

def _setup_api_keys():
    """Sets API keys from the API_keys file."""
    set_environment_variables_from_file(API_KEYS_PATH)

def main():
    parser = argparse.ArgumentParser(description="Run gender combination test debates")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of claims to test (indices 0 to n-1). If -1, runs all claims.")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of debates per gender combination per claim")
    parser.add_argument("--helper_type", default="Default_No_Helper", 
                        help="Name of the helper type configuration in settings.yaml to use.")
    parser.add_argument("--settings_path", default="./config/settings.yaml", 
                        help="Path to the main settings configuration file.")
    parser.add_argument("--models_path", default="./config/models.yaml", 
                        help="Path to the LLM models configuration file.")
    parser.add_argument("--max_rounds", type=int, default=None,
                        help="Override the maximum number of debate rounds (default is from settings.yaml)")
    parser.add_argument("--debates_dir", default="debates",
                        help="Directory where debate logs should be saved (default: debates)")
    
    args = parser.parse_args()
    
    print("Starting gender combination test...")

    _setup_api_keys()
    
    try:
        # Load configuration
        debate_settings, agent_config, prompt_templates = load_app_config(
            settings_path=args.settings_path,
            models_path=args.models_path,
            run_config_name=args.helper_type
        )
        print("Configuration loaded successfully.")
        
        # Load claims data
        claims_file_path = debate_settings['claims_file_path']
        print(f"Loading claim data from: {claims_file_path}")
        claims_df = pd.read_csv(claims_file_path)
        num_claims = len(claims_df)
        print(f"Loaded {num_claims} claims.")
        
        if args.n > num_claims:
            print(f"Warning: n={args.n} is greater than available claims ({num_claims}). Using {num_claims} claims.")
            args.n = num_claims
        
        if args.n == -1:
            print("Running all claims.")
            args.n = num_claims
        
        print(f"n={args.n} claims, k={args.k} debates per combination")
        print(f"Total debates: {5 * args.n * args.k}")  # 5 combinations: M_M, M_F, F_M, F_F, None_None
    
        
        # Override max_rounds if provided
        if args.max_rounds is not None:
            print(f"Overriding max_rounds from {debate_settings['max_rounds']} to {args.max_rounds}")
            debate_settings['max_rounds'] = args.max_rounds
        
        helper_type = agent_config['helper_type']
        
        # Gender combinations: (persuader_gender, debater_gender)
        gender_combinations = [
            ("M", "M", "M_M"),
            ("M", "F", "M_F"),
            ("F", "M", "F_M"),
            ("F", "F", "F_F"),
            (None, None, "No-gender")
        ]
        
        # Name mappings
        persuader_names = {"M": "Josh", "F": "Karen"}
        debater_names = {"M": "Mike", "F": "Laura"}
        
        # Total number of debates
        total_debates = 5 * args.n * args.k
        pbar = tqdm(total=total_debates, desc="Running Gender Tests")
        
        # Run debates for each claim index
        for claim_idx in range(args.n):
            claim_data = claims_df.iloc[claim_idx]
            
            # For each gender combination
            for persuader_gender, debater_gender, gender_case in gender_combinations:
                # Select prompts based on gender flags for this combination
                selected_prompts = select_prompts_by_gender(
                    prompt_templates,
                    persuader_gender=persuader_gender,
                    debater_gender=debater_gender
                )
                
                # Get names only if gender is specified (None for legacy mode)
                if persuader_gender is not None:
                    persuader_name = persuader_names[persuader_gender]
                else:
                    persuader_name = None
                
                if debater_gender is not None:
                    debater_name = debater_names[debater_gender]
                else:
                    debater_name = None
                
                # Run k debates for this combination
                for run_num in range(args.k):
                    try:
                        run_result = _run_single_debate(
                            index=claim_idx,
                            claim_data=claim_data,
                            debate_settings=debate_settings,
                            agent_config=agent_config,
                            prompt_templates=selected_prompts,
                            helper_type=helper_type,
                            debates_base_dir=args.debates_dir,
                            persuader_name_by_gender=persuader_name,
                            debater_name_by_gender=debater_name,
                            gender_case=gender_case
                        )
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error in debate {claim_idx}, {gender_case}, run {run_num}: {e}", 
                                     extra={"msg_type": "system"})
                        pbar.update(1)
                        continue
        
        pbar.close()
        print(f"\nCompleted {total_debates} debates across all gender combinations.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

