"""
Expert Persona Campaign Script - Simplified version leveraging existing infrastructure.
Runs debates between expert persuaders (Education/Diplomat/Health) and standard debaters.
Similar structure to test_gender_combinations.py but for expert domains.
"""
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# --- Direct Imports ---
from utils.log_main import logger as debate_logger
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH
from config.loader import load_app_config, _load_prompt
from main import _run_single_debate

# Use colorama for terminal colors
from colorama import init
init(autoreset=True)

logger = debate_logger


def _setup_api_keys():
    """Sets API keys from the API_keys file."""
    set_environment_variables_from_file(API_KEYS_PATH)


def select_prompts_for_expert(loaded_prompts: dict, expert_domain: str) -> dict:
    """
    Selects expert-specific prompts based on expert_domain.
    
    Args:
        loaded_prompts: Dictionary of all loaded prompts
        expert_domain: 'E' (Education), 'I' (International/Diplomat), or 'H' (Health)
    
    Returns:
        Dictionary with expert-specific prompts replacing standard persuader prompts
    """
    # Map domain codes to expert types
    expert_mapping = {
        'E': 'education',
        'I': 'diplomat', 
        'H': 'health'
    }
    
    if expert_domain not in expert_mapping:
        raise ValueError(f"Unknown expert_domain: {expert_domain}. Expected 'E', 'I', or 'H'.")
    
    expert_type = expert_mapping[expert_domain]
    
    # Load expert-specific prompts
    expert_persona_path = f"prompts/persuader/expert_{expert_type}_persona.txt"
    expert_initial_path = f"prompts/persuader/expert_{expert_type}_initial.txt"
    
    expert_persona = _load_prompt(expert_persona_path)
    expert_initial = _load_prompt(expert_initial_path)
    
    # Create modified prompts dictionary
    selected_prompts = loaded_prompts.copy()
    
    # Modify persuader wrapper to include expert persona
    base_wrapper = loaded_prompts.get('persuader_wrapper', '')
    selected_prompts['persuader_wrapper'] = f"{expert_persona}\n\n{base_wrapper}"
    
    # Use expert initial prompt
    selected_prompts['persuader_initial'] = expert_initial
    
    logger.debug(f"Selected {expert_type} expert prompts", extra={"msg_type": "system"})
    
    return selected_prompts


def main():
    """Main execution logic for expert campaign."""
    parser = argparse.ArgumentParser(description="Run Expert Persona Debate Campaign")
    parser.add_argument("--k", type=int, default=3,
                       help="Number of repetitions per claim (default=3)")
    parser.add_argument("--helper_type", default="Default_No_Helper",
                       help="Name of the helper type configuration in settings.yaml")
    parser.add_argument("--settings_path", default="./config/settings.yaml",
                       help="Path to the main settings configuration file")
    parser.add_argument("--models_path", default="./config/models.yaml",
                       help="Path to the LLM models configuration file")
    parser.add_argument("--max_rounds", type=int, default=None,
                       help="Override the maximum number of debate rounds")
    parser.add_argument("--debates_dir", default="expert_debates",
                       help="Directory where debate logs should be saved")
    
    args = parser.parse_args()
    
    print("Starting Expert Persona Campaign...")
    print(f"Repetitions per claim: k={args.k}")
    
    # Setup API keys
    _setup_api_keys()
    
    try:
        # Load configuration
        debate_settings, agent_config, prompt_templates = load_app_config(
            settings_path=args.settings_path,
            models_path=args.models_path,
            run_config_name=args.helper_type
        )
        print("Configuration loaded successfully.")
        
        # Load claims with experts
        claims_file = "claims/claims_with_experts.csv"
        print(f"Loading claims from: {claims_file}")
        claims_df = pd.read_csv(claims_file)
        
        # Filter for expert claims (E, I, H)
        expert_claims = claims_df[claims_df['expert_domain'].isin(['E', 'I', 'H'])].copy()
        num_expert_claims = len(expert_claims)
        print(f"Found {num_expert_claims} claims with expert domains (E/I/H)")
        
        if num_expert_claims == 0:
            print("No expert claims found. Exiting.")
            return
        
        # Override max_rounds if provided
        if args.max_rounds is not None:
            print(f"Overriding max_rounds from {debate_settings['max_rounds']} to {args.max_rounds}")
            debate_settings['max_rounds'] = args.max_rounds
        
        helper_type = agent_config['helper_type']
        
        # Expert type mapping for naming
        expert_type_names = {'E': 'Education', 'I': 'Diplomat', 'H': 'Health'}
        
        # Calculate total debates
        total_debates = num_expert_claims * args.k
        print(f"Total debates to run: {total_debates} ({num_expert_claims} claims × {args.k} repetitions)")
        
        # Run debates
        pbar = tqdm(total=total_debates, desc="Running Expert Debates")
        
        for idx, row in expert_claims.iterrows():
            expert_domain = row['expert_domain']
            expert_type_name = expert_type_names[expert_domain]
            expert_case = f"Expert-{expert_type_name}"
            
            # Select expert-specific prompts
            selected_prompts = select_prompts_for_expert(prompt_templates, expert_domain)
            
            # Run k repetitions for this claim
            for rep in range(args.k):
                try:
                    run_result = _run_single_debate(
                        index=idx,
                        claim_data=row,
                        debate_settings=debate_settings,
                        agent_config=agent_config,
                        prompt_templates=selected_prompts,
                        helper_type=helper_type,
                        debates_base_dir=args.debates_dir,
                        persuader_name_by_gender=None,  # No names for experts
                        debater_name_by_gender=None,
                        persuader_gender_label=None,  # No gender labels for experts
                        debater_gender_label=None,
                        gender_case=expert_case,  # Use expert case instead of gender case
                        excel_file="expert_debates_summary.xlsx"  # Save to separate file
                    )
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Failed to run debate for claim {idx}, rep {rep}: {e}",
                               extra={"msg_type": "system"})
                    print(f"Error in debate (claim {idx}, rep {rep}): {e}")
                    pbar.update(1)
                    continue
        
        pbar.close()
        print(f"\nCompleted expert debates campaign.")
        print(f"Debate logs saved to: {args.debates_dir}/")
        print("Results summary saved to: expert_debates_summary.xlsx")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logger.error(f"Main execution error: {e}", extra={"msg_type": "system"})
        sys.exit(1)


if __name__ == '__main__':
    main()
