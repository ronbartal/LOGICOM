"""
Classification script to analyze claims and extract domain and masculinity scores.
Processes even-indexed claims only (odd indices are negations).
"""
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
import os

# --- Direct Imports ---
from utils.log_main import logger
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH
from config.loader import load_yaml_config, _load_prompt
from llm.llm_factory import LLMFactory
from agents.domain_classifier_agent import DomainClassifierAgent
from agents.masculinity_classifier_agent import MasculinityClassifierAgent

# Use colorama for terminal colors
from colorama import init
init(autoreset=True)


def _setup_api_keys():
    """Sets API keys from the API_keys file."""
    set_environment_variables_from_file(API_KEYS_PATH)


def load_models_config(models_path: str) -> Dict[str, Any]:
    """Load the models configuration from YAML file."""
    models_config = load_yaml_config(models_path)
    if 'llm_models' not in models_config:
        raise ValueError(f"No 'llm_models' section found in {models_path}")
    return models_config['llm_models']


def create_classifier_agents(models_dict: Dict[str, Any], 
                            domain_prompt_path: str,
                            masculinity_prompt_path: str,
                            model_name: str = "gpt4o") -> tuple:
    """
    Create domain and masculinity classifier agents.
    
    Args:
        models_dict: Dictionary of available models from models.yaml
        domain_prompt_path: Path to domain classification prompt
        masculinity_prompt_path: Path to masculinity scoring prompt
        model_name: Name of the model to use from models.yaml
    
    Returns:
        Tuple of (domain_agent, masculinity_agent)
    """
    # Load prompts
    domain_instruction = _load_prompt(domain_prompt_path)
    masculinity_instruction = _load_prompt(masculinity_prompt_path)
    
    # Get model config
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not found in models configuration. "
                       f"Available models: {list(models_dict.keys())}")
    
    model_config = models_dict[model_name]
    logger.info(f"Using model: {model_name} (provider: {model_config.get('provider')})")
    
    # Create LLM clients with system instructions
    domain_llm_client = LLMFactory.create_llm_client(model_config, domain_instruction)
    masculinity_llm_client = LLMFactory.create_llm_client(model_config, masculinity_instruction)
    
    # Create agents
    domain_agent = DomainClassifierAgent(
        llm_client=domain_llm_client,
        agent_name="DomainClassifier",
        model_config=model_config.get('default_config', {})
    )
    
    masculinity_agent = MasculinityClassifierAgent(
        llm_client=masculinity_llm_client,
        agent_name="MasculinityScorer",
        model_config=model_config.get('default_config', {})
    )
    
    return domain_agent, masculinity_agent


def classify_claims(claims_df: pd.DataFrame,
                    domain_agent: DomainClassifierAgent,
                    masculinity_agent: MasculinityClassifierAgent,
                    n_claims: int) -> pd.DataFrame:
    """
    Classify claims using both agents.
    Processes only even indices (0, 2, 4, 6...) to avoid negations.
    
    Args:
        claims_df: DataFrame with claims
        domain_agent: Agent for domain classification
        masculinity_agent: Agent for masculinity scoring
        n_claims: Number of claims to process
    
    Returns:
        DataFrame with columns: claim_id, claim, domain, masculine_score
    """
    results = []
    
    # Calculate even indices to process: 0, 2, 4, 6, ... up to n_claims
    total_rows = len(claims_df)
    max_even_idx = min(n_claims * 2, total_rows)
    even_indices = range(0, max_even_idx, 2)
    
    logger.info(f"Processing {len(even_indices)} claims (even indices 0 to {max_even_idx-2} by 2s)")
    
    # Progress bar
    pbar = tqdm(even_indices, desc="Classifying Claims")
    
    for idx in pbar:
        try:
            row = claims_df.iloc[idx]
            claim_id = row.get('id', f"unknown_{idx}")
            claim_text = row.get('claim', '')
            
            if not claim_text:
                logger.warning(f"Empty claim at index {idx}, skipping")
                continue
            
            # Call domain agent first
            domain = domain_agent.call(claim_text)
            
            # Then score the domain (not the claim)
            masculine_score = masculinity_agent.call(domain)
            
            # Store result
            results.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'domain': domain,
                'masculine_score': masculine_score
            })
            
            # Update progress bar with current classification
            pbar.set_postfix({'domain': domain[:20], 'score': masculine_score})
            
        except Exception as e:
            logger.error(f"Error processing claim at index {idx}: {e}", 
                        extra={"msg_type": "system"})
            # Continue with next claim
            continue
    
    pbar.close()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def main():
    print("=" * 80)
    print("CLAIM CLASSIFICATION")
    print("=" * 80)
    parser = argparse.ArgumentParser(
        description="Classify claims by domain and masculinity score"
    )
    parser.add_argument("--n", type=int, default=100,
                       help="Number of claims to process (processes even indices 0,2,4...)")
    parser.add_argument("--input", default="./claims/all-claim-not-claim.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", default="./claims/claims_classified.csv",
                       help="Output CSV file path")
    parser.add_argument("--models_path", default="./config/models.yaml",
                       help="Path to models configuration file")
    parser.add_argument("--model_name", default="gpt4o",
                       help="Model name to use from models.yaml (default: gpt4o)")
    
    args = parser.parse_args()
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Processing: {args.n} claims (even indices only)")
    print(f"Model: {args.model_name}")
    print()
    
    try:
        # Setup API keys
        _setup_api_keys()
        
        # Load models configuration
        logger.info(f"Loading models from: {args.models_path}")
        models_dict = load_models_config(args.models_path)
        
        # Load claims
        logger.info(f"Loading claims from: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Claims file not found: {args.input}")
        
        claims_df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(claims_df)} total claims")
        
        # Validate we have enough claims
        if args.n * 2 > len(claims_df):
            logger.warning(f"Requested {args.n} claims but only {len(claims_df)//2} pairs available")
            args.n = len(claims_df) // 2
        
        # Create classifier agents
        logger.info("Creating classifier agents...")
        domain_agent, masculinity_agent = create_classifier_agents(
            models_dict=models_dict,
            domain_prompt_path="./prompts/classifier/domain_instruction.txt",
            masculinity_prompt_path="./prompts/classifier/masculinity_instruction.txt",
            model_name=args.model_name
        )
        logger.info("Agents created successfully")
        
        # Classify claims
        print("\nStarting classification...")
        results_df = classify_claims(
            claims_df=claims_df,
            domain_agent=domain_agent,
            masculinity_agent=masculinity_agent,
            n_claims=args.n
        )
        
        # Save results
        logger.info(f"Saving results to: {args.output}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results_df.to_csv(args.output, index=False)
        
        print()
        print("=" * 80)
        print("CLASSIFICATION COMPLETE")
        print("=" * 80)
        print(f"Processed: {len(results_df)} claims")
        print(f"Results saved to: {args.output}")
        print()
        
        # Show sample results
        if len(results_df) > 0:
            print("Sample results (first 5):")
            print(results_df.head().to_string(index=False))
            print()
        
        logger.info("Classification completed successfully")
        
    except Exception as e:
        logger.error(f"Classification failed: {e}", extra={"msg_type": "system"})
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
