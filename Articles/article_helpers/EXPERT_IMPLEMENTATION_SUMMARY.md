# Expert Persona Experiment - Simplified Implementation Summary

## Overview

Simplified implementation that leverages existing codebase patterns, similar to `test_gender_combinations.py`.

## Files Created

### 1. Prompt Files (6 .txt files)
```
prompts/persuader/
├── expert_education_persona.txt    # Education expert credentials
├── expert_education_initial.txt    # Education expert opening
├── expert_diplomat_persona.txt     # Diplomat expert credentials  
├── expert_diplomat_initial.txt     # Diplomat expert opening
├── expert_health_persona.txt       # Health expert credentials
└── expert_health_initial.txt       # Health expert opening
```

### 2. Expert Agent Class (`agents/expert_agents.py` - 17 lines)

**`ExpertPersuaderAgent`** - Simple wrapper that prepends expert persona to prompt wrapper
- Inherits from `PersuaderAgent`
- Only modification: prepends expert persona to the prompt wrapper
- Everything else uses parent class functionality

### 3. Campaign Script (`run_expert_campaign.py` - 149 lines)

Similar structure to `test_gender_combinations.py`:
- Uses `_run_single_debate` from `main.py` (no custom agent creation)
- Uses existing `DebateInstanceSetup` infrastructure
- Leverages existing Excel logging via `save_debate_in_excel`
- Simple prompt selection function `select_prompts_for_expert`

## Key Design Decisions

### 1. Prompts in .txt Files ✓
Following existing codebase patterns, expert personas are stored as text files in `prompts/persuader/` directory.

### 2. Minimal Code Through Better Inheritance ✓
- **Before**: 298 lines in expert_agents.py with 3 subclasses
- **After**: 17 lines with 1 simple wrapper class
- No need for subclasses - prompt files handle the differences

### 3. Leverage Existing Infrastructure ✓
- Uses `_run_single_debate` from `main.py` (same as `test_gender_combinations.py`)
- Uses `DebateInstanceSetup` from `core/debate_setup.py` (lines 222-223 pattern)
- Uses `save_debate_in_excel` for logging (existing infrastructure)
- No custom `_create_standard_debater` - setup handles it

### 4. No Helper Complexity ✓
- Runs with `Default_No_Helper` by default
- Removed 400+ lines of helper-related code
- Simpler and cleaner implementation

### 5. Consistent with `test_gender_combinations.py` ✓
Both scripts now follow the same pattern:
1. Load configuration
2. Load and filter claims CSV
3. Select appropriate prompts (gender vs expert)
4. Loop through claims and repetitions
5. Call `_run_single_debate` with appropriate parameters
6. Use existing logging infrastructure

## Expert Domains

| Domain Code | Expert Type | Persona File | Initial Prompt File |
|-------------|-------------|--------------|---------------------|
| E | Education | expert_education_persona.txt | expert_education_initial.txt |
| I | Diplomat | expert_diplomat_persona.txt | expert_diplomat_initial.txt |
| H | Health | expert_health_persona.txt | expert_health_initial.txt |

## Usage

```bash
# Basic usage (k=3 repetitions per claim, no helper)
python run_expert_campaign.py

# With custom repetitions
python run_expert_campaign.py --k 5

# With max rounds override
python run_expert_campaign.py --k 3 --max_rounds 10

# With custom debates directory
python run_expert_campaign.py --debates_dir my_expert_debates
```

## Output Structure

Same as existing infrastructure:
- Debate logs: `debates/{topic_id}/Expert-{Education|Diplomat|Health}/{chat_id}/`
- Excel results: `debates_results.xlsx` (via existing `save_debate_in_excel`)
- Same log format as gender experiments for consistency

## Code Comparison

### Before (Complex)
- `agents/expert_agents.py`: 298 lines
- `run_expert_campaign.py`: 642 lines
- **Total**: 940 lines
- Custom agent creation, custom moderators, custom CSV saving

### After (Simplified)
- `agents/expert_agents.py`: 17 lines
- `run_expert_campaign.py`: 149 lines
- 6 prompt .txt files
- **Total**: 166 lines of Python code
- **82% reduction in code** through better reuse

## How It Works

### Expert Prompt Selection
```python
def select_prompts_for_expert(loaded_prompts: dict, expert_domain: str) -> dict:
    # Load expert-specific persona and initial prompt from .txt files
    expert_persona = _load_prompt(f"prompts/persuader/expert_{type}_persona.txt")
    expert_initial = _load_prompt(f"prompts/persuader/expert_{type}_initial.txt")
    
    # Prepend persona to wrapper, replace initial prompt
    prompts['persuader_wrapper'] = f"{expert_persona}\n\n{base_wrapper}"
    prompts['persuader_initial'] = expert_initial
    
    return prompts
```

### Main Loop (Same Pattern as test_gender_combinations.py)
```python
for claim in expert_claims:
    for rep in range(k):
        selected_prompts = select_prompts_for_expert(prompt_templates, expert_domain)
        
        _run_single_debate(
            claim_data=claim,
            prompt_templates=selected_prompts,
            gender_case=f"Expert-{expert_type_name}",
            # ... other existing parameters
        )
```

## Validation

- ✅ Personas stored in .txt files (like existing prompts)
- ✅ Minimal code through inheritance
- ✅ Uses `_run_single_debate` pattern from `test_gender_combinations.py`
- ✅ No custom agent creation (uses `DebateInstanceSetup`)
- ✅ No helper complexity (runs with Default_No_Helper)
- ✅ Same logging infrastructure as existing code
- ✅ 82% code reduction while maintaining functionality
- ✅ No linter errors

## Expert Personas

All three expert personas are gender-neutral with professional credentials:

- **Education Expert**: Distinguished Professor with 30 years research, published 100+ papers
- **Diplomat Expert**: Senior Diplomat with 25 years in conflict zones, negotiated treaties
- **Health Expert**: Chief Medical Officer, Board-Certified Epidemiologist, published in Lancet/JAMA

Each persona includes:
- Detailed credentials and experience
- Field-specific expertise areas
- Argumentation style guidance (reference theories, cite research)
- Professional tone expectations
