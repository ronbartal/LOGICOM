"""
Expert Persuader Agents - Simple implementation using existing infrastructure.
These agents only need to override the system prompt to inject expert credentials.
"""
from agents.persuader_agent import PersuaderAgent


class ExpertPersuaderAgent(PersuaderAgent):
    """
    Expert persuader that prepends expert persona to prompt wrapper.
    This is the only modification needed - everything else uses parent class.
    """
    
    def __init__(self, expert_persona: str, *args, **kwargs):
        """Initialize with expert persona prepended to wrapper."""
        # Get the base prompt wrapper from kwargs
        base_wrapper = kwargs.get('prompt_wrapper', '')
        
        # Prepend expert persona to wrapper
        if base_wrapper:
            kwargs['prompt_wrapper'] = f"{expert_persona}\n\n{base_wrapper}"
        else:
            kwargs['prompt_wrapper'] = expert_persona
        
        # Call parent with modified wrapper
        super().__init__(*args, **kwargs)
