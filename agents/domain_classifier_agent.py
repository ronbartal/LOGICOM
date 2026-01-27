from typing import Any, Optional, Dict

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface
from utils.log_main import logger

class DomainClassifierAgent(BaseAgent):
    """Agent responsible for classifying a claim's domain/topic in 1-4 words."""

    def __init__(self,
                 llm_client: LLMInterface,
                 agent_name: str = "DomainClassifierAgent", 
                 model_config: Optional[Dict[str, Any]] = None):

        # Pass None for memory to BaseAgent, since classifiers don't need memory
        super().__init__(llm_client=llm_client, memory=None, agent_name=agent_name, model_config=model_config)

    def call(self, claim: str) -> str:
        """
        Classifies the domain/topic of a claim.

        Args:
            claim: The claim text to classify.

        Returns:
            The domain as a string (1-4 words), or empty string if classification fails.
        """
        if not claim or not isinstance(claim, str):
            logger.warning(f"{self.agent_name} received invalid claim input: {type(claim)}")
            return ""

        # Construct the prompt with the claim
        user_content = f"Claim: \"{claim}\""
        prompt = [{"role": "user", "content": user_content}]

        try:
            response_content = self._generate_response(prompt)
            
            # Clean the response (remove extra whitespace, newlines)
            domain = response_content.strip() if response_content else ""
            
            logger.debug(f"{self.agent_name} classified domain: '{domain}' for claim: '{claim[:50]}...'")
            return domain
            
        except Exception as e:
            logger.error(f"{self.agent_name} failed to classify claim: {e}", 
                        extra={"msg_type": "system"})
            return ""
