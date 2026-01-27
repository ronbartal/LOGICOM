from typing import Any, Optional, Dict

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface
from utils.log_main import logger

class MasculinityClassifierAgent(BaseAgent):
    """Agent responsible for scoring how masculine a domain/topic is (1-10)."""

    def __init__(self,
                 llm_client: LLMInterface,
                 agent_name: str = "MasculinityClassifierAgent", 
                 model_config: Optional[Dict[str, Any]] = None):

        # Pass None for memory to BaseAgent, since classifiers don't need memory
        super().__init__(llm_client=llm_client, memory=None, agent_name=agent_name, model_config=model_config)

    def call(self, domain: str) -> int:
        """
        Scores how masculine a domain/topic is based on stereotypical associations.

        Args:
            domain: The domain/topic text to score.

        Returns:
            Integer score 1-10, or -1 if scoring fails.
        """
        if not domain or not isinstance(domain, str):
            logger.warning(f"{self.agent_name} received invalid domain input: {type(domain)}")
            return -1

        # Construct the prompt with the domain
        user_content = f"Domain: \"{domain}\""
        prompt = [{"role": "user", "content": user_content}]

        try:
            response_content = self._generate_response(prompt)
            
            # Parse the response to extract the numeric score
            score_str = response_content.strip() if response_content else ""
            
            # Try to convert to integer
            try:
                score = int(score_str)
                
                # Validate score is in range 1-10
                if score < 1 or score > 10:
                    logger.warning(f"{self.agent_name} got out-of-range score {score}, clamping to 1-10")
                    score = max(1, min(10, score))
                
                logger.debug(f"{self.agent_name} scored {score} for domain: '{domain}'")
                return score
                
            except ValueError:
                logger.error(f"{self.agent_name} could not parse score from response: '{score_str}'")
                return -1
            
        except Exception as e:
            logger.error(f"{self.agent_name} failed to score domain: {e}", 
                        extra={"msg_type": "system"})
            return -1
