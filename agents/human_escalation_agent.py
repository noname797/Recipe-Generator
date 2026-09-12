from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import HumanEscalationOutput
from utils.prompt import human_escalation_agent_prompt


class HumanEscalationAgent(BaseLLMAgent):
    """
    Packages context and suggested responses for a human reviewer/operator UI to review and continue or override.
    """

    output_model = HumanEscalationOutput
    fixture_key = "human_escalation_agent"
    agent_name = "HumanEscalationAgent"

    def escalate(self, state: RecipeState) -> RecipeState:
        """
        Escalate to human using LLM, validate with Pydantic, and update state.
        """
        state.human_escalation = self.run(state)
        return state

    def _build_prompt(self, state: RecipeState) -> str:
        return human_escalation_agent_prompt().format(
            output_instruction=self.output_schema_yaml(), state=state.to_dict()
        )
