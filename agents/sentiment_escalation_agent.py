from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import SentimentEscalationOutput
from utils.prompt import sentiment_escalation_agent_prompt


class SentimentEscalationAgent(BaseLLMAgent):
    """
    Monitors user responses and sentiment across interactions; if frustration or urgency detected, flips escalation_required flag for human handoff.
    """

    output_model = SentimentEscalationOutput
    fixture_key = "sentiment_escalation_agent"
    agent_name = "SentimentEscalationAgent"

    def analyze_sentiment(
        self, user_responses: str, state: RecipeState
    ) -> RecipeState:
        """
        Analyze sentiment using LLM, validate with Pydantic, and update state.
        """
        state.sentiment_escalation = self.run(user_responses, state)
        return state

    def _build_prompt(self, user_responses: list, state: RecipeState) -> str:
        return sentiment_escalation_agent_prompt().format(
            user_responses=user_responses,
            state=state.to_dict(),
            output_instruction=self.output_schema_yaml(),
        )
