from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import SentimentEscalationOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import sentiment_escalation_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class SentimentEscalationAgent:
    """
    Monitors user responses and sentiment across interactions; if frustration or urgency detected, flips escalation_required flag for human handoff.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=SentimentEscalationOutput)

    def analyze_sentiment(
        self, user_responses: str, state: RecipeState
    ) -> RecipeState:
        """
        Analyze sentiment using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["sentiment_escalation_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(user_responses, state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"SentimentEscalationAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.sentiment_escalation = validated

        return state

    def _build_prompt(self, user_responses: list, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            SentimentEscalationOutput.model_json_schema(), sort_keys=False, indent=2
        )
        return sentiment_escalation_agent_prompt().format(
            user_responses=user_responses,
            state=state.to_dict(),
            output_instruction=output_schema,
        )
