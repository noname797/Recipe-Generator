from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import HumanEscalationOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import human_escalation_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class HumanEscalationAgent:
    """
    Packages context and suggested responses for a human reviewer/operator UI to review and continue or override.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=HumanEscalationOutput)

    def escalate(self, state: RecipeState) -> RecipeState:
        """
        Escalate to human using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["human_escalation_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"HumanEscalationAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.human_escalation = validated

        return state

    def _build_prompt(self, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            HumanEscalationOutput.model_json_schema(), sort_keys=False, indent=2
        )
        return human_escalation_agent_prompt().format(
            output_instruction=output_schema, state=state.to_dict()
        )
