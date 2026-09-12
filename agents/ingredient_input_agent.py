from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import IngredientInputOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import ingredient_input_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class IngredientInputAgent:
    """
    Handles initial collection of user-provided ingredients, dietary preferences, allergies, cuisine, and flavor profiles.
    Manages ambiguity resolution for ingredient inputs, ensuring downstream agents receive precise and actionable data.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=IngredientInputOutput)

    def collect_ingredients(self, user_input: str, state: RecipeState) -> RecipeState:
        """
        Collect and parse user input using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["ingredient_input_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(user_input, state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"IngredientInputAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.ingredient_input = validated

        return state

    def _build_prompt(self, user_input: str, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            IngredientInputOutput.model_json_schema(), sort_keys=False, indent=2
        )
        return ingredient_input_agent_prompt().format(
            output_instruction=output_schema, user_input=user_input
        )
