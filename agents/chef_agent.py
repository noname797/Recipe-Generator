from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import ChefAgentOutput, IngredientInputOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import chef_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class ChefAgent:
    """
    Generates multiple recipe suggestions based on normalized ingredients and user preferences.
    Maximizes ingredient usage, aligns with dietary and flavor constraints, and provides detailed metadata for each recipe option.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=ChefAgentOutput)

    def generate_recipes(self, state: RecipeState) -> RecipeState:
        """
        Generate recipes using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["chef_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"ChefAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.chef_output = validated

        return state

    def _build_prompt(self, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            ChefAgentOutput.model_json_schema(), sort_keys=False, indent=2
        )
        ingredients = state.ingredient_input
        normalized_ingredients = ingredients.normalized_ingredients
        dietary_preferences = ingredients.dietary_preferences
        allergies = ingredients.allergies
        cuisine = ingredients.cuisine
        flavor_profiles = ingredients.flavor_profiles
        servings = ingredients.servings

        return chef_agent_prompt().format(
            output_instruction=output_schema,
            normalized_ingredients=normalized_ingredients,
            dietary_preferences=dietary_preferences,
            allergies=allergies,
            cuisine=cuisine,
            flavor_profiles=flavor_profiles,
            servings=servings,
        )
