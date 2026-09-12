from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import ChefAgentOutput, NutritionAgentOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import nutrition_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class NutritionAgent:
    """
    Analyzes the nutritional profile of each recipe, suggests add-ons and new ingredients to enhance nutritional value,
    and synthesizes user preferences with recipe content to deliver actionable nutrition insights.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=NutritionAgentOutput)

    def analyze_nutrition(self, state: RecipeState) -> RecipeState:
        """
        Analyze nutrition using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["nutrition_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"NutritionAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.nutrition_info = validated

        return state

    def _build_prompt(self, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            NutritionAgentOutput.model_json_schema(), sort_keys=False, indent=2
        )
        recipes = self.chef_agent_output_to_text(state.chef_output)
        dietary_preferences = state.ingredient_input.dietary_preferences
        allergies = state.ingredient_input.allergies
        return nutrition_agent_prompt().format(
            recipes=recipes,
            dietary_preferences=dietary_preferences,
            allergies=allergies,
            output_instruction=output_schema,
        )

    def chef_agent_output_to_text(self, output: ChefAgentOutput) -> str:
        lines = [""]
        for idx, recipe in enumerate(output.recipes, 1):
            lines.append(f"  Recipe {idx}:")
            lines.append(f"    Name: {recipe.name}")
            lines.append(f"    Description: {recipe.description}")
            lines.append(f"    Ingredients: {', '.join(recipe.ingredients)}")
            lines.append(f"    Instructions:")
            for idx, instruction in enumerate(recipe.instructions, 1):
                lines.append(f"      {idx}. {instruction}")
            lines.append(f"    Prep Time: {recipe.prep_time}")
            lines.append(f"    Cook Time: {recipe.cook_time}")
            lines.append(f"    Servings: {recipe.servings}")
            if recipe.tags:
                lines.append(f"    Tags: {', '.join(recipe.tags)}")
            lines.append("")  # Blank line between recipes
        return "\n".join(lines)
