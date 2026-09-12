from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import FinalRecipeOutput
import yaml as pyyaml
from pydantic import ValidationError
from utils.test_yaml import fixed_yaml_definitions
from utils.prompt import final_recipe_agent_prompt
from langchain.output_parsers.yaml import YamlOutputParser
import yaml


class FinalRecipeAgent:
    """
    Synthesizes all prior outputs into a complete, step-by-step recipe, including ingredients, instructions, tips, serving suggestions, and nutritional data.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()
        self.yaml_parser = YamlOutputParser(pydantic_object=FinalRecipeOutput)

    def finalize_recipe(self, state: RecipeState) -> RecipeState:
        """
        Synthesize final recipe using LLM, validate with Pydantic, and update state.
        """
        if self.test_mode:
            fixed_yaml = fixed_yaml_definitions["final_recipe_agent"]
            output = yaml.safe_load(fixed_yaml)
        else:
            prompt = self._build_prompt(state)
            response = self.llm.invoke(prompt)
            try:
                output = self.yaml_parser.parse(response)
            except Exception as e:
                raise RuntimeError(
                    f"FinalRecipeAgent: Failed to parse YAML from LLM response: {e}\nRaw response: {response}"
                )

        validated = output
        state.final_recipe = validated
        return state

    def _build_prompt(self, state: RecipeState) -> str:
        # Dynamically generate output instruction from Pydantic model
        output_schema = pyyaml.safe_dump(
            FinalRecipeOutput.model_json_schema(), sort_keys=False, indent=2
        )
        selected_recipe = self.selected_recipe_to_text(state.selected_recipe)
        selected_nutrition_info = self.selected_nutritional_info_to_text(
            state.selected_nutrition_info
        )
        user_modifications = state.user_modifications.request

        return final_recipe_agent_prompt().format(
            output_instruction=output_schema,
            selected_recipe=selected_recipe,
            selected_nutrition_info=selected_nutrition_info,
            user_modifications=user_modifications,
        )

    def selected_recipe_to_text(self, recipe) -> str:
        lines = [""]
        lines.append(f"  Name: {recipe.name}")
        lines.append(f"  Description: {recipe.description}")
        lines.append(f"  Ingredients: {', '.join(recipe.ingredients)}")
        lines.append(f"  Instructions:")
        for idx, instruction in enumerate(recipe.instructions, 1):
            lines.append(f"    {idx}. {instruction}")
        lines.append(f"  Prep Time: {recipe.prep_time}")
        lines.append(f"  Cook Time: {recipe.cook_time}")
        lines.append(f"  Servings: {recipe.servings}")
        if recipe.tags:
            lines.append(f"  Tags: {', '.join(recipe.tags)}")
        lines.append("")  # Blank line between recipes
        return "\n".join(lines)

    def selected_nutritional_info_to_text(self, nutrition_info) -> str:
        lines = []
        np = nutrition_info.nutrition_profile
        lines.append(
            f"  Nutrition Profile: "
            f"Calories: {np.calories} kcal, "
            f"Protein: {np.protein}g, "
            f"Fat: {np.fat}g, "
            f"Carbs: {np.carbs}g"
            + (f", Fiber: {np.fiber}g" if np.fiber is not None else "")
            + (f", Sodium: {np.sodium}mg" if np.sodium is not None else "")
        )
        if nutrition_info.suggested_addons:
            lines.append(
                f"  Suggested Add-ons: {', '.join(nutrition_info.suggested_addons)}"
            )
        if nutrition_info.nutrition_insights:
            lines.append(
                f"  Nutrition Insights: {', '.join(nutrition_info.nutrition_insights)}"
            )
        lines.append("")  # Blank line between recipes
        return "\n".join(lines)
