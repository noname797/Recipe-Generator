from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import FinalRecipeOutput
from utils.prompt import final_recipe_agent_prompt


class FinalRecipeAgent(BaseLLMAgent):
    """
    Synthesizes all prior outputs into a complete, step-by-step recipe, including ingredients, instructions, tips, serving suggestions, and nutritional data.
    """

    output_model = FinalRecipeOutput
    fixture_key = "final_recipe_agent"
    agent_name = "FinalRecipeAgent"

    def finalize_recipe(self, state: RecipeState) -> RecipeState:
        """
        Synthesize final recipe using LLM, validate with Pydantic, and update state.
        """
        state.final_recipe = self.run(state)
        return state

    def _build_prompt(self, state: RecipeState) -> str:
        selected_recipe = self.selected_recipe_to_text(state.selected_recipe)
        selected_nutrition_info = self.selected_nutritional_info_to_text(
            state.selected_nutrition_info
        )
        user_modifications = state.user_modifications.request

        return final_recipe_agent_prompt().format(
            output_instruction=self.output_schema_yaml(),
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
