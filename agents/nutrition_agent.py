from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import ChefAgentOutput, NutritionAgentOutput
from utils.prompt import nutrition_agent_prompt


class NutritionAgent(BaseLLMAgent):
    """
    Analyzes the nutritional profile of each recipe, suggests add-ons and new ingredients to enhance nutritional value,
    and synthesizes user preferences with recipe content to deliver actionable nutrition insights.
    """

    output_model = NutritionAgentOutput
    fixture_key = "nutrition_agent"
    agent_name = "NutritionAgent"

    def analyze_nutrition(self, state: RecipeState) -> RecipeState:
        """
        Analyze nutrition using LLM, validate with Pydantic, and update state.
        """
        state.nutrition_info = self.run(state)
        return state

    def _build_prompt(self, state: RecipeState) -> str:
        recipes = self.chef_agent_output_to_text(state.chef_output)
        return nutrition_agent_prompt().format(
            recipes=recipes,
            dietary_preferences=state.ingredient_input.dietary_preferences,
            allergies=state.ingredient_input.allergies,
            output_instruction=self.output_schema_yaml(),
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
