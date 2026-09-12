from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import ChefAgentOutput
from utils.prompt import chef_agent_prompt


class ChefAgent(BaseLLMAgent):
    """
    Generates multiple recipe suggestions based on normalized ingredients and user preferences.
    Maximizes ingredient usage, aligns with dietary and flavor constraints, and provides detailed metadata for each recipe option.
    """

    output_model = ChefAgentOutput
    fixture_key = "chef_agent"
    agent_name = "ChefAgent"

    def generate_recipes(self, state: RecipeState) -> RecipeState:
        """
        Generate recipes using LLM, validate with Pydantic, and update state.
        """
        state.chef_output = self.run(state)
        return state

    def _build_prompt(self, state: RecipeState) -> str:
        ingredients = state.ingredient_input
        return chef_agent_prompt().format(
            output_instruction=self.output_schema_yaml(),
            normalized_ingredients=ingredients.normalized_ingredients,
            dietary_preferences=ingredients.dietary_preferences,
            allergies=ingredients.allergies,
            cuisine=ingredients.cuisine,
            flavor_profiles=ingredients.flavor_profiles,
            servings=ingredients.servings,
        )
