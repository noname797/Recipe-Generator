from agents.base_agent import BaseLLMAgent
from utils.state import RecipeState
from utils.models import IngredientInputOutput
from utils.prompt import ingredient_input_agent_prompt


class IngredientInputAgent(BaseLLMAgent):
    """
    Handles initial collection of user-provided ingredients, dietary preferences, allergies, cuisine, and flavor profiles.
    Manages ambiguity resolution for ingredient inputs, ensuring downstream agents receive precise and actionable data.
    """

    output_model = IngredientInputOutput
    fixture_key = "ingredient_input_agent"
    agent_name = "IngredientInputAgent"

    def collect_ingredients(self, user_input: str, state: RecipeState) -> RecipeState:
        """
        Collect and parse user input using LLM, validate with Pydantic, and update state.
        """
        state.ingredient_input = self.run(user_input, state)
        return state

    def _build_prompt(self, user_input: str, state: RecipeState) -> str:
        return ingredient_input_agent_prompt().format(
            output_instruction=self.output_schema_yaml(), user_input=user_input
        )
