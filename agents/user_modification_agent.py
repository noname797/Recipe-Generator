from utils.aws_bedrock import BedrockClient
from utils.state import RecipeState
from utils.models import UserModificationOutput
from pydantic import ValidationError
import yaml


class UserModificationAgent:
    """
    Processes specific nutritional adjustment requests and ingredient modification requests.
    Modifies recipes and updates nutritional profiles accordingly.
    Ensures user-driven changes are reflected in both the recipe and the overall workflow state.
    """

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        if not test_mode:
            self.llm = BedrockClient()

    def modify_recipe(self, user_mod_request: str, state: RecipeState) -> RecipeState:
        """
        Process user modification request, validate with Pydantic, and update state.
        """
        # In a real LLM scenario, you would parse LLM output here. For now, we simulate.
        output = {"request": user_mod_request, "applied": True}
        try:
            validated = UserModificationOutput(**output)
            state.user_modifications = validated
        except ValidationError as e:
            raise RuntimeError(f"UserModificationAgent output validation failed: {e}")
        return state
