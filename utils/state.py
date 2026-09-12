# State management for the multi-agent workflow


from typing import Optional
from utils.models import (
    IngredientInputOutput,
    ChefAgentOutput,
    NutritionAgentOutput,
    FinalRecipeOutput,
    NutritionRecipeInfo,
    RecipeMetadata,
    SentimentEscalationOutput,
    HumanEscalationOutput,
    UserModificationOutput,
)
from pydantic import BaseModel


class RecipeState(BaseModel):
    """
    Central state object for the multi-agent workflow.
    Tracks all LLM agent outputs using strongly typed Pydantic models for reliability and validation.
    """

    ingredient_input: Optional[IngredientInputOutput] = None
    chef_output: Optional[ChefAgentOutput] = None
    selected_recipe: Optional[RecipeMetadata] = None
    nutrition_info: Optional[NutritionAgentOutput] = None
    selected_nutrition_info: Optional[NutritionRecipeInfo] = None
    user_modifications: Optional[UserModificationOutput] = None
    final_recipe: Optional[FinalRecipeOutput] = None
    sentiment_escalation: Optional[SentimentEscalationOutput] = None
    human_escalation: Optional[HumanEscalationOutput] = None
    test_mode: bool = False  # Flag to indicate test mode for fixed outputs

    def to_dict(self):
        """Return a serializable dict representation of the state."""
        return self.model_dump()
