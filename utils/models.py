"""
Pydantic v2 models for LLM-powered agent outputs in the Recipe Generator system.
Strict validation and nested typing for robust, reliable LLM output handling.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import (
    BaseModel,
    Field,
    StrictStr,
    StrictInt,
    StrictBool,
    StrictFloat,
    field_validator,
    model_validator,
)


# Ingredient Input Agent Output
class IngredientInputOutput(BaseModel):
    ingredients: List[StrictStr] = Field(
        ..., min_length=1, description="List of user-provided ingredients."
    )
    dietary_preferences: Optional[List[str]] = Field(
        default=["None"], description="List of user-provided dietary preferences."
    )
    allergies: Optional[List[str]] = Field(
        default=["None"], description="List of user-provided allergies."
    )
    cuisine: Optional[str] = Field(
        default=["Any"], description="User-provided cuisine type."
    )
    flavor_profiles: Optional[List[str]] = Field(
        default=["Any"], description="List of user-provided flavor profiles."
    )
    ambiguous_ingredients: Optional[List[str]] = Field(
        default=None,
        description="List of user-provided ambiguous ingredients needing clarification.",
    )
    normalized_ingredients: Optional[List[str]] = Field(
        default=None, description="List of user-provided normalized ingredient names."
    )
    servings: Optional[int] = Field(
        default=1, description="User-provided serving size. eg. for 2 people, 4 servings."
    )


# Chef Agent Output


class RecipeMetadata(BaseModel):
    name: StrictStr = Field(..., description="Name of the recipe.")
    description: StrictStr = Field(..., description="Short description of the recipe.")
    ingredients: List[StrictStr] = Field(
        ..., description="List of ingredients required for the recipe."
    )
    instructions: List[StrictStr] = Field(
        ..., description="List of step-by-step instructions to prepare the recipe. Must have atleast 4 steps."
    )
    prep_time: Optional[Union[str, int]] = Field(
        default="Unknown",
        description="Preparation time for the recipe. Should be a string or int.",
    )
    cook_time: Optional[Union[str, int]] = Field(
        default="Unknown",
        description="Cooking time for the recipe. Should be a string or int.",
    )
    servings: Optional[int] = Field(
        default=1, description="Number of servings."
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="List of tags for the recipe."
    )


class ChefAgentOutput(BaseModel):
    recipes: List[RecipeMetadata] = Field(
        ..., min_length=1, description="Generated recipe suggestions."
    )


# Nutrition Agent Output
class NutritionProfile(BaseModel):
    calories: float = Field(..., description="Total calories in the recipe.")
    protein: float = Field(..., description="Total protein content in grams.")
    fat: float = Field(..., description="Total fat content in grams.")
    carbs: float = Field(..., description="Total carbohydrate content in grams.")
    fiber: Optional[float] = Field(
        default=None, description="Total fiber content in grams, if available."
    )
    sodium: Optional[float] = Field(
        default=None, description="Total sodium content in milligrams, if available."
    )
    # Add more fields as needed


class NutritionRecipeInfo(BaseModel):
    name: StrictStr = Field(..., description="Name of the recipe.")
    nutrition_profile: NutritionProfile = Field(
        ..., description="Detailed nutrition profile for the recipe."
    )
    suggested_addons: List[str] = Field(
        ..., description="List of suggested add-ons to improve nutrition or taste."
    )
    nutrition_insights: List[str] = Field(
        ..., description="List of nutrition-related insights or comments."
    )


class NutritionAgentOutput(BaseModel):
    recipes: List[NutritionRecipeInfo] = Field(
        ..., min_length=1, description="List of nutrition-analyzed recipes."
    )


# Final Recipe Agent Output
class FinalRecipeOutput(BaseModel):
    name: StrictStr = Field(..., description="Name of the final recipe.")
    description: StrictStr = Field(..., description="Description of the final recipe.")
    ingredients: List[StrictStr] = Field(
        ..., description="List of ingredients for the final recipe."
    )
    instructions: List[StrictStr] = Field(
        ..., description="List of step-by-step instructions for the final recipe. Must have atleast 4 steps."
    )
    tips: Optional[List[str]] = Field(
        default=None,
        description="List of optional tips for preparing or serving the recipe.",
    )
    serving_suggestions: Optional[List[str]] = Field(
        default=None, description="List of optional serving suggestions for the recipe."
    )
    nutrition: Optional[NutritionProfile] = Field(
        default=None, description="Optional nutrition profile for the recipe."
    )


# Sentiment Escalation Agent Output
class SentimentEscalationOutput(BaseModel):
    sentiment: str = Field(..., description="Detected sentiment from user feedback.")
    escalation_required: StrictBool = Field(
        ..., description="Whether escalation to another agent is required."
    )


# Human Escalation Agent Output
class HumanEscalationOutput(BaseModel):
    human_notes: str = Field(
        ..., description="Notes or comments provided by a human agent."
    )


# User Modification Agent Output
class UserModificationOutput(BaseModel):
    request: StrictStr = Field(..., description="User's modification request.")
    applied: StrictBool = Field(
        ..., description="Whether the modification was successfully applied."
    )
