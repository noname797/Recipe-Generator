"""
Unit tests for individual agents, exercised in `test_mode` so no real LLM
calls (and no AWS credentials) are required.
"""
import pytest

from agents.ingredient_input_agent import IngredientInputAgent
from agents.chef_agent import ChefAgent
from agents.nutrition_agent import NutritionAgent
from agents.user_selection_agent import UserSelectionAgent
from agents.user_modification_agent import UserModificationAgent
from agents.final_recipe_agent import FinalRecipeAgent
from agents.sentiment_escalation_agent import SentimentEscalationAgent
from agents.human_escalation_agent import HumanEscalationAgent
from utils.state import RecipeState


@pytest.fixture
def state():
    return RecipeState(test_mode=True)


def test_ingredient_input_agent(state):
    agent = IngredientInputAgent(test_mode=True)
    state = agent.collect_ingredients("chicken, rice", state)
    assert state.ingredient_input is not None
    assert "test_ingredient1" in state.ingredient_input.normalized_ingredients


def test_chef_agent(state):
    state = IngredientInputAgent(test_mode=True).collect_ingredients("chicken, rice", state)
    state = ChefAgent(test_mode=True).generate_recipes(state)
    assert state.chef_output is not None
    assert len(state.chef_output.recipes) == 2


def test_nutrition_agent(state):
    state = IngredientInputAgent(test_mode=True).collect_ingredients("chicken, rice", state)
    state = ChefAgent(test_mode=True).generate_recipes(state)
    state = NutritionAgent(test_mode=True).analyze_nutrition(state)
    assert state.nutrition_info is not None
    assert len(state.nutrition_info.recipes) == 2


def _state_with_recipes(state):
    state = IngredientInputAgent(test_mode=True).collect_ingredients("chicken, rice", state)
    state = ChefAgent(test_mode=True).generate_recipes(state)
    state = NutritionAgent(test_mode=True).analyze_nutrition(state)
    return state


def test_user_selection_agent_valid_choice(state):
    state = _state_with_recipes(state)
    agent = UserSelectionAgent()
    state = agent.select_recipe(state, "1")
    assert state.selected_recipe == state.chef_output.recipes[0]
    assert state.selected_nutrition_info == state.nutrition_info.recipes[0]


@pytest.mark.parametrize("bad_choice", ["0", "99", "abc", None])
def test_user_selection_agent_invalid_choice(state, bad_choice):
    state = _state_with_recipes(state)
    agent = UserSelectionAgent()
    with pytest.raises(ValueError):
        agent.select_recipe(state, bad_choice)


def test_user_modification_agent(state):
    state = _state_with_recipes(state)
    state = UserSelectionAgent().select_recipe(state, "1")
    state = UserModificationAgent(test_mode=True).modify_recipe("less salt", state)
    assert state.user_modifications is not None
    assert state.user_modifications.applied is True


def test_final_recipe_agent(state):
    state = _state_with_recipes(state)
    state = UserSelectionAgent().select_recipe(state, "1")
    state = UserModificationAgent(test_mode=True).modify_recipe("less salt", state)
    state = FinalRecipeAgent(test_mode=True).finalize_recipe(state)
    assert state.final_recipe is not None
    assert state.final_recipe.name == "Test Final Recipe"


def test_sentiment_escalation_agent(state):
    agent = SentimentEscalationAgent(test_mode=True)
    state = agent.analyze_sentiment("This is frustrating!", state)
    assert state.sentiment_escalation is not None
    assert state.sentiment_escalation.escalation_required is True


def test_human_escalation_agent(state):
    agent = HumanEscalationAgent(test_mode=True)
    state = agent.escalate(state)
    assert state.human_escalation is not None
    assert "review" in state.human_escalation.human_notes.lower()
