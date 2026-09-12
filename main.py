from agents.ingredient_input_agent import IngredientInputAgent

from agents.chef_agent import ChefAgent
from agents.nutrition_agent import NutritionAgent
from agents.user_selection_agent import UserSelectionAgent
from agents.user_modification_agent import UserModificationAgent
from agents.final_recipe_agent import FinalRecipeAgent
from agents.sentiment_escalation_agent import SentimentEscalationAgent
from agents.human_escalation_agent import HumanEscalationAgent
from utils.state import RecipeState


import sys

from workflow import run_workflow


def main(test_mode=False):
    run_workflow(test_mode=test_mode)


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
