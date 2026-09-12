"""
LangGraph workflow for Multi-Agent Recipe Generator
"""
import sys
from langgraph.graph import StateGraph, END
from agents.ingredient_input_agent import IngredientInputAgent
from agents.chef_agent import ChefAgent
from agents.nutrition_agent import NutritionAgent
from agents.user_selection_agent import UserSelectionAgent
from agents.user_modification_agent import UserModificationAgent
from agents.final_recipe_agent import FinalRecipeAgent
from agents.sentiment_escalation_agent import SentimentEscalationAgent
from agents.human_escalation_agent import HumanEscalationAgent
from utils.state import RecipeState

# Node functions for each agent

def ingredient_node(state, user_input=None):
    # user_input must be provided by caller (web or CLI)
    if user_input is None:
        raise ValueError("user_input must be provided to ingredient_node")
    agent = IngredientInputAgent(test_mode=state.test_mode)
    state = agent.collect_ingredients(user_input, state)
    # print("\nParsed Input:")
    # print(state.ingredient_input.model_dump())
    return state

def chef_node(state):
    agent = ChefAgent(test_mode=state.test_mode)
    state = agent.generate_recipes(state)
    # print("\nRecipe Suggestions:")

    # for idx, recipe in enumerate(state.chef_output.recipes, 1):
    #     print(f"\nRecipe {idx}: {recipe.name}")
    #     print(f"Description: {recipe.description}")
    #     print(f"Ingredients: {recipe.ingredients}")
    #     print(f"Instructions: {recipe.instructions}")
    #     print(
    #         f"Prep Time: {recipe.prep_time}, Cook Time: {recipe.cook_time}, Servings: {recipe.servings}"
    #     )
    #     print(f"Tags: {recipe.tags}")
    #     print(f"Serving Size: {recipe.servings}")

    return state

def nutrition_node(state):
    agent = NutritionAgent(test_mode=state.test_mode)
    state = agent.analyze_nutrition(state)
    return state

def user_selection_node(state):
    agent = UserSelectionAgent()
    state = agent.select_recipe(state)
    print(f"\nYou selected: {state.selected_recipe.name}")
    return state

def user_modification_node(state, user_mod_request=None):
    # user_mod_request must be provided by caller (web or CLI)
    if user_mod_request and user_mod_request.strip():
        agent = UserModificationAgent(test_mode=state.test_mode)
        state = agent.modify_recipe(user_mod_request, state)
        # print("\nModification request:")
        # print(user_mod_request)
    return state

def final_recipe_node(state):
    agent = FinalRecipeAgent(test_mode=state.test_mode)
    state = agent.finalize_recipe(state)
    return state

def sentiment_node(state, user_feedback=None):
    # user_feedback must be provided by caller (web or CLI)
    agent = SentimentEscalationAgent(test_mode=state.test_mode)
    state = agent.analyze_sentiment(user_feedback or "", state)
    # print(f"\nDetected Sentiment: {state.sentiment_escalation.sentiment}")
    # print(f"Escalation Required: {state.sentiment_escalation.escalation_required}")
    return state

def human_escalation_node(state):
    agent = HumanEscalationAgent(test_mode=state.test_mode)
    state = agent.escalate(state)
    print("\n--- HUMAN ESCALATION ---")
    print(state.human_escalation.human_notes)

    return state

# Build the workflow graph
graph = StateGraph(RecipeState)
graph.add_node("ingredient", ingredient_node)
graph.add_node("chef", chef_node)
graph.add_node("nutrition", nutrition_node)
graph.add_node("user_selection", user_selection_node)
graph.add_node("user_modification", user_modification_node)
graph.add_node("final_recipe", final_recipe_node)
graph.add_node("sentiment", sentiment_node)
graph.add_node("human_escalation", human_escalation_node)

# Define edges
graph.add_edge("ingredient", "chef")
graph.add_edge("chef", "nutrition")
graph.add_edge("nutrition", "user_selection")
graph.add_edge("user_selection", "user_modification")
graph.add_edge("user_modification", "final_recipe")
graph.add_edge("final_recipe", "sentiment")

def sentiment_decision(state):
    if hasattr(state, "sentiment_escalation") and getattr(state.sentiment_escalation, "escalation_required", False):
        return "human_escalation"
    return END

graph.add_conditional_edges("sentiment", sentiment_decision, {"human_escalation": "human_escalation", END: END})
graph.add_edge("human_escalation", END)

graph.set_entry_point("ingredient")
workflow = graph.compile()

def run_workflow(test_mode=False):
    init_state = RecipeState()
    init_state.test_mode = test_mode
    result = workflow.invoke(init_state)
    state = RecipeState(**result)
    print("\nFinal Recipe:")
    print(state.final_recipe)
    if hasattr(state, "sentiment_escalation"):
        print(f"\nDetected Sentiment: {state.sentiment_escalation.sentiment}")
        print(f"Escalation Required: {state.sentiment_escalation.escalation_required}")
    if hasattr(state, "human_escalation"):
        print("\n--- HUMAN ESCALATION ---")
        print(state.human_escalation.human_notes)
    return state
