"""
LangGraph workflow for Multi-Agent Recipe Generator
"""
import logging
import os
import uuid

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from agents.ingredient_input_agent import IngredientInputAgent
from agents.chef_agent import ChefAgent
from agents.nutrition_agent import NutritionAgent
from agents.user_selection_agent import UserSelectionAgent
from agents.user_modification_agent import UserModificationAgent
from agents.final_recipe_agent import FinalRecipeAgent
from agents.sentiment_escalation_agent import SentimentEscalationAgent
from agents.human_escalation_agent import HumanEscalationAgent
from utils.state import RecipeState

logger = logging.getLogger(__name__)

# Node functions for each agent

def ingredient_node(state, user_input=None):
    # user_input must be provided by caller (web or CLI)
    if user_input is None:
        raise ValueError("user_input must be provided to ingredient_node")
    agent = IngredientInputAgent(test_mode=state.test_mode)
    state = agent.collect_ingredients(user_input, state)
    logger.debug("Parsed ingredient input: %s", state.ingredient_input.model_dump())
    return state

def chef_node(state):
    agent = ChefAgent(test_mode=state.test_mode)
    state = agent.generate_recipes(state)
    logger.debug(
        "Generated %d recipe suggestions: %s",
        len(state.chef_output.recipes),
        [r.name for r in state.chef_output.recipes],
    )
    return state

def nutrition_node(state):
    agent = NutritionAgent(test_mode=state.test_mode)
    state = agent.analyze_nutrition(state)
    return state

def user_selection_node(state, user_choice=None):
    agent = UserSelectionAgent()
    if user_choice is None:
        # CLI mode: prompt interactively via input()
        state = agent.prompt_and_select_cli(state)
    else:
        # Web/API mode: choice supplied by caller
        state = agent.select_recipe(state, user_choice)
    logger.info("User selected recipe: %s", state.selected_recipe.name)
    return state

def user_modification_node(state, user_mod_request=None):
    # user_mod_request must be provided by caller (web or CLI)
    if user_mod_request and user_mod_request.strip():
        agent = UserModificationAgent(test_mode=state.test_mode)
        state = agent.modify_recipe(user_mod_request, state)
        logger.debug("Applied user modification request: %s", user_mod_request)
    return state

def final_recipe_node(state):
    agent = FinalRecipeAgent(test_mode=state.test_mode)
    state = agent.finalize_recipe(state)
    return state

def sentiment_node(state, user_feedback=None):
    # user_feedback must be provided by caller (web or CLI)
    agent = SentimentEscalationAgent(test_mode=state.test_mode)
    state = agent.analyze_sentiment(user_feedback or "", state)
    logger.debug(
        "Detected sentiment=%s escalation_required=%s",
        state.sentiment_escalation.sentiment,
        state.sentiment_escalation.escalation_required,
    )
    return state

def human_escalation_node(state):
    agent = HumanEscalationAgent(test_mode=state.test_mode)
    state = agent.escalate(state)
    logger.warning("HUMAN ESCALATION triggered: %s", state.human_escalation.human_notes)

    # Persist the escalation so a human reviewer has somewhere to see and act
    # on it, instead of it only existing ephemerally in this session's state.
    try:
        from utils.escalation_store import save_escalation

        escalation_id = save_escalation(
            state_json=state.model_dump_json(),
            human_notes=state.human_escalation.human_notes,
            sentiment=getattr(state.sentiment_escalation, "sentiment", None),
        )
        logger.info("Escalation persisted with id=%s", escalation_id)
    except Exception:
        logger.exception("Failed to persist escalation to the escalation store")

    return state

# --- Interrupt-aware node wrappers -------------------------------------------
# The plain node functions above take explicit `user_input`/`user_choice`
# arguments so they can be unit-tested and driven directly (see tests/). But
# LangGraph only ever calls a node with `(state)` — it has no built-in way to
# pass ad hoc arguments mid-run. To actually pause the graph and wait for a
# human response (rather than reimplementing the step machine separately in
# app.py), these wrappers use LangGraph's `interrupt()` to pause execution
# and receive the resumed value, then delegate to the plain node functions
# above. This is what makes `interactive_workflow` usable end-to-end by both
# the CLI and the Flask web app.

def ingredient_interrupt_node(state):
    user_input = interrupt(
        {"prompt": "Please enter your ingredients, dietary preferences, allergies, cuisine, and flavor profile:"}
    )
    return ingredient_node(state, user_input)

def user_selection_interrupt_node(state):
    recipes_summary = []
    for idx, recipe in enumerate(state.chef_output.recipes, 1):
        entry = {
            "number": idx,
            "name": recipe.name,
            "description": recipe.description,
            "ingredients": recipe.ingredients,
            "instructions": recipe.instructions,
            "prep_time": getattr(recipe, "prep_time", "N/A"),
            "cook_time": getattr(recipe, "cook_time", "N/A"),
            "servings": getattr(recipe, "servings", "N/A"),
            "tags": getattr(recipe, "tags", None),
        }
        if state.nutrition_info:
            nutrition = state.nutrition_info.recipes[idx - 1]
            entry["nutrition_profile"] = nutrition.nutrition_profile.model_dump()
            entry["suggested_addons"] = nutrition.suggested_addons
            entry["nutrition_insights"] = nutrition.nutrition_insights
        recipes_summary.append(entry)

    user_choice = interrupt(
        {
            "prompt": "Here are some recipe suggestions. Please select a recipe by number.",
            "recipes": recipes_summary,
        }
    )
    return user_selection_node(state, user_choice)

def user_modification_interrupt_node(state):
    user_mod_request = interrupt(
        {"prompt": "Would you like to modify the recipe? Type your modification, or 'no' to continue."}
    )
    return user_modification_node(state, user_mod_request)

def sentiment_interrupt_node(state):
    user_feedback = interrupt(
        {"prompt": "How do you feel about your experience? (Optional feedback for sentiment analysis)"}
    )
    return sentiment_node(state, user_feedback)


def sentiment_decision(state):
    if hasattr(state, "sentiment_escalation") and getattr(state.sentiment_escalation, "escalation_required", False):
        return "human_escalation"
    return END


def _build_interactive_graph(checkpointer):
    """Build the single LangGraph definition shared by the CLI and web app."""
    graph = StateGraph(RecipeState)
    graph.add_node("ingredient", ingredient_interrupt_node)
    graph.add_node("chef", chef_node)
    graph.add_node("nutrition", nutrition_node)
    graph.add_node("user_selection", user_selection_interrupt_node)
    graph.add_node("user_modification", user_modification_interrupt_node)
    graph.add_node("final_recipe", final_recipe_node)
    graph.add_node("sentiment", sentiment_interrupt_node)
    graph.add_node("human_escalation", human_escalation_node)

    graph.add_edge("ingredient", "chef")
    graph.add_edge("chef", "nutrition")
    graph.add_edge("nutrition", "user_selection")
    graph.add_edge("user_selection", "user_modification")
    graph.add_edge("user_modification", "final_recipe")
    graph.add_edge("final_recipe", "sentiment")
    graph.add_conditional_edges(
        "sentiment", sentiment_decision, {"human_escalation": "human_escalation", END: END}
    )
    graph.add_edge("human_escalation", END)

    graph.set_entry_point("ingredient")
    return graph.compile(checkpointer=checkpointer)


# Single shared checkpointer + compiled graph used by both the CLI and the
# Flask web app, so both entry points drive the exact same LangGraph
# definition instead of maintaining two divergent step machines.
#
# By default this uses an in-memory checkpointer, which is fine for local
# development and tests, but means any in-flight (interrupted) session is
# lost if the process restarts, and isn't shared across multiple worker
# processes. Set `CHECKPOINT_DB_PATH` to persist checkpoints to a SQLite
# database instead (recommended for anything beyond local dev), e.g.:
#
#   CHECKPOINT_DB_PATH=./checkpoints.db
def _build_checkpointer():
    db_path = os.environ.get("CHECKPOINT_DB_PATH")
    if not db_path:
        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("Using persistent SQLite checkpointer at %s", db_path)
        return saver
    except ImportError:
        logger.warning(
            "CHECKPOINT_DB_PATH is set but 'langgraph-checkpoint-sqlite' is not "
            "installed; falling back to in-memory checkpointer. Install it with "
            "`uv add langgraph-checkpoint-sqlite`."
        )
        return MemorySaver()


_checkpointer = _build_checkpointer()
interactive_workflow = _build_interactive_graph(_checkpointer)


def start_interactive_session(test_mode=False):
    """
    Start a new interactive workflow run. Returns (thread_id, result) where
    `result` is either a dict containing `__interrupt__` (if the graph is
    waiting for user input) or the final workflow state dict (if it somehow
    completed without needing any input, which shouldn't normally happen).
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    init_state = RecipeState(test_mode=test_mode)
    result = interactive_workflow.invoke(init_state, config=config)
    return thread_id, result


def resume_interactive_session(thread_id, user_input):
    """Resume a paused interactive workflow run with the given user input."""
    config = {"configurable": {"thread_id": thread_id}}
    return interactive_workflow.invoke(Command(resume=user_input), config=config)


def get_interrupt_prompt(result):
    """Extract the human-readable prompt payload from an interrupted result."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def run_workflow(test_mode=False):
    """CLI entry point: drives `interactive_workflow` end-to-end, prompting
    the user via stdin whenever the graph pauses for input."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    thread_id, result = start_interactive_session(test_mode=test_mode)
    while "__interrupt__" in result:
        payload = get_interrupt_prompt(result)
        prompt_text = payload.get("prompt", "Input required") if isinstance(payload, dict) else str(payload)
        recipes = payload.get("recipes") if isinstance(payload, dict) else None
        if recipes:
            # `recipes` entries are rich dicts (see user_selection_interrupt_node)
            for entry in recipes:
                print(f"{entry['number']}. {entry['name']} — {entry.get('description', '')}")
        user_input = input(f"\n{prompt_text}\n> ")
        result = resume_interactive_session(thread_id, user_input)

    state = RecipeState(**result)
    print("\nFinal Recipe:")
    print(state.final_recipe)
    if state.sentiment_escalation is not None:
        print(f"\nDetected Sentiment: {state.sentiment_escalation.sentiment}")
        print(f"Escalation Required: {state.sentiment_escalation.escalation_required}")
    if state.human_escalation is not None:
        print("\n--- HUMAN ESCALATION ---")
        print(state.human_escalation.human_notes)
    return state
