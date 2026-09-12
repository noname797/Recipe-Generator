from flask import Flask, render_template, request, jsonify, session

from workflow import (
    ingredient_node,
    chef_node,
    nutrition_node,
    user_selection_node,
    user_modification_node,
    final_recipe_node,
    sentiment_node,
    human_escalation_node
)
from utils.state import RecipeState

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-key'

def get_user_state():
    user_id = session.get('user_id')
    if not user_id:
        import uuid
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
    if 'user_states' not in session:
        session['user_states'] = {}
    user_states = session['user_states']
    if user_id not in user_states:
        user_states[user_id] = RecipeState().model_dump()
    return RecipeState(**user_states[user_id])

def save_user_state(state):
    user_id = session.get('user_id')
    if not user_id:
        return
    if 'user_states' not in session:
        session['user_states'] = {}
    user_states = session['user_states']
    user_states[user_id] = state.model_dump()
    session['user_states'] = user_states

@app.route('/')
def index():
    session["step"] = None
    return render_template('index.html')


# Step order for the workflow
WORKFLOW_STEPS = [
    'ingredient',
    'chef',
    'nutrition',
    'user_selection',
    'user_modification',
    'final_recipe',
    'sentiment',
    'human_escalation',
]

def get_next_step(state, current_step):
    idx = WORKFLOW_STEPS.index(current_step)
    # Sentiment escalation logic
    if current_step == 'sentiment' and hasattr(state, 'sentiment_escalation'):
        if getattr(state.sentiment_escalation, 'escalation_required', False):
            return 'human_escalation'
        else:
            return None
    if idx + 1 < len(WORKFLOW_STEPS):
        return WORKFLOW_STEPS[idx + 1]
    return None

@app.route('/chat', methods=['POST'])

def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    state = get_user_state()
    step = session.get('step')
    
    if not step:
        step = 'ingredient'

    message = ""
    recipe_card = None

    while True:
        if step == 'ingredient':
            state = ingredient_node(state, user_input)
            message = "Ingredients received! Generating recipes..."
            step = 'chef'
            user_input = ''  # Clear user input for next step
        elif step == 'chef':
            state = chef_node(state)
            step = 'nutrition'
        elif step == 'nutrition':
            state = nutrition_node(state)
            message = "Here are some recipe suggestions. Please select a recipe by number."
            recipe_list = "<ul>"
            for idx, recipe in enumerate(state.chef_output.recipes, 1):
                recipe_list += f"<li><b>{idx}.</b> <b>{recipe.name}</b><br>"
                recipe_list += f"Description: {recipe.description}<br>"
                recipe_list += f"Ingredients: {', '.join(recipe.ingredients)}<br>"
                recipe_list += "Instructions:<ol>"
                for step_idx, instruction in enumerate(recipe.instructions, 1):
                    recipe_list += f"<li> {step_idx}. {instruction}</li>"
                recipe_list += "</ol>"
                recipe_list += f"Prep Time: {getattr(recipe, 'prep_time', 'N/A')}<br>"
                recipe_list += f"Cook Time: {getattr(recipe, 'cook_time', 'N/A')}<br>"
                recipe_list += f"Servings: {getattr(recipe, 'servings', 'N/A')}<br>"
                if getattr(recipe, 'tags', None):
                    recipe_list += f"Tags: {', '.join(recipe.tags)}<br>"
                if state.nutrition_info:
                    recipe_list += f"Nutrition: {state.nutrition_info.recipes[idx-1].nutrition_profile}<br>"
                    recipe_list += f"Suggested Addons: {state.nutrition_info.recipes[idx-1].suggested_addons}<br>"
                    recipe_list += f"Insights: {state.nutrition_info.recipes[idx-1].nutrition_insights}<br>"
                recipe_list += "</li>"
            recipe_list += "</ul>"

            message += recipe_list
            step = 'user_selection'
            break  # Wait for user modification input

        elif step == 'user_selection':
            try:
                choice = int(user_input)
                if 1 <= choice <= len(state.chef_output.recipes):
                    state.selected_recipe = state.chef_output.recipes[choice - 1]
                    state.selected_nutrition_info = state.nutrition_info.recipes[choice - 1] if state.nutrition_info else None
                    message = f"You selected: {state.selected_recipe.name}. Would you like to modify the recipe? (e.g., reduce salt, swap an ingredient, adjust nutrition) If yes, type your modification. If not, type 'no'."
                    step = 'user_modification'
                    break  # Wait for user modification input
                else:
                    message = f"Please enter a number between 1 and {len(state.chef_output.recipes)}."
                    break
            except Exception:
                message = "Invalid input. Please enter a valid number."
                break
        elif step == 'user_modification':
            if user_input.strip().lower() == 'no':
                message = "Finalizing recipe..."
                step = 'final_recipe'
                user_input = ''
            else:
                state = user_modification_node(state, user_input)
                message = "Modification applied. Finalizing recipe..."
                step = 'final_recipe'
                user_input = ''
        elif step == 'final_recipe':
            state = final_recipe_node(state)
            recipe = state.final_recipe
            recipe_card = f"<b>{recipe.name}</b><br>Description: {recipe.description}<br>Ingredients: {', '.join(recipe.ingredients)}<br>Instructions: {' '.join(recipe.instructions)}"
            message = "Here is your final recipe! How do you feel about your experience? (Optional feedback for sentiment analysis)"
            step = 'sentiment'
            break  # Wait for user feedback
        elif step == 'sentiment':
            state = sentiment_node(state, user_input)
            if hasattr(state, 'sentiment_escalation') and getattr(state.sentiment_escalation, 'escalation_required', False):
                message = "It seems you need human assistance. Escalating..."
                step = 'human_escalation'
                user_input = ''
            else:
                message = "Thank you for your feedback! Enjoy your meal!"
                step = 'ingredient'  # Restart for new session
                break
        elif step == 'human_escalation':
            state = human_escalation_node(state)
            message = "A human will review your request. Thank you!"
            step = 'ingredient'  # Restart for new session
            break
        else:
            message = "Sorry, something went wrong."
            break

    save_user_state(state)
    session['step'] = step
    return jsonify({'message': message, 'recipe_card': recipe_card})

if __name__ == '__main__':
    app.run(debug=True)
