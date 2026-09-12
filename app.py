import logging
import os
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

from workflow import start_interactive_session, resume_interactive_session, get_interrupt_prompt
from utils.state import RecipeState
from utils.escalation_store import list_escalations, get_escalation, resolve_escalation

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set. Add it to your .env file, e.g.\n"
        "FLASK_SECRET_KEY=$(python -c \"import secrets; print(secrets.token_hex(32))\")"
    )

# Store session data server-side (filesystem by default) instead of inside the
# client-side cookie. The cookie itself only carries an opaque session id, so
# we avoid the 4KB cookie-size limit and stop leaking recipe/user data to the
# client. Set SESSION_TYPE=redis (with SESSION_REDIS configured) in production
# for a more robust, horizontally-scalable backend.
app.config["SESSION_TYPE"] = os.environ.get("SESSION_TYPE", "filesystem")
app.config["SESSION_FILE_DIR"] = os.environ.get("SESSION_FILE_DIR", "./.flask_session")
app.config["SESSION_PERMANENT"] = False
Session(app)

def get_thread_id():
    """Get (or create) the LangGraph checkpoint thread id for this browser session.

    Kept for any external callers, but `/chat` now manages `thread_id`
    directly to avoid implicitly creating a thread id before a graph run has
    actually been started for it.
    """
    thread_id = session.get('thread_id')
    if not thread_id:
        import uuid
        thread_id = str(uuid.uuid4())
        session['thread_id'] = thread_id
    return thread_id

@app.route('/')
def index():
    # Note: we deliberately do NOT clear the session here. Doing so used to
    # wipe an in-progress conversation on every page refresh, which is a
    # jarring UX regression (users lose all progress just from reloading the
    # tab). The frontend persists a lightweight transcript in
    # sessionStorage and, combined with the still-alive server-side thread,
    # can seamlessly restore the chat. Use the explicit `/reset` endpoint
    # (wired to the "Start Over" button) to intentionally clear state.
    return render_template('index.html')


@app.route('/reset', methods=['POST'])
def reset():
    """Explicitly clear the current conversation. Called by the "Start Over"
    button so a page refresh alone never destroys progress, but the user can
    still deliberately start fresh."""
    session.pop('thread_id', None)
    session['started'] = False
    return jsonify({'status': 'reset'})



def _message_for_interrupt(payload):
    """Turn an interrupt() payload dict into (message, recipes) for the
    client. Rather than pre-rendering an HTML blob server-side, we hand back
    the plain prompt text and the raw recipe data; the frontend renders the
    recipes as proper cards using safe DOM APIs (textContent), which also
    sidesteps any HTML-escaping edge cases entirely.
    """
    if not isinstance(payload, dict):
        return str(payload), None
    message = payload.get("prompt", "Input required")
    recipes = payload.get("recipes")
    return message, recipes


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'message': 'Invalid request body; expected JSON object.'}), 400

    user_input = data.get('message', '')
    if not isinstance(user_input, str):
        return jsonify({'message': 'Field "message" must be a string.'}), 400
    user_input = user_input.strip()
    if len(user_input) > 4000:
        return jsonify({'message': 'Message is too long (max 4000 characters).'}), 400

    thread_id = session.get('thread_id')
    started = session.get('started', False)

    message = ""
    recipe_card = None
    recipes = None
    final_recipe = None

    try:
        if not started:
            # First message of a new conversation: explicitly start a fresh
            # graph run (rather than starting-and-resuming in one step) so a
            # failure here can't leave `started` unset while a `thread_id`
            # has already been assigned/persisted.
            thread_id, result = start_interactive_session()
            session['thread_id'] = thread_id
            session['started'] = True
        else:
            if not thread_id:
                # Defensive: `started` was set but the thread id is missing
                # (e.g. session storage was cleared out-of-band). Restart
                # cleanly rather than calling resume with no thread_id.
                thread_id, result = start_interactive_session()
                session['thread_id'] = thread_id
            else:
                result = resume_interactive_session(thread_id, user_input)

        if "__interrupt__" in result:
            payload = get_interrupt_prompt(result)
            message, recipes = _message_for_interrupt(payload)
        else:
            # Graph reached END: final recipe (and possibly escalation) done.
            state = RecipeState(**result)
            recipe = state.final_recipe
            final_recipe = {
                "name": recipe.name,
                "description": recipe.description,
                "ingredients": recipe.ingredients,
                "instructions": recipe.instructions,
                "prep_time": getattr(recipe, "prep_time", None),
                "cook_time": getattr(recipe, "cook_time", None),
                "servings": getattr(recipe, "servings", None),
                "tags": getattr(recipe, "tags", None),
            }
            # Kept for older clients: a plain-text fallback rendering of the
            # final recipe (no HTML). Modern clients should prefer the
            # structured `final_recipe` field above.
            recipe_card = (
                f"{recipe.name}\nDescription: {recipe.description}\n"
                f"Ingredients: {', '.join(recipe.ingredients)}\n"
                f"Instructions: {' '.join(recipe.instructions)}"
            )
            if getattr(state, "sentiment_escalation", None) and state.sentiment_escalation.escalation_required:
                message = "It seems you need human assistance. A human will review your request shortly. Thank you!"
            else:
                message = "Here is your final recipe! Thank you for your feedback. Enjoy your meal!"
            # Reset for a new conversation on the next message.
            session.pop('thread_id', None)
            session['started'] = False
    except ValueError as e:
        # Expected, user-facing validation errors (e.g. bad recipe number).
        # The graph node already raised before mutating state, so the
        # in-progress checkpoint is unaffected; the same interrupt will be
        # re-presented on the next call once the user retries.
        message = str(e)
    except Exception:
        logger.exception("Unhandled error while processing chat message for thread %s", thread_id)
        message = (
            "Sorry, something went wrong on our end while processing that. "
            "Please try again."
        )

    return jsonify({
        'message': str(message),
        'recipe_card': recipe_card,
        'recipes': recipes,
        'final_recipe': final_recipe,
    })



# --- Human reviewer endpoints -------------------------------------------------
# Simple JSON API for a human operator to see and act on escalated sessions.
# Protected by a static API key (set via the `ESCALATION_API_KEY` env var),
# expected in the `X-API-Key` request header. This is intentionally minimal;
# swap in a proper auth scheme (OAuth/JWT/etc.) before exposing these routes
# beyond a trusted internal network.

def require_escalation_api_key(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        expected_key = os.environ.get("ESCALATION_API_KEY")
        if not expected_key:
            logger.error(
                "ESCALATION_API_KEY is not configured; refusing escalation "
                "API request. Set ESCALATION_API_KEY in your .env to enable "
                "this endpoint."
            )
            return jsonify({'error': 'Escalation API is not configured'}), 503
        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key or provided_key != expected_key:
            return jsonify({'error': 'Unauthorized'}), 401
        return view_func(*args, **kwargs)

    return wrapped


@app.route('/escalations', methods=['GET'])
@require_escalation_api_key
def get_escalations():
    include_resolved = request.args.get('include_resolved', 'false').lower() == 'true'
    records = list_escalations(include_resolved=include_resolved)
    return jsonify([
        {
            'id': r.id,
            'created_at': r.created_at.isoformat(),
            'sentiment': r.sentiment,
            'human_notes': r.human_notes,
            'resolved': r.resolved,
            'resolution_notes': r.resolution_notes,
        }
        for r in records
    ])


@app.route('/escalations/<escalation_id>', methods=['GET'])
@require_escalation_api_key
def get_escalation_detail(escalation_id):
    record = get_escalation(escalation_id)
    if record is None:
        return jsonify({'error': 'Escalation not found'}), 404
    return jsonify({
        'id': record.id,
        'created_at': record.created_at.isoformat(),
        'sentiment': record.sentiment,
        'human_notes': record.human_notes,
        'resolved': record.resolved,
        'resolution_notes': record.resolution_notes,
        'state_snapshot': record.state_snapshot,
    })


@app.route('/escalations/<escalation_id>/resolve', methods=['POST'])
@require_escalation_api_key
def resolve_escalation_route(escalation_id):
    data = request.get_json(silent=True) or {}
    notes = data.get('resolution_notes')
    updated = resolve_escalation(escalation_id, resolution_notes=notes)
    if not updated:
        return jsonify({'error': 'Escalation not found'}), 404
    return jsonify({'status': 'resolved', 'id': escalation_id})


if __name__ == '__main__':
    app.run(debug=True)
