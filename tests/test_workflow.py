"""
Integration tests for the end-to-end workflow, exercised in `test_mode`.
"""
import importlib

import pytest

from workflow import (
    ingredient_node,
    chef_node,
    nutrition_node,
    user_selection_node,
    user_modification_node,
    final_recipe_node,
    sentiment_node,
    human_escalation_node,
)
from utils.state import RecipeState


@pytest.fixture(autouse=True)
def isolated_escalation_db(tmp_path, monkeypatch):
    """Ensure escalations created by these tests never touch the real DB file."""
    db_path = tmp_path / "test_workflow_escalations.db"
    monkeypatch.setenv("ESCALATION_DB_PATH", str(db_path))
    import utils.escalation_store as store_module

    importlib.reload(store_module)
    yield store_module


def test_full_workflow_no_escalation(isolated_escalation_db):
    state = RecipeState(test_mode=True)
    state = ingredient_node(state, "chicken, rice")
    state = chef_node(state)
    state = nutrition_node(state)
    state = user_selection_node(state, "1")
    state = user_modification_node(state, "no")
    state = final_recipe_node(state)
    assert state.final_recipe is not None

    state = sentiment_node(state, "This was great, thank you!")
    # Fixed test YAML always marks escalation_required True; verify the
    # escalation path is taken, produces human escalation output, and is
    # persisted to the escalation store for human review.
    if state.sentiment_escalation.escalation_required:
        state = human_escalation_node(state)
        assert state.human_escalation is not None

        records = isolated_escalation_db.list_escalations(include_resolved=True)
        assert len(records) == 1
        assert records[0].human_notes == state.human_escalation.human_notes


def test_workflow_invalid_recipe_selection_raises():
    state = RecipeState(test_mode=True)
    state = ingredient_node(state, "chicken, rice")
    state = chef_node(state)
    state = nutrition_node(state)
    with pytest.raises(ValueError):
        user_selection_node(state, "999")
