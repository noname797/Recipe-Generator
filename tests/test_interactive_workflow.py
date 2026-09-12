"""
Integration tests for the interrupt-driven `interactive_workflow`, which is
the single LangGraph definition shared by both the CLI (`run_workflow`) and
the Flask web app (`app.py`). These exercise the same start/resume driver
functions app.py uses, all in `test_mode` so no AWS credentials are needed.
"""
import importlib

import pytest

from workflow import (
    start_interactive_session,
    resume_interactive_session,
    get_interrupt_prompt,
)


@pytest.fixture(autouse=True)
def isolated_escalation_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_interactive_escalations.db"
    monkeypatch.setenv("ESCALATION_DB_PATH", str(db_path))
    import utils.escalation_store as store_module

    importlib.reload(store_module)
    yield store_module


def test_interactive_session_pauses_for_ingredient_input():
    thread_id, result = start_interactive_session(test_mode=True)
    assert "__interrupt__" in result
    payload = get_interrupt_prompt(result)
    assert "ingredients" in payload["prompt"].lower()


def test_interactive_session_full_happy_path(isolated_escalation_db):
    thread_id, result = start_interactive_session(test_mode=True)
    assert "__interrupt__" in result

    # Resume with ingredients -> should run chef+nutrition and pause at
    # recipe selection.
    result = resume_interactive_session(thread_id, "chicken, rice")
    assert "__interrupt__" in result
    payload = get_interrupt_prompt(result)
    assert "select a recipe" in payload["prompt"].lower()
    assert len(payload["recipes"]) == 2
    assert payload["recipes"][0]["name"] == "Test Recipe 1"

    # Resume with recipe choice -> pause at modification prompt.
    result = resume_interactive_session(thread_id, "1")
    assert "__interrupt__" in result
    payload = get_interrupt_prompt(result)
    assert "modify" in payload["prompt"].lower()

    # Decline modification -> pause at sentiment/feedback prompt.
    result = resume_interactive_session(thread_id, "no")
    assert "__interrupt__" in result
    payload = get_interrupt_prompt(result)
    assert "experience" in payload["prompt"].lower()

    # Give feedback -> fixed test YAML always escalates, so the graph
    # should now complete (human_escalation runs automatically, no interrupt).
    result = resume_interactive_session(thread_id, "This was great!")
    assert "__interrupt__" not in result
    assert result["final_recipe"].name == "Test Final Recipe"
    assert result["sentiment_escalation"].escalation_required is True
    assert result["human_escalation"] is not None

    # The escalation should have been persisted for human review.
    records = isolated_escalation_db.list_escalations(include_resolved=True)
    assert len(records) == 1


def test_interactive_session_invalid_recipe_choice_raises():
    thread_id, result = start_interactive_session(test_mode=True)
    result = resume_interactive_session(thread_id, "chicken, rice")
    assert "__interrupt__" in result

    with pytest.raises(ValueError):
        resume_interactive_session(thread_id, "999")
