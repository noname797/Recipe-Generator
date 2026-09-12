"""
Tests for utils.escalation_store, using a temporary SQLite DB file per test
run so we never touch a developer's real escalations.db.
"""
import importlib
import os

import pytest


@pytest.fixture
def escalation_store(tmp_path, monkeypatch):
    """Reload utils.escalation_store with an isolated, temporary DB file."""
    db_path = tmp_path / "test_escalations.db"
    monkeypatch.setenv("ESCALATION_DB_PATH", str(db_path))

    import utils.escalation_store as store_module

    importlib.reload(store_module)
    return store_module


def test_save_and_list_escalation(escalation_store):
    escalation_id = escalation_store.save_escalation(
        state_json='{"foo": "bar"}',
        human_notes="please review",
        sentiment="frustrated",
    )
    assert escalation_id

    records = escalation_store.list_escalations()
    assert len(records) == 1
    assert records[0].id == escalation_id
    assert records[0].human_notes == "please review"
    assert records[0].resolved is False


def test_get_escalation_by_id(escalation_store):
    escalation_id = escalation_store.save_escalation(state_json="{}")
    record = escalation_store.get_escalation(escalation_id)
    assert record is not None
    assert record.id == escalation_id


def test_resolve_escalation(escalation_store):
    escalation_id = escalation_store.save_escalation(state_json="{}")
    updated = escalation_store.resolve_escalation(escalation_id, resolution_notes="handled")
    assert updated is True

    record = escalation_store.get_escalation(escalation_id)
    assert record.resolved is True
    assert record.resolution_notes == "handled"

    # Resolved records are excluded from the default (unresolved-only) listing.
    assert escalation_store.list_escalations(include_resolved=False) == []
    assert len(escalation_store.list_escalations(include_resolved=True)) == 1


def test_resolve_nonexistent_escalation_returns_false(escalation_store):
    assert escalation_store.resolve_escalation("does-not-exist") is False
