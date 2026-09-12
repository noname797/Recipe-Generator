"""
Lightweight persistence layer for escalated sessions, backed by SQLite (via
SQLAlchemy Core). This gives a human reviewer somewhere to actually see and
act on escalated cases, instead of everything being ephemeral per Flask
session.

Usage:
    from utils.escalation_store import save_escalation, list_escalations, get_escalation, resolve_escalation

The database file location is controlled by the `ESCALATION_DB_PATH` env var
(defaults to `./escalations.db`).
"""
import datetime
import os
import uuid
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )
    user_id = Column(String, nullable=True)
    sentiment = Column(String, nullable=True)
    human_notes = Column(Text, nullable=True)
    state_snapshot = Column(Text, nullable=False)  # JSON-serialized RecipeState
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(Text, nullable=True)


_DB_PATH = os.environ.get("ESCALATION_DB_PATH", "./escalations.db")
_engine = create_engine(f"sqlite:///{_DB_PATH}", future=True)
_SessionLocal = sessionmaker(bind=_engine, future=True)
Base.metadata.create_all(_engine)


def save_escalation(
    state_json: str,
    human_notes: Optional[str] = None,
    sentiment: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Persist a new escalation record and return its id."""
    with _SessionLocal() as db:
        record = Escalation(
            user_id=user_id,
            sentiment=sentiment,
            human_notes=human_notes,
            state_snapshot=state_json,
        )
        db.add(record)
        db.commit()
        return record.id


def list_escalations(include_resolved: bool = False):
    """Return escalation records, most recent first."""
    with _SessionLocal() as db:
        stmt = select(Escalation).order_by(Escalation.created_at.desc())
        if not include_resolved:
            stmt = stmt.where(Escalation.resolved.is_(False))
        return db.execute(stmt).scalars().all()


def get_escalation(escalation_id: str) -> Optional[Escalation]:
    with _SessionLocal() as db:
        return db.get(Escalation, escalation_id)


def resolve_escalation(escalation_id: str, resolution_notes: Optional[str] = None) -> bool:
    """Mark an escalation as resolved. Returns True if a record was updated."""
    with _SessionLocal() as db:
        stmt = (
            update(Escalation)
            .where(Escalation.id == escalation_id)
            .values(resolved=True, resolution_notes=resolution_notes)
        )
        result = db.execute(stmt)
        db.commit()
        return result.rowcount > 0
