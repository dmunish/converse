from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, Session, create_engine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str
    created_at: datetime = Field(default_factory=_utcnow)


class Escalation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    # low_confidence | no_docs | not_answerable | ungrounded
    # | user_requested | out_of_scope | generation_error
    reason: str
    confidence: float
    created_at: datetime = Field(default_factory=_utcnow)


class QueryLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    intent: str = Field(default="kb_question", index=True)
    query: str
    answer: str
    confidence: float
    sources: str
    escalated: bool
    latency_ms: int
    created_at: datetime = Field(default_factory=_utcnow)


engine = create_engine(
    "sqlite:///./converse.db",
    connect_args={"check_same_thread": False},
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session