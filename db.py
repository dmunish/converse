from datetime import datetime
from sqlmodel import SQLModel, Field, Session, create_engine
from typing import Optional

# --- Tables ---

class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str                    # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Escalation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    reason: str                  # "low_confidence" | "no_docs" | "user_requested"
    confidence: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

class QueryLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    query: str
    answer: str
    confidence: float
    sources: str                 # JSON-serialized source list
    escalated: bool
    latency_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Engine + session dependency ---

engine = create_engine(
    "sqlite:///./converse.db",
    connect_args={"check_same_thread": False},  # SQLite + FastAPI
)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session