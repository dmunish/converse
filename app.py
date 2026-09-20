from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Depends
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from chainlit.utils import mount_chainlit

from db import create_db_and_tables, get_session, ChatMessage, Escalation, QueryLog
from rag import answer, stream_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="Converse: LearnForge Support Assistant", lifespan=lifespan)


# --- Schemas ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class Source(BaseModel):
    content: str
    source: str
    score: float | None = None
    last_reviewed: str | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[Source]
    human_handoff: bool
    reason: str | None = None


# --- Root redirect ---

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# --- Chat (non-streaming) ---

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return answer(req.message, req.session_id)


# --- Chat (streaming via SSE) ---

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    def event_stream():
        for event in stream_answer(req.message, req.session_id):
            if event["type"] == "meta":
                event = {k: v for k, v in event.items() if k != "sources"}
            yield f"data: {json.dumps(event)}\n\n"


# --- Reviewer / ops endpoints ---

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, db: Session = Depends(get_session)):
    return db.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).all()


@app.get("/escalations")
def list_escalations(db: Session = Depends(get_session)):
    return db.exec(
        select(Escalation).order_by(Escalation.created_at.desc())
    ).all()


@app.get("/evals/summary")
def eval_summary(db: Session = Depends(get_session)):
    logs = db.exec(select(QueryLog)).all()
    if not logs:
        return {"count": 0}
    latencies = sorted(l.latency_ms for l in logs)
    return {
        "count": len(logs),
        "escalation_rate": round(sum(1 for l in logs if l.escalated) / len(logs), 3),
        "avg_confidence": round(sum(l.confidence for l in logs) / len(logs), 3),
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
    }


mount_chainlit(app=app, target="ui.py", path="/chat-ui")