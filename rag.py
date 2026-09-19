import os, json, time
from dotenv import load_dotenv
from sqlmodel import Session, select
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from lancedb.rerankers import RRFReranker

from db import ChatMessage, Escalation, QueryLog, engine

load_dotenv()

# --- Configure models ---
Settings.llm = GoogleGenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

# --- Load index (streaming=True enables token-by-token generation) ---
vector_store = LanceDBVectorStore(
    uri=os.getenv("LANCEDB_URI", "./lancedb"),
    table_name=os.getenv("LANCEDB_TABLE", "edtech_kb"),
    query_type="hybrid",
)
vector_store._add_reranker(RRFReranker())
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=4,
    vector_store_kwargs={"query_type": "hybrid"},
)

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
STALE_MARKERS = ["outdated", "older version", "no longer", "obsolete", "retired"]

# --- Helpers ---

def _load_history(session_id: str, limit: int = 6) -> str:
    """Return recent conversation as a formatted string (excludes current message)."""
    with Session(engine) as db:
        msgs = db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).all()
    msgs.reverse()
    return "\n".join(f"{m.role}: {m.content}" for m in msgs)

def _condense_query(message: str, history: str) -> str:
    """Rewrite a follow-up into a standalone question using prior turns."""
    if not history.strip():
        return message
    prompt = (
        "Given the conversation history below, rewrite the user's latest message "
        "into a single standalone question that can be understood without the history. "
        "If the message is already standalone, return it unchanged. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"History:\n{history}\n\nLatest: {message}\n\nStandalone question:"
    )
    try:
        return str(Settings.llm.complete(prompt)).strip()
    except Exception:
        return message  # graceful fallback

def _is_stale(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in STALE_MARKERS)

def _log_query(session_id, query, answer, confidence, sources, escalated, latency_ms):
    with Session(engine) as db:
        db.add(QueryLog(
            session_id=session_id, query=query, answer=answer,
            confidence=confidence, sources=json.dumps(sources),
            escalated=escalated, latency_ms=latency_ms,
        ))
        db.commit()

def _record_escalation(session_id, reason, confidence):
    with Session(engine) as db:
        db.add(Escalation(session_id=session_id, reason=reason, confidence=confidence))
        db.commit()

# --- Streaming generator ---

def stream_answer(message: str, session_id: str = "default"):
    """Yield {'type': 'token'|'meta', ...} events. This is the single source of truth."""
    start = time.time()

    # 1. Load prior history BEFORE persisting the current turn
    history = _load_history(session_id, limit=6)

    # 2. Persist user message
    with Session(engine) as db:
        db.add(ChatMessage(session_id=session_id, role="user", content=message))
        db.commit()

    # 3. Rewrite follow-ups into standalone queries
    standalone = _condense_query(message, history)

    # 4. Stream tokens from the LLM
    response = query_engine.query(standalone)
    tokens = []
    for token in response.response_gen:
        tokens.append(token)
        yield {"type": "token", "content": token}

    answer_text = "".join(tokens)

    # 5. Build sources + demote stale chunks
    chunks = []
    for n in response.source_nodes:
        score = n.score
        stale = _is_stale(n.text)
        if stale:
            score *= 0.7
        chunks.append({
            "content": n.text,
            "source": n.metadata.get("file_name", "unknown"),
            "score": score,
            "stale": stale,
        })
    max_score = max((c["score"] for c in chunks), default=0.0)

    # 6. Escalation gate
    reason = None
    if not chunks:
        reason = "no_docs"
    elif max_score < THRESHOLD:
        reason = "low_confidence"
    elif "don't have enough information" in answer_text.lower():
        reason = "llm_unsure"
    elif any(w in message.lower() for w in ["human", "agent", "representative"]):
        reason = "user_requested"

    escalate = reason is not None
    if escalate:
        _record_escalation(session_id, reason, max_score)

    # 7. Persist assistant message + query log
    with Session(engine) as db:
        db.add(ChatMessage(session_id=session_id, role="assistant", content=answer_text))
        db.commit()

    latency_ms = int((time.time() - start) * 1000)
    _log_query(session_id, message, answer_text, max_score, chunks, escalate, latency_ms)

    yield {
        "type": "meta",
        "confidence": round(max_score, 2),
        "sources": chunks,
        "human_handoff": escalate,
        "reason": reason,
    }

# --- Non-streaming wrapper (for API clients that want the full response) ---

def answer(message: str, session_id: str = "default") -> dict:
    events = list(stream_answer(message, session_id))
    text = "".join(e["content"] for e in events if e["type"] == "token")
    meta = next(e for e in events if e["type"] == "meta")
    return {
        "answer": text,
        "confidence": meta["confidence"],
        "sources": meta["sources"],
        "human_handoff": meta["human_handoff"],
    }