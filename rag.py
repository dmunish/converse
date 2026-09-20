import json
import os
import re
import time

from dotenv import load_dotenv
from sqlmodel import Session, select
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from lancedb.rerankers import RRFReranker

from db import ChatMessage as ChatMessageRow, Escalation, QueryLog, engine
from prompts import SYSTEM_PROMPT, GROUNDING_SYSTEM_PROMPT, ROUTER_PROMPT

load_dotenv()

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
Settings.llm = GoogleGenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

# --------------------------------------------------------------------------- #
# Index / retriever
# --------------------------------------------------------------------------- #
vector_store = LanceDBVectorStore(
    uri=os.getenv("LANCEDB_URI", "./lancedb"),
    table_name=os.getenv("LANCEDB_TABLE", "edtech_kb"),
    query_type="hybrid",
)
vector_store._add_reranker(RRFReranker())
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage",
)
index = load_index_from_storage(storage_context)

retriever = index.as_retriever(
    similarity_top_k=4,
    vector_store_kwargs={"query_type": "hybrid"},
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_HUMAN_REQUEST_PATTERNS = [
    re.compile(r"\b(human|real person|live agent|human agent|human representative)\b", re.I),
    re.compile(r"\b(talk|speak|chat)\s+to\s+(a\s+|an\s+)?(human|agent|representative|person|someone)\b", re.I),
    re.compile(r"\bconnect me (to|with)\b", re.I),
]


def _load_history(session_id: str, limit: int = HISTORY_LIMIT) -> list[ChatMessageRow]:
    with Session(engine) as db:
        rows = db.exec(
            select(ChatMessageRow)
            .where(ChatMessageRow.session_id == session_id)
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        ).all()
    rows.reverse()
    return rows


def _persist_message(session_id: str, role: str, content: str) -> None:
    with Session(engine) as db:
        db.add(ChatMessageRow(session_id=session_id, role=role, content=content))
        db.commit()


def _condense_query(message: str, history: list[ChatMessageRow]) -> str:
    """Rewrite a follow-up into a standalone question using prior turns."""
    if not history:
        return message
    hist_text = "\n".join(f"{m.role}: {m.content}" for m in history)
    prompt = (
        "Rewrite the user's latest message into a single standalone question "
        "that can be understood without the conversation history. If it is "
        "already standalone, return it unchanged. Return ONLY the rewritten "
        "question, nothing else.\n\n"
        f"History:\n{hist_text}\n\nLatest: {message}\n\nStandalone question:"
    )
    try:
        return str(Settings.llm.complete(prompt)).strip() or message
    except Exception:
        return message


def _retrieve(query: str) -> list[dict]:
    try:
        nodes = retriever.retrieve(query)
    except Exception:
        return []
    chunks: list[dict] = []
    for n in nodes:
        meta = n.metadata or {}
        chunks.append({
            "content": n.text,
            "source": meta.get("file_name", "unknown"),
            "last_reviewed": meta.get("last_reviewed"),
            "score": float(n.score) if n.score is not None else None,
        })
    return chunks


def _build_messages(
    query: str,
    history: list[ChatMessageRow],
    chunks: list[dict],
) -> list[ChatMessage]:
    """Assemble the chat payload.

    Order matters for prompt caching: the static system prompt is first, then
    the conversation history, then the per-query retrieved excerpts as
    separate system messages, then the current user turn.
    """
    msgs: list[ChatMessage] = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
    ]

    for m in history:
        role = MessageRole.USER if m.role == "user" else MessageRole.ASSISTANT
        msgs.append(ChatMessage(role=role, content=m.content))

    if chunks:
        for i, c in enumerate(chunks, start=1):
            header = f"[{i}] source={c['source']}"
            if c.get("last_reviewed"):
                header += f", last_reviewed={c['last_reviewed']}"
            msgs.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Retrieved LearnForge excerpt:\n{header}\n{c['content']}",
            ))
    else:
        msgs.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "No knowledge-base excerpts were retrieved for this question. "
                "You do not have enough information to answer. Politely say so "
                "and offer to escalate to a human agent."
            ),
        ))

    msgs.append(ChatMessage(role=MessageRole.USER, content=query))
    return msgs

def _build_messages_direct(
        message: str,
        history: list[ChatMessageRow],
    ) -> list[ChatMessage]:
        """Static system prompt + history + current user turn. No excerpts."""
        msgs: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]
        for m in history:
            role = MessageRole.USER if m.role == "user" else MessageRole.ASSISTANT
            msgs.append(ChatMessage(role=role, content=m.content))
        msgs.append(ChatMessage(role=MessageRole.USER, content=message))
        return msgs

def _route(message: str) -> str:
    try:
        raw = str(Settings.llm.chat([
            ChatMessage(role=MessageRole.SYSTEM, content=ROUTER_PROMPT),
            ChatMessage(role=MessageRole.USER, content=message),
        ]).message.content).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw).get("intent", "kb_question")
    except Exception:
        return "kb_question"  # safe default: fall through to the strict path

def _grade(query: str, chunks: list[dict], answer_text: str) -> dict:
    """Strict LLM-as-judge pass that catches hallucination / non-answer."""
    excerpts = "\n\n".join(
        f"[{i}] {c['content']}" for i, c in enumerate(chunks, start=1)
    ) or "(none)"
    user_block = (
        f"Question:\n{query}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        f"Draft answer:\n{answer_text}"
    )
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=GROUNDING_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_block),
    ]
    try:
        raw = str(Settings.llm.chat(msgs).message.content).strip()
    except Exception as e:
        return {
            "answerable": bool(chunks),
            "grounded": bool(chunks),
            "confidence": 0.5 if chunks else 0.0,
            "reason": f"grader_error: {e}",
        }

    raw = (
        raw.removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )
    try:
        data = json.loads(raw)
    except Exception:
        return {
            "answerable": bool(chunks),
            "grounded": bool(chunks),
            "confidence": 0.5 if chunks else 0.0,
            "reason": "grader_parse_error",
        }

    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "answerable": bool(data.get("answerable", False)),
        "grounded": bool(data.get("grounded", False)),
        "confidence": max(0.0, min(1.0, conf)),
        "reason": str(data.get("reason", "")).strip(),
    }


def _decide_escalation(
    user_message: str,
    chunks: list[dict],
    grade: dict,
) -> str | None:
    if any(p.search(user_message) for p in _HUMAN_REQUEST_PATTERNS):
        return "user_requested"
    if not chunks:
        return "no_docs"
    if not grade["answerable"]:
        return "not_answerable"
    if not grade["grounded"]:
        return "ungrounded"
    if grade["confidence"] < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return None


def _finalize(
    session_id: str,
    query: str,
    answer_text: str,
    chunks: list[dict],
    grade: dict,
    reason: str | None,
    start: float,
) -> None:
    _persist_message(session_id, "assistant", answer_text)
    if reason:
        with Session(engine) as db:
            db.add(Escalation(
                session_id=session_id,
                reason=reason,
                confidence=grade["confidence"],
            ))
            db.commit()
    latency_ms = int((time.time() - start) * 1000)
    with Session(engine) as db:
        db.add(QueryLog(
            session_id=session_id,
            query=query,
            answer=answer_text,
            confidence=grade["confidence"],
            sources=json.dumps(chunks),
            escalated=bool(reason),
            latency_ms=latency_ms,
        ))
        db.commit()


def _meta_event(grade: dict, chunks: list[dict], reason: str | None) -> dict:
    return {
        "type": "meta",
        "confidence": round(grade["confidence"], 2),
        "answerable": grade["answerable"],
        "grounded": grade["grounded"],
        "sources": chunks,
        "human_handoff": reason is not None,
        "reason": reason,
        "grader_reason": grade.get("reason", ""),
    }


# --------------------------------------------------------------------------- #
# Streaming generator — single source of truth
# --------------------------------------------------------------------------- #
def stream_answer(message: str, session_id: str = "default"):
    """Yield {'type': 'token'|'meta', ...} events."""
    start = time.time()

    # 1. Load prior history BEFORE persisting the current turn.
    history = _load_history(session_id)

    # 2. Persist the user turn.
    _persist_message(session_id, "user", message)

    # 3. Route the turn before doing any retrieval.
    intent = _route(message)

    # ---- Short-circuit branches: no retrieval, no grader -------------------
    if intent in {"smalltalk", "human_request", "out_of_scope"}:
        messages = _build_messages_direct(message, history)

        if intent == "human_request":
            # Skip the LLM entirely — hand off deterministically.
            answer_text = (
                "Of course — I'm connecting you with a LearnForge human agent. "
                "They'll be able to help with account-specific details."
            )
            yield {"type": "token", "content": answer_text}
            grade = {"answerable": False, "grounded": True, "confidence": 0.0,
                     "reason": "user requested human"}
            _finalize(session_id, message, answer_text, [], grade,
                      "user_requested", start)
            yield _meta_event(grade, [], "user_requested")
            return

        if intent == "out_of_scope":
            answer_text = (
                "I'm Converse, LearnForge's support assistant, so I can only help "
                "with LearnForge accounts, courses, purchases, and subscriptions. "
                "For anything else I can connect you with a human agent if you'd like."
            )
            yield {"type": "token", "content": answer_text}
            grade = {"answerable": False, "grounded": True, "confidence": 0.0,
                     "reason": "out of scope"}
            _finalize(session_id, message, answer_text, [], grade,
                      "out_of_scope", start)
            yield _meta_event(grade, [], "out_of_scope")
            return

        # intent == "smalltalk": let the LLM stay in character, no citation
        # or grading required.
        tokens: list[str] = []
        try:
            for delta in Settings.llm.stream_chat(messages):
                piece = delta.delta or ""
                if not piece:
                    continue
                tokens.append(piece)
                yield {"type": "token", "content": piece}
            answer_text = "".join(tokens).strip()
        except Exception as e:
            answer_text = (
                "Hi — I'm Converse, LearnForge's support assistant. "
                "How can I help with your account or courses today?"
            )
            yield {"type": "token", "content": answer_text}
            grade = {"answerable": True, "grounded": True, "confidence": 1.0,
                     "reason": f"smalltalk_fallback: {e}"}
            _finalize(session_id, message, answer_text, [], grade, None, start)
            yield _meta_event(grade, [], None)
            return

        grade = {"answerable": True, "grounded": True, "confidence": 1.0,
                 "reason": "smalltalk"}
        _finalize(session_id, message, answer_text, [], grade, None, start)
        yield _meta_event(grade, [], None)
        return

    # ---- kb_question: existing retrieve → generate → grade → gate path ----

    # 4. Rewrite follow-ups into standalone queries.
    standalone = _condense_query(message, history) if history else message

    # 5. Retrieve.
    chunks = _retrieve(standalone)

    # 6. Assemble chat payload and stream tokens.
    messages = _build_messages(standalone, history, chunks)
    tokens: list[str] = []
    try:
        for delta in Settings.llm.stream_chat(messages):
            piece = delta.delta or ""
            if not piece:
                continue
            tokens.append(piece)
            yield {"type": "token", "content": piece}
        answer_text = "".join(tokens).strip()
    except Exception as e:
        answer_text = (
            "Sorry — I ran into a temporary problem generating a response. "
            "I'm connecting you with a human agent."
        )
        yield {"type": "token", "content": answer_text}
        grade = {"answerable": False, "grounded": False, "confidence": 0.0,
                 "reason": f"generation_error: {e}"}
        _finalize(session_id, message, answer_text, chunks, grade,
                  "generation_error", start)
        yield _meta_event(grade, chunks, "generation_error")
        return

    # 7. Grade the answer against the retrieved excerpts.
    if not chunks:
        grade = {"answerable": False, "grounded": False, "confidence": 0.0,
                 "reason": "no excerpts retrieved"}
    elif not answer_text:
        grade = {"answerable": False, "grounded": False, "confidence": 0.0,
                 "reason": "empty answer"}
    else:
        grade = _grade(standalone, chunks, answer_text)

    # 8. Escalation gate.
    reason = _decide_escalation(message, chunks, grade)

    # 9. Persist + log.
    _finalize(session_id, message, answer_text, chunks, grade, reason, start)

    yield _meta_event(grade, chunks, reason)

# --------------------------------------------------------------------------- #
# Non-streaming wrapper
# --------------------------------------------------------------------------- #
def answer(message: str, session_id: str = "default") -> dict:
    events = list(stream_answer(message, session_id))
    text = "".join(e["content"] for e in events if e["type"] == "token")
    meta = next(e for e in events if e["type"] == "meta")
    return {
        "answer": text,
        "confidence": meta["confidence"],
        "sources": meta["sources"],
        "human_handoff": meta["human_handoff"],
        "reason": meta["reason"],
    }