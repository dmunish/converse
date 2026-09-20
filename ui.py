import json
import os

import httpx
import chainlit as cl

API_BASE = os.getenv("CONVERSE_API", "http://localhost:8000")
API_STREAM = f"{API_BASE}/chat/stream"

_REASON_COPY = {
    "user_requested": "you asked for a human",
    "no_docs": "no relevant knowledge-base excerpts",
    "not_answerable": "the knowledge base doesn't cover this",
    "ungrounded": "the draft answer couldn't be verified against sources",
    "low_confidence": "low confidence",
    "generation_error": "a temporary generation error",
    "out_of_scope": "the question is outside LearnForge support",
}


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session_id", cl.context.session.id)


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    answer_msg = cl.Message(content="")
    await answer_msg.send()

    meta = None
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                API_STREAM,
                json={"message": message.content, "session_id": session_id},
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    await answer_msg.stream_token(
                        "Sorry — the support backend returned "
                        f"HTTP {response.status_code}. Please try again."
                    )
                    await answer_msg.update()
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        await answer_msg.stream_token(event["content"])
                    elif event["type"] == "meta":
                        meta = event
    except httpx.HTTPError as e:
        await answer_msg.stream_token(
            "Sorry — I couldn't reach the support backend. "
            f"({e.__class__.__name__})"
        )
        await answer_msg.update()
        return

    if not meta:
        return

    if meta.get("human_handoff"):
        why = _REASON_COPY.get(meta.get("reason"), meta.get("reason") or "")
        await cl.Message(
            content=f"**Escalating to a human agent** — {why}."
        ).send()