import json

import httpx
import chainlit as cl

API_STREAM = "http://localhost:8000/chat/stream"

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
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            API_STREAM,
            json={"message": message.content, "session_id": session_id},
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "token":
                    await answer_msg.stream_token(event["content"])
                elif event["type"] == "meta":
                    meta = event

    if not meta:
        return

    if meta.get("human_handoff"):
        why = _REASON_COPY.get(meta.get("reason"), meta.get("reason") or "")
        await cl.Message(
            content=f"**Escalating to a human agent** — {why}."
        ).send()

