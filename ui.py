import json
import httpx
import chainlit as cl

API_STREAM = "http://localhost:8000/chat/stream"


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

    if meta["human_handoff"]:
        await cl.Message(content="⚠️ **Escalated to human agent**").send()

    if meta["sources"]:
        lines = []
        for s in meta["sources"]:
            badge = " ⚠️ stale" if s.get("stale") else ""
            lines.append(f"- `{s['source']}` (score: {s['score']:.2f}){badge}")
        await cl.Message(content="**Sources:**\n" + "\n".join(lines)).send()