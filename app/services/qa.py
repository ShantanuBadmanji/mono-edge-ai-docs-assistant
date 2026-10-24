import asyncio
import json
import secrets
from typing import Literal

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Ticket

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly from the provided document excerpts. "
    "If the answer cannot be found in the excerpts, respond with exactly: "
    "'The answer is not available in the provided documents.' "
    "Do not use any prior knowledge beyond the excerpts provided."
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket in the system when the user explicitly asks to raise or create one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A brief, descriptive title for the ticket"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Urgency level of the ticket",
                    },
                },
                "required": ["title", "priority"],
            },
        },
    }
]


async def answer_question(
    question: str,
    model,
    collection,
    db: AsyncSession,
) -> dict:
    q_embedding = await asyncio.to_thread(lambda: model.encode([question])[0].tolist())

    results = await asyncio.to_thread(
        lambda: collection.query(
            query_embeddings=[q_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Source: {m['file_name']}, page {m['page']}]\n{doc}"
        for doc, m in zip(docs, metas)
    )

    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n{context}\n\nQuestion: {question}"},
    ]

    response = await groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        tools=_TOOLS,
        tool_choice="auto",
        temperature=0,
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_result = await _run_create_ticket(tool_call, db)

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        final = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0,
        )
        answer = final.choices[0].message.content
    else:
        answer = message.content

    seen: set[tuple] = set()
    sources = []
    for m in metas:
        key = (m["file_name"], m["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file_name": m["file_name"], "page": m["page"]})

    return {"answer": answer, "sources": sources}


async def _run_create_ticket(tool_call, db: AsyncSession) -> str:
    class CreateTicketInput(BaseModel):
        title: str
        priority: Literal["low", "medium", "high"]

    if tool_call.function.name != "create_ticket":
        return "Error: unknown tool requested"

    try:
        args = json.loads(tool_call.function.arguments)
        validated = CreateTicketInput(**args)
    except (json.JSONDecodeError, ValidationError) as e:
        return f"Error: invalid input — {e}"

    ticket_id = f"T-{secrets.token_hex(4).upper()}"
    ticket = Ticket(
        event_id=f"agent-{secrets.token_hex(8)}",
        ticket_id=ticket_id,
        title=validated.title,
        priority=validated.priority,
        raw_payload="{}",
    )
    db.add(ticket)
    await db.commit()

    return f"Ticket {ticket_id} created: '{validated.title}' (priority: {validated.priority})"
