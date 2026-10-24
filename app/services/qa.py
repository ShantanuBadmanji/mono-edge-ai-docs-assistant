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
    "You are a helpful assistant. "
    "When document excerpts are provided, answer strictly from them. "
    "If the user asks to create or raise a ticket, always use the create_ticket tool — "
    "no document context is needed for that. "
    "For all other questions without excerpts, respond with exactly: "
    "'The answer is not available in the provided documents.'"
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

    # Filter to chunks with cosine distance < 0.5 (0 = identical, 2 = opposite)
    DISTANCE_THRESHOLD = 0.75
    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_distances = results["distances"][0]

    print(json.dumps({"raw_metas":raw_metas, "raw_distances" : raw_distances}, indent=4))

    docs = [d for d, dist in zip(raw_docs, raw_distances) if dist < DISTANCE_THRESHOLD]
    metas = [m for m, dist in zip(raw_metas, raw_distances) if dist < DISTANCE_THRESHOLD]

    # Build user message — include doc context only when relevant chunks exist
    if docs:
        context = "\n\n".join(
            f"[Source: {m['file_name']}, page {m['page']}]\n{doc}"
            for doc, m in zip(docs, metas)
        )
        user_content = f"Document excerpts:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
        return {"answer": final.choices[0].message.content, "sources": []}

    # No tool call and no relevant docs — don't hallucinate
    if not docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": [],
        }

    seen: set[tuple] = set()
    sources = []
    for m in metas:
        key = (m["file_name"], m["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file_name": m["file_name"], "page": m["page"]})

    return {"answer": message.content, "sources": sources}


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
