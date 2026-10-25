import asyncio
import json
import secrets
from typing import Literal

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Ticket

# Intent detection: short, focused only on whether to call a tool
_INTENT_SYSTEM_PROMPT = (
    "You are a router. "
    "If the user wants to create or raise a support ticket, ALWAYS call the create_ticket tool immediately. "
    "Infer a reasonable title from the user's message and choose a suitable priority. "
    "Never ask for clarification — make your best guess and call the tool. "
    "For everything else, reply with an empty response — do not answer the question."
)

# RAG answering: strict doc-only response
_RAG_SYSTEM_PROMPT = (
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
            "description": "Create a support ticket when the user explicitly asks to raise or create one.",
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

DISTANCE_THRESHOLD = 0.75


async def answer_question(
    question: str,
    model,
    collection,
    db: AsyncSession,
) -> dict:
    groq = AsyncGroq(api_key=settings.GROQ_API_KEY)

    # Step 1 — intent detection (no RAG, no embed, no Chroma)
    intent_resp = await groq.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tools=_TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    intent_msg = intent_resp.choices[0].message

    print("Intent Message: ", json.dumps(intent_msg, indent=4, default= lambda x: x.__dict__))

    if intent_msg.tool_calls:
        return await _handle_tool_call(intent_msg, db)

    # Step 2 — RAG pipeline (only reached for non-tool intents)
    return await _answer_from_rag(question, model, collection, groq)


async def _handle_tool_call(intent_msg, db: AsyncSession) -> dict:
    tool_call = intent_msg.tool_calls[0]
    if tool_call.function.name != "create_ticket":
        return {"answer": f"Unknown tool requested: {tool_call.function.name}", "sources": []}
    tool_result = await _run_create_ticket(tool_call, db)
    return {"answer": tool_result, "sources": []}


async def _answer_from_rag(question: str, model, collection, groq: AsyncGroq) -> dict:
    q_embedding = await asyncio.to_thread(lambda: model.encode([question])[0].tolist())

    results = await asyncio.to_thread(
        lambda: collection.query(
            query_embeddings=[q_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
    )

    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_distances = results["distances"][0]

    docs = [d for d, dist in zip(raw_docs, raw_distances) if dist < DISTANCE_THRESHOLD]
    metas = [m for m, dist in zip(raw_metas, raw_distances) if dist < DISTANCE_THRESHOLD]

    if not docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Source: {m['file_name']}, page {m['page']}]\n{doc}"
        for doc, m in zip(docs, metas)
    )

    response = await groq.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document excerpts:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    seen: set[tuple] = set()
    sources = []
    for m in metas:
        key = (m["file_name"], m["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file_name": m["file_name"], "page": m["page"]})

    return {"answer": response.choices[0].message.content, "sources": sources}


async def _run_create_ticket(tool_call, db: AsyncSession) -> str:
    class CreateTicketInput(BaseModel):
        title: str
        priority: Literal["low", "medium", "high"]

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
