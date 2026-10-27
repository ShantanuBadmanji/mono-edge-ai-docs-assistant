import asyncio
import hashlib
import hmac
import json

from sqlalchemy import select

import app.database as db_module
from app.models import Ticket

WEBHOOK_SECRET = "test-secret"


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payload(event_id: str = "evt_001") -> dict:
    return {
        "event_id": event_id,
        "type": "ticket.created",
        "data": {"ticket_id": "T-55", "title": "Login fails", "priority": "high"},
    }


async def test_invalid_signature(client):
    body = json.dumps(_payload()).encode()
    response = await client.post(
        "/webhooks/tickets",
        content=body,
        headers={"X-Signature": "bad-signature", "Content-Type": "application/json"},
    )
    assert response.status_code == 401


async def test_valid_webhook_accepted(client):
    body = json.dumps(_payload("evt_valid")).encode()
    response = await client.post(
        "/webhooks/tickets",
        content=body,
        headers={"X-Signature": _sign(body), "Content-Type": "application/json"},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


async def test_duplicate_event_processed_only_once(client):
    payload = _payload("evt_dup_42")
    body = json.dumps(payload).encode()
    headers = {"X-Signature": _sign(body), "Content-Type": "application/json"}

    await client.post("/webhooks/tickets", content=body, headers=headers)
    await client.post("/webhooks/tickets", content=body, headers=headers)

    # Yield to the event loop so background tasks can complete
    await asyncio.sleep(0.05)

    async with db_module.AsyncSessionLocal() as db:
        result = await db.execute(
            select(Ticket).where(Ticket.event_id == "evt_dup_42")
        )
        tickets = result.scalars().all()

    assert len(tickets) == 1, f"Expected 1 ticket, got {len(tickets)}"
