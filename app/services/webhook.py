import hashlib
import hmac

from sqlalchemy import select

from app import database
from app.models import ProcessedEvent, Ticket


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def process_ticket_event(event_id: str, data: dict, raw_payload: str) -> None:
    """Background task: idempotently save a webhook event to the database."""
    async with database.AsyncSessionLocal() as db:
        async with db.begin():
            existing = await db.execute(
                select(ProcessedEvent).where(ProcessedEvent.event_id == event_id)
            )
            if existing.scalar_one_or_none() is not None:
                return

            db.add(ProcessedEvent(event_id=event_id))
            db.add(
                Ticket(
                    event_id=event_id,
                    ticket_id=data.get("ticket_id", ""),
                    title=data.get("title", ""),
                    priority=data.get("priority", ""),
                    raw_payload=raw_payload,
                )
            )
