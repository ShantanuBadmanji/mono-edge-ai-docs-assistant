import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Ticket
from app.schemas import TicketResponse, WebhookAcceptedResponse
from app.services.webhook import process_ticket_event, verify_signature

router = APIRouter(tags=["webhooks"])


@router.post("/webhooks/tickets", response_model=WebhookAcceptedResponse, status_code=202)
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("X-Signature", "")

    if not verify_signature(raw_body, signature, settings.WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
        event_id: str = payload["event_id"]
        data: dict = payload.get("data", {})
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=422, detail="Payload must be JSON with an 'event_id' field")

    background_tasks.add_task(process_ticket_event, event_id, data, raw_body.decode())
    return WebhookAcceptedResponse(status="accepted")


@router.get("/tickets", response_model=list[TicketResponse])
async def list_tickets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Ticket).order_by(Ticket.created_at.desc()))
    return result.scalars().all()
