from datetime import datetime
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    file_name: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class TicketResponse(BaseModel):
    id: int
    event_id: str
    ticket_id: str
    title: str
    priority: str
    created_at: datetime

    model_config = {"from_attributes": True}


class WebhookAcceptedResponse(BaseModel):
    status: str
