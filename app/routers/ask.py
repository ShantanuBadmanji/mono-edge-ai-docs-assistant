from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import AskRequest, AskResponse
from app.services.qa import answer_question

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request, db: AsyncSession = Depends(get_db)):
    model = request.app.state.embedding_model
    collection = request.app.state.chroma_collection

    result = await answer_question(
        question=body.question,
        model=model,
        collection=collection,
        db=db,
    )
    return result
