import asyncio

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas import DocumentUploadResponse
from app.services.embeddings import index_document

router = APIRouter(tags=["documents"])

_ALLOWED_EXTENSIONS = {".pdf", ".txt"}
_MAX_SIZE_MB = 20


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), request: Request = None):
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Upload PDF or TXT.")

    file_bytes = await file.read()
    if len(file_bytes) > _MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {_MAX_SIZE_MB} MB limit.")

    model = request.app.state.embedding_model
    collection = request.app.state.chroma_collection

    count = await asyncio.to_thread(
        index_document, file_bytes, filename, file.content_type or "", model, collection
    )

    if count == 0:
        raise HTTPException(status_code=422, detail="No text content could be extracted from the file.")

    return DocumentUploadResponse(
        message=f"Indexed {count} chunks from '{filename}'",
        chunks_indexed=count,
    )
