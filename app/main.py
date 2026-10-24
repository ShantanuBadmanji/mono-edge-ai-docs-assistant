import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.database import Base, engine
from app.routers import ask, documents, webhooks


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(app.state, "test_mode", False):
        os.makedirs(os.path.dirname(settings.DATABASE_URL.split("///")[-1]), exist_ok=True)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        os.makedirs(settings.CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        app.state.chroma_collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    yield


app = FastAPI(title="AI Docs Assistant", version="1.0.0", lifespan=lifespan)

app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(webhooks.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
