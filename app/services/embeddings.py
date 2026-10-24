import io
import hashlib

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _parse_pdf(file_bytes: bytes) -> list[tuple[str, int]]:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages: list[tuple[str, int]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, page_num))
    return pages


def _parse_text(file_bytes: bytes) -> list[tuple[str, int]]:
    return [(file_bytes.decode("utf-8", errors="replace"), 1)]


def _chunk_pages(pages: list[tuple[str, int]]) -> list[tuple[str, int]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for text, page_num in pages:
        for chunk in splitter.split_text(text):
            if chunk.strip():
                chunks.append((chunk, page_num))
    return chunks


def index_document(file_bytes: bytes, file_name: str, content_type: str, model, collection) -> int:
    """Parse, chunk, embed, and store a document. Returns number of chunks indexed."""
    if content_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        pages = _parse_pdf(file_bytes)
    else:
        pages = _parse_text(file_bytes)

    chunks = _chunk_pages(pages)
    if not chunks:
        return 0

    texts = [c[0] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [
        hashlib.md5(f"{file_name}:chunk:{i}:{text[:40]}".encode()).hexdigest()
        for i, text in enumerate(texts)
    ]
    metadatas = [{"file_name": file_name, "page": page_num} for _, page_num in chunks]

    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return len(chunks)
