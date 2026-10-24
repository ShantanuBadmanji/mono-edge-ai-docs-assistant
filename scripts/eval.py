"""
Retrieval evaluation script.

Usage:
  python scripts/eval.py

Requires documents to be indexed first via POST /documents.
Edit TEST_CASES below to match the content of your uploaded documents.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings

TEST_CASES = [
    {
        "question": "What is the main topic of the document?",
        "expected_keywords": [],
        "note": "Replace with real Q&A pairs from your uploaded documents.",
    },
    {
        "question": "Who are the authors?",
        "expected_keywords": [],
        "note": "",
    },
    {
        "question": "What conclusions are drawn?",
        "expected_keywords": [],
        "note": "",
    },
    {
        "question": "What methodology was used?",
        "expected_keywords": [],
        "note": "",
    },
    {
        "question": "What are the key findings?",
        "expected_keywords": [],
        "note": "",
    },
]


def evaluate():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to ChromaDB at '{settings.CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = client.get_or_create_collection("documents")

    count = collection.count()
    if count == 0:
        print("ERROR: No documents indexed. Run POST /documents first.")
        sys.exit(1)

    print(f"Collection has {count} chunks. Running {len(TEST_CASES)} test cases...\n")

    hits = 0
    for i, tc in enumerate(TEST_CASES, start=1):
        q_embedding = model.encode([tc["question"]])[0].tolist()
        results = collection.query(query_embeddings=[q_embedding], n_results=3)
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]

        if tc["expected_keywords"]:
            found = any(
                kw.lower() in doc.lower()
                for doc in retrieved_docs
                for kw in tc["expected_keywords"]
            )
        else:
            found = bool(retrieved_docs)

        status = "HIT " if found else "MISS"
        if found:
            hits += 1

        print(f"[{status}] Q{i}: {tc['question']}")
        if retrieved_metas:
            top = retrieved_metas[0]
            print(f"       Top result: {top['file_name']} page {top['page']}")
        if tc.get("note"):
            print(f"       Note: {tc['note']}")

    print(f"\nRetrieval accuracy: {hits}/{len(TEST_CASES)} = {hits / len(TEST_CASES) * 100:.1f}%")


if __name__ == "__main__":
    evaluate()
