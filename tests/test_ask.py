import pytest


async def test_ask_no_documents_returns_not_available(client):
    """When the vector store is empty the service must not hallucinate."""
    response = await client.post("/ask", json={"question": "What is the refund policy?"})
    assert response.status_code == 200
    data = response.json()
    assert "not available" in data["answer"].lower()
    assert data["sources"] == []


async def test_ask_missing_question_field(client):
    response = await client.post("/ask", json={})
    assert response.status_code == 422


async def test_ask_empty_question(client):
    """An empty string question should succeed but return not available (no docs indexed)."""
    response = await client.post("/ask", json={"question": ""})
    assert response.status_code == 200
    data = response.json()
    assert "not available" in data["answer"].lower()
