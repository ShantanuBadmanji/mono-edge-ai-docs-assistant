# AI Docs Assistant

A FastAPI backend that lets you upload documents, ask questions about them using RAG, and receive webhook events from external systems.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY (free at https://console.groq.com) and WEBHOOK_SECRET

# 2. Run with Docker
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Running locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only torch keeps the install lean
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env   # fill in your values

uvicorn app.main:app --reload
```

## API Reference

### Upload a document
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@your-document.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

Response:
```json
{
  "answer": "The document covers ...",
  "sources": [{"file_name": "your-document.pdf", "page": 3}]
}
```

### Send a webhook event
```bash
SECRET="your-webhook-secret"
BODY='{"event_id":"evt_001","type":"ticket.created","data":{"ticket_id":"T-55","title":"Login fails","priority":"high"}}'
SIG=$(echo -n "$BODY" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}')

curl -X POST http://localhost:8000/webhooks/tickets \
  -H "Content-Type: application/json" \
  -H "X-Signature: $SIG" \
  -d "$BODY"
```

### List tickets
```bash
curl http://localhost:8000/tickets
```

## Running tests

```bash
pytest tests/ -v
```

Tests use an in-memory SQLite DB and mock the embedding model/Chroma, so no API keys or running services are needed.

## Retrieval evaluation (bonus)

Index some documents first, then:
```bash
python scripts/eval.py
```

Edit `TEST_CASES` in the script to match your uploaded documents.

---

## Design Choices

### Chunk size: 800 characters, 150 overlap
Balances two competing needs: small enough that each chunk maps to a single coherent concept (improving retrieval precision), large enough to give the LLM sufficient context for a meaningful answer. At ~4 chars/token, 800 chars ≈ 200 tokens — well within the LLM's context limit even with 5 retrieved chunks.

### Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Runs entirely on CPU — no GPU or API key required
- ~90 MB, downloaded once and cached in `~/.cache/huggingface`
- 384-dimensional vectors with strong semantic quality for English text
- Free, reproducible, no rate limits

### Vector database: ChromaDB (embedded mode)
- File-based storage (`data/chroma/`) — zero configuration, no extra service
- Mounted as a Docker volume for persistence across restarts
- Cosine similarity metric via `hnsw:space: cosine`
- Easy to swap to Qdrant Cloud or a Chroma server for production scale

### LLM: Groq `llama-3.3-70b-versatile`
- Completely free tier at [console.groq.com](https://console.groq.com) — no billing required
- OpenAI-compatible API (easy to swap to Claude or GPT-4 by changing one env var)
- Supports tool/function calling for the agent feature
- `temperature=0` for deterministic, factual RAG answers

### Relational DB: SQLite via `aiosqlite`
- Single file, zero configuration, built into Python
- Perfectly adequate for ticket storage at assessment/early-production scale
- Swap path to a Postgres URL when you need concurrent writes or advanced queries

### Webhook idempotency
A dedicated `processed_events` table stores every `event_id` that has been handled. The background task checks this table atomically before inserting, so duplicate events are silently dropped.

### HMAC verification
Uses `hmac.compare_digest()` (constant-time comparison) to prevent timing attacks. The raw request body is signed before JSON parsing so whitespace/ordering is preserved exactly as sent.

---

## Deployment on AWS

### Architecture

```
Internet → ALB → ECS Fargate (app container) → EFS volume (SQLite + Chroma)
```

| Component | Service | Notes |
|---|---|---|
| Container registry | ECR | Store the app image |
| App runtime | ECS Fargate | Serverless containers, no EC2 to manage |
| Persistent storage | EFS | Mounted into the Fargate task for SQLite + ChromaDB files |
| Load balancer | ALB | HTTPS termination, health checks |

### Approximate monthly cost (1 Fargate task, minimal load)

| Item | Est. cost |
|---|---|
| ECS Fargate (0.25 vCPU, 0.5 GB, ~730 hrs) | ~$10 |
| EFS (< 1 GB) | < $1 |
| ALB | ~$18 |
| ECR (1 GB) | < $1 |
| Data transfer | ~$2 |
| **Total** | **~$30–35 / month** |

### Deployment steps
1. `aws ecr create-repository --repository-name ai-docs-assistant`
2. `docker build -t ai-docs-assistant .` → push to ECR
3. Create ECS cluster + Fargate task definition (mount EFS at `/app/data`)
4. Create ALB → target group → ECS service
5. Set `GROQ_API_KEY` and `WEBHOOK_SECRET` as ECS task environment variables (or via Secrets Manager)

### Production upgrades (with more time/budget)
- Replace SQLite → RDS PostgreSQL + pgvector extension (handles concurrent writes, vector search in one query)
- Replace ChromaDB embedded → Qdrant Cloud (managed, scalable, replicated)
- Add Redis + Celery for robust background task processing
- Stream LLM responses via Server-Sent Events

---

## What I Would Improve With More Time

1. **Streaming responses** — pipe Groq's streaming API to SSE so users see the answer as it's generated
2. **Semantic chunking** — split on section headings rather than character count; preserves document structure
3. **Re-ranking** — add a cross-encoder reranker pass after vector retrieval to improve answer accuracy
4. **Auth layer** — API key auth on all endpoints so the service isn't open to the internet
5. **Async embeddings** — move document indexing to a Celery worker queue so large uploads don't block
6. **Multi-tenancy** — namespace Chroma collections per user/org so each tenant only sees their own documents
7. **Observability** — structured JSON logging + OpenTelemetry traces to Grafana/Honeycomb
