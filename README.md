# AI Docs Assistant

A FastAPI backend that lets you upload documents, ask questions about them using RAG, and receive webhook events from external systems.

Built as a take-home assessment for MonoEdge AI.

---

## Quick Start (Docker)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY (free at https://console.groq.com) and WEBHOOK_SECRET

# 2. Start the app
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Running Locally

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- Python 3.11 (uv downloads it automatically if not present)

### Setup

```bash
# Create the virtual environment pinned to Python 3.11
uv venv --python 3.11

# Install CPU-only PyTorch first (avoids downloading the 2 GB CUDA build)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY and WEBHOOK_SECRET
```

### Run

```bash
uv run uvicorn app.main:app --reload
```

> The `all-MiniLM-L6-v2` embedding model (~90 MB) downloads automatically on first startup and is cached in `~/.cache/huggingface`.

---

## API Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@your-document.pdf"
```

Response:
```json
{"message": "Indexed 12 chunks from your-document.pdf"}
```

Accepts `.pdf` and `.txt` files.

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

If the answer is not in the uploaded documents the response is:
```json
{"answer": "The answer is not available in the provided documents.", "sources": []}
```

### Raise a ticket via natural language (bonus)

You can ask `/ask` to create a support ticket directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Please raise a high-priority ticket: login page is broken"}'
```

Response:
```json
{"answer": "Ticket T-3A1F9C created: 'Login page is broken' (priority: high)", "sources": []}
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

Response: `202 Accepted` with `{"status": "accepted"}`.  
Invalid or missing signature returns `401`.  
Duplicate `event_id` is silently deduplicated — you still get `202` but the event is not reprocessed.

### List tickets

```bash
curl http://localhost:8000/tickets
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests use an in-memory SQLite database and mock the embedding model/Chroma. No API keys or running services are needed.

```
tests/test_webhooks.py::test_invalid_signature          PASSED
tests/test_webhooks.py::test_valid_webhook_accepted     PASSED
tests/test_webhooks.py::test_duplicate_event_processed_only_once  PASSED
tests/test_ask.py::test_ask_no_documents                PASSED
tests/test_ask.py::test_ask_returns_answer_and_sources  PASSED
tests/test_ask.py::test_ask_creates_ticket_via_tool     PASSED
```

---

## Retrieval Evaluation (Bonus)

Index some documents first, then run the evaluation script:

```bash
uv run python scripts/eval.py
```

Edit `TEST_CASES` in the script to match your uploaded documents. The script reports hit-rate: how often the correct chunk appears in the top-5 retrieved results.

---

## Design Choices

### Chunk size: 800 characters, 150 overlap
Balances precision and context: small enough that each chunk maps to a single concept (improves retrieval accuracy), large enough to give the LLM enough text to form a meaningful answer. At ~4 chars/token this is ~200 tokens per chunk — comfortably within the LLM's context window even with 5 chunks retrieved.

### Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Runs entirely on CPU — no GPU or API key needed
- ~90 MB, downloaded once on first run and cached locally
- 384-dimensional vectors with strong semantic quality for English text
- Zero cost, no rate limits, fully reproducible

PyTorch (`torch`) is installed from the CPU-only wheel index (`download.pytorch.org/whl/cpu`) to keep the Docker image at ~300 MB instead of ~2 GB for the CUDA build.

### Vector database: ChromaDB (embedded mode)
- File-based (`data/chroma/`) — zero configuration, no extra service to run
- Mounted as a Docker volume for persistence across restarts
- Cosine similarity with a distance threshold (0.75) to filter irrelevant chunks before they reach the LLM
- Easy to swap to Qdrant Cloud or a dedicated Chroma server for production

### LLM: Groq (free tier)
- Completely free at [console.groq.com](https://console.groq.com) — no billing required
- OpenAI-compatible API — swap to Claude or GPT-4 by changing one env var (`GROQ_MODEL`)
- Supports tool/function calling for the agent ticket-creation feature
- `temperature=0` for deterministic, factual RAG answers

### Intent-first router in `/ask`
Two separate LLM calls with focused system prompts:
1. **Intent call** — no RAG, just decide: tool or question? If a ticket is requested, call the tool immediately and return.
2. **RAG call** — only reached for document questions: embed → vector search → answer from chunks.

This avoids running the vector search for ticket requests (saves latency and Chroma overhead) and keeps the two concerns cleanly separated.

### Webhook idempotency
A `processed_events` table stores every `event_id` that has been handled. The background task checks this table before inserting — if the ID already exists, the event is silently dropped. Guarantees exactly-once processing regardless of retries.

### HMAC signature verification
Uses `hmac.compare_digest()` (constant-time string comparison) to prevent timing attacks. The raw request body is signed before JSON parsing — whitespace and key ordering are preserved exactly as the sender transmitted them.

### Relational DB: SQLite via `aiosqlite`
Single file, zero configuration. Adequate for ticket storage at assessment and early-production scale. Swap the `DATABASE_URL` to a Postgres connection string when you need concurrent writes or advanced queries — SQLAlchemy async makes this a one-line change.

---

## Deployment on AWS

### Architecture

```
Internet → ALB → ECS Fargate (app container) → EFS volume (SQLite + Chroma)
```

| Component | Service | Notes |
|---|---|---|
| Container registry | ECR | Store the Docker image |
| App runtime | ECS Fargate | Serverless containers — no EC2 to manage |
| Persistent storage | EFS | Mounted at `/app/data` for SQLite + ChromaDB files |
| Load balancer | ALB | HTTPS termination, health checks |
| Secrets | Secrets Manager | `GROQ_API_KEY`, `WEBHOOK_SECRET` |

### Deployment steps

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name ai-docs-assistant
docker build -t ai-docs-assistant .
# tag and push to ECR

# 2. Create ECS Fargate cluster + task definition
#    Mount EFS at /app/data in the task definition

# 3. Create ALB → target group → ECS service

# 4. Set GROQ_API_KEY and WEBHOOK_SECRET via ECS environment variables or Secrets Manager
```

### Approximate monthly cost

| Item | Spec | Est. cost/month |
|---|---|---|
| ECS Fargate | 0.25 vCPU, 0.5 GB RAM, ~730 hrs | ~$10 |
| EFS | < 1 GB | < $1 |
| ALB | 1 instance | ~$18 |
| ECR | ~1 GB image | < $1 |
| Data transfer | ~1 GB egress | ~$1 |
| Secrets Manager | 2 secrets | < $1 |
| **Total** | | **~$30–35 / month** |

Cost scales linearly with Fargate task count. At 3 tasks (basic HA): ~$65/month.

### Production upgrades (with more time/budget)

| Current | Production swap | Reason |
|---|---|---|
| SQLite | RDS PostgreSQL + pgvector | Concurrent writes, vector search in SQL, managed backups |
| ChromaDB embedded | Qdrant Cloud | Managed, scalable, replicated vector search |
| FastAPI BackgroundTasks | SQS + Lambda or Celery | Durable queue for webhook processing — survives app restarts |
| No auth | API key auth / Cognito | Prevent open access |

Estimated production cost at modest scale: **~$120–180/month** (adds RDS `db.t3.micro` ~$25, Qdrant free tier, SQS < $1).

---

## What I Would Improve With More Time

1. **Streaming responses** — pipe Groq's streaming API to SSE so users see the answer as it's generated
2. **Semantic chunking** — split on section headings and paragraphs rather than character count; preserves document structure and improves retrieval
3. **Re-ranking** — add a cross-encoder reranker pass after vector retrieval to improve answer relevance before the LLM sees the chunks
4. **Auth layer** — API key authentication on all endpoints; without it the service is open to anyone
5. **Async document indexing** — move embedding + Chroma inserts to a background worker queue so large uploads don't block the request thread
6. **Multi-tenancy** — namespace Chroma collections per user/org so each tenant only retrieves from their own documents
7. **Observability** — structured JSON logging + OpenTelemetry traces to Grafana or Honeycomb for latency visibility across the RAG pipeline
