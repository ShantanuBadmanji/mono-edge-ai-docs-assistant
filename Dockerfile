FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first to keep the image lean (~300 MB vs ~2 GB for CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Pre-create the data directory (volume mount overrides it at runtime)
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite+aiosqlite:///./data/data.db \
    CHROMA_PATH=./data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
