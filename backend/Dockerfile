# Backend image — FastAPI RAG API served by Uvicorn.
# Intended for Azure Container Apps eventually; runs on any Docker host locally.
# Build from the repository root so the `backend` package is importable:
#   docker build -f backend/Dockerfile -t ai-engineer-backend .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend package (kept under /app so `backend.app.api` resolves).
COPY backend ./backend

EXPOSE 8000

# Azure credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, DEPLOYMENT_NAME,
# AZURE_SEARCH_API_KEY) are injected as environment variables / Container Apps secrets.
CMD ["uvicorn", "backend.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
