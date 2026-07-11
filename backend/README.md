# Backend

## Install

From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

## Run API

From repo root:

```powershell
fastapi dev backend/app/api.py
```

API URL: http://127.0.0.1:8000

## Optional: Rebuild Vector DB

```powershell
python -m backend.ingestion.ingest_chroma --mode ingest
```

## Ollama Required

```powershell
ollama pull granite4.1:3b
ollama serve
```
