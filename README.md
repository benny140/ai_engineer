# AI Engineer

This repository contains:

- a backend FastAPI application for querying a local ChromaDB-backed RAG system
- an ingestion pipeline for rebuilding the Chroma database from raw source files
- a frontend Vite React application for the user interface

## Current Structure

```text
backend/
  app/
    api.py
    cli.py
    multi_agent_rag.py
    ollama_client.py
  ingestion/
    ingest_chroma.py
    ingest_config.json
  config/
    rag_runtime_config.json
  paths.py

frontend/
  src/
  public/
  package.json
  vite.config.js

data/
  raw/
  processed/
  chroma_db/
installs.txt
```

## What Each Part Does

- `data/raw/`: input source files for ingestion
- `data/processed/`: generated intermediate artifacts such as chunk files and reports
- `data/chroma_db/`: generated persistent Chroma database used at query time
- `backend/ingestion/ingest_chroma.py`: preprocessing and ingestion pipeline that rebuilds `data/chroma_db/`
- `backend/app/api.py`: FastAPI app for serving RAG queries
- `backend/app/multi_agent_rag.py`: planner, retrieval, and response flow
- `frontend/`: Vite React app with `npm` scripts for local development and production build

## Important Assumption

The current ingestion pipeline expects `parquet` files in `data/raw/`.

If you have different raw files:

- if they are different parquet datasets with different schemas, you can use the current pipeline directly
- if they are not parquet files, you need a preprocessing step to convert them into parquet before running ingestion

## Setup

Create and activate a virtual environment, then install dependencies from `installs.txt`.

Example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r installs.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

You also need:

- local Ollama running on `http://localhost:11434`
- the `granite4.1:3b` model pulled in Ollama

Example:

```powershell
ollama pull granite4.1:3b
```

## End-to-End Flow

### 1. Put source data into `data/raw/`

Drop your input parquet files into `data/raw/`.

Examples:

- `data/raw/qa.parquet`
- `data/raw/functions.parquet`
- `data/raw/my_new_dataset.parquet`

### 2. Review or update ingestion config

The ingestion config is:

- `backend/ingestion/ingest_config.json`

Important fields:

- `input_dir`: folder containing parquet files
- `output_jsonl`: generated chunk output
- `profile_report`: schema and sample inspection report
- `persist_dir`: Chroma output folder
- `collection_name`: Chroma collection name
- `embedding_model`: embedding model used for ingestion and retrieval
- `column_overrides`: optional per-file list of columns to include

If a new parquet file has a different schema, you can either:

- let the pipeline auto-pick non-empty text/object columns
- or explicitly set `column_overrides` for that file

Example:

```json
{
  "column_overrides": {
    "my_new_dataset.parquet": ["title", "summary", "body"]
  }
}
```

### 3. Inspect new raw files before rebuilding

Run inspect mode first to understand what columns were found and selected:

```powershell
c:/Users/benja/Projects/ai_engineer/.venv/Scripts/python.exe -m backend.ingestion.ingest_chroma --mode inspect
```

This writes a profile report to `data/processed/source_profile.json`.

Use inspect mode when:

- you added new datasets
- column names changed
- you are unsure which text fields are being used

### 4. Rebuild the Chroma database

Run full ingestion:

```powershell
c:/Users/benja/Projects/ai_engineer/.venv/Scripts/python.exe -m backend.ingestion.ingest_chroma --mode ingest
```

What this does:

- profiles the raw parquet files
- converts rows into text records
- chunks them into token-sized pieces
- rebuilds `data/processed/chunks.jsonl`
- deletes and recreates `data/chroma_db/`
- writes all chunks into the Chroma collection

Important:

- ingestion is a rebuild process, not an append process
- existing `data/chroma_db/` is deleted before rebuilding

### 5. Review runtime query config

The query/runtime config is:

- `backend/config/rag_runtime_config.json`

Important fields:

- `vector_store_provider`: vector store backend selector (currently `chroma`)
- `persist_dir`: where query-time Chroma data lives
- `collection_name`: Chroma collection to query
- `embedding_model`: must match the model used for ingestion
- `top_k`: number of chunks retrieved per query
- `max_distance`: strict cosine distance cutoff for normal retrieval
- `fallback_max_distance`: if nothing passes max_distance, allow best chunk up to this distance
- `ollama_max_retries`: retry count used by planner/responder when Ollama calls fail
- `ollama_base_url`: local Ollama endpoint
- `ollama_model`: local generation model
- `ollama_timeout_sec`: timeout for local model responses

### 6. Run the FastAPI backend

Start the API server:

```powershell
c:/Users/benja/Projects/ai_engineer/.venv/Scripts/fastapi.exe dev backend/app/api.py
```

Default local URL:

- `http://127.0.0.1:8000`

Available endpoints:

- `GET /health`
- `POST /query` (optional `session_id` for ephemeral chat memory)
- `POST /query/stream` (SSE stream with backend trace events and final result)

### 7. Start the frontend

From the repo root, run:

```powershell
cd frontend
npm run dev
```

Default local URL:

- `http://127.0.0.1:5173`

Useful frontend commands:

- `npm run dev`: start the Vite development server
- `npm run build`: create a production build in `frontend/dist/`
- `npm run preview`: serve the production build locally
- `npm run lint`: run Oxlint

### 8. Query the backend

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType application/json -Body '{"question":"What is this knowledge base about?"}'
```

Example request with session memory:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType application/json -Body '{"session_id":"demo-session","question":"Can you continue from earlier?"}'
```

Session behavior notes:

- session memory is stored in a separate local SQLite file at `data/session/chat_sessions.sqlite3`
- this session database is wiped each time the API starts or restarts
- Chroma RAG storage in `data/chroma_db/` is not affected
- planner and retriever remain stateless; only the responder uses summary + recent verbatim turns

The response includes:

- `answer`
- `plan`
- `sources`
- `session_id`

For `POST /query`, you can set `include_trace=true` to include a `trace` array in the final JSON response.

For `POST /query/stream`, events include:

- `started`: stream initialized
- `trace`: planner/retriever/responder pipeline events
- `completed`: final payload with `result` and `session_id`
- `error`: terminal error payload

## If You Have Different Raw Files

### Case 1: New parquet files with different columns

Use the same flow:

1. place the files in `data/raw/`
2. run inspect mode
3. review `data/processed/source_profile.json`
4. adjust `backend/ingestion/ingest_config.json` if needed
5. rerun ingest mode

### Case 2: CSV, JSON, text, or other formats

The current ingestion script does not ingest those directly.

You need an earlier preprocessing step to convert them into parquet first, or the ingestion code must be extended to support those formats.

Recommended approach:

1. normalize the source into a dataframe-like structure
2. export to parquet
3. place the parquet in `data/raw/`
4. run the normal inspect and ingest flow

## Generated Files

These are produced during ingestion:

- `data/processed/source_profile.json`
- `data/processed/chunks.jsonl`
- `data/processed/ingest_manifest.json`
- `data/chroma_db/`

These should be treated as generated artifacts.

## Common Pitfalls

- `embedding_model` must match between ingestion and query-time retrieval
- `collection_name` must match between ingestion and API runtime config
- `data/chroma_db/` is rebuilt from scratch during ingest mode
- weak retrieval quality often means the wrong columns were selected during ingestion
- non-parquet sources require preprocessing before the current ingestion pipeline can use them

## Current Recommendation

Keep generated runtime artifacts under `data/` and keep backend code under `backend/`.