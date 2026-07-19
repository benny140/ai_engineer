# AI Engineer

A RAG assistant orchestrated with **Microsoft Agent Framework**, powered by **Azure
OpenAI** for inference and **Azure AI Search** for retrieval, with a FastAPI backend and a
React (Vite) frontend.

## 1. Prerequisites

- Python 3.12+
- Node.js 18+
- An Azure subscription with the two services below

### Azure services required

| Service | Purpose | What you create |
|---------|---------|-----------------|
| **Azure OpenAI** | Chat completions for Agent Framework planner, responder, and summarizer agents | A resource + a **chat model deployment** (e.g. `gpt-4o-mini`). Note the deployment name. |
| **Azure AI Search** | Hybrid (vector + keyword) retrieval over ingested chunks | A search service. The index (`rag-docs`) is created automatically by the ingestion script. |

## 2. Install

From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

The frozen requirements include `agent-framework` and its OpenAI provider. Microsoft Agent
Framework is currently distributed as a meta-package, so its optional provider dependencies
are also present even though this application only configures the OpenAI provider.

## 3. Configure Azure credentials

Create a `.env` file in the repo root:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-openai-key>
DEPLOYMENT_NAME=<your-chat-deployment-name>

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_SEARCH_API_KEY=<your-search-admin-key>
```

Then point the runtime config at your search service by setting `azure_search_endpoint`
in `backend/config/rag_runtime_config.json` to the same `AZURE_SEARCH_ENDPOINT` value.

## 4. Ingest the knowledge base

The query path is read-only, so you must build the search index first.

1. Place your source `.parquet` files in `data/raw/`.
2. (Optional) Adjust chunking / index settings in
   `backend/ingestion/azure_search/ingest_azure_search_config.json`.
3. Run the ingestion pipeline from the repo root (venv active):

```powershell
# Inspect sources only (profiles the parquet files, no upload)
python -m backend.ingestion.azure_search.ingest_azure_search --mode inspect

# Full run: profile -> chunk -> embed -> rebuild index -> upload
python -m backend.ingestion.azure_search.ingest_azure_search --mode ingest
```

This creates/rebuilds the `rag-docs` index in Azure AI Search and writes
`data/processed/chunks.jsonl`, `source_profile.json`, and `ingest_manifest.json`.

## 5. Start Backend

```powershell
fastapi dev backend/app/api.py
```

Backend URL: http://127.0.0.1:8000

## 6. Start Frontend

```powershell
cd frontend
npm run dev
```

Frontend URL: http://127.0.0.1:5173
