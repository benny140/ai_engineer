# Backend

## Install

From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

The query pipeline uses Microsoft Agent Framework's `OpenAIChatCompletionClient`, typed
structured output, workflow executors, and `WorkflowBuilder`. Azure AI Search is a
deterministic retrieval executor between the framework planner and responder agents.

## Configure

Create a `.env` in the repo root with your Azure credentials:

```env
AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-openai-key>
DEPLOYMENT_NAME=<your-chat-deployment-name>
AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_SEARCH_API_KEY=<your-search-admin-key>
```

Set `azure_search_endpoint` in `backend/config/rag_runtime_config.json` to the same
`AZURE_SEARCH_ENDPOINT` value.

## Run API

From repo root:

```powershell
fastapi dev backend/app/api.py
```

API URL: http://127.0.0.1:8000

## Build the Azure AI Search Index

Place `.parquet` sources in `data/raw/`, then from the repo root:

```powershell
python -m backend.ingestion.azure_search.ingest_azure_search --mode ingest
```

Use `--mode inspect` to profile the sources without uploading.
