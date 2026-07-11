# AI Engineer

Simple local setup for backend + frontend.

## 1. Prerequisites

- Python 3.12+
- Node.js 18+
- Ollama

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

## 3. Start Ollama

```powershell
ollama pull granite4.1:3b
ollama serve
```

Optional check:

```powershell
curl http://localhost:11434/api/tags
```

## 4. Start Backend

```powershell
fastapi dev backend/app/api.py
```

Backend URL: http://127.0.0.1:8000

## 5. Start Frontend

```powershell
cd frontend
npm run dev
```

Frontend URL: http://127.0.0.1:5173

## Stop Ollama

```powershell
Get-Process ollama, "ollama app" -ErrorAction SilentlyContinue | Stop-Process
```
