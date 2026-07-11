import json
import queue
import threading
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.multi_agent_rag import MultiAgentRAG, load_runtime_config
from backend.app.session_store import SessionStore
from backend.paths import resolve_repo_path


app = FastAPI(title="AI Engineer RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECENT_VERBATIM_MESSAGES = 8
SUMMARIZE_IF_TOTAL_MESSAGES_OVER = 14
SUMMARY_BATCH_SIZE = 4

session_store: SessionStore | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    include_trace: bool = False


class QueryResponse(BaseModel):
    answer: str
    plan: dict
    sources: list[dict]
    session_id: str | None = None
    trace: list[dict[str, Any]] | None = None


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _trace_emitter(
    collector: list[dict[str, Any]] | None = None,
    stream_queue: queue.Queue[dict[str, Any]] | None = None,
) -> Callable[[dict[str, Any]], None]:
    seq = 0

    def emit(event: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        payload = {
            "seq": seq,
            "timestamp": time.time(),
            **event,
        }
        if collector is not None:
            collector.append(payload)
        if stream_queue is not None:
            stream_queue.put({"type": "trace", "payload": payload})

    return emit


def _run_query(
    request: QueryRequest,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], str | None]:
    cfg = load_runtime_config()
    rag = MultiAgentRAG(cfg)

    if not request.session_id:
        result = rag.run(request.question, trace_callback=trace_callback)
        return result, None

    if session_store is None:
        result = rag.run(request.question, trace_callback=trace_callback)
        return result, request.session_id

    sid = session_store.ensure_session(request.session_id)
    session_store.add_message(sid, "user", request.question)

    summary_text, last_summarized_message_id = session_store.get_summary(sid)
    total_messages = session_store.get_message_count(sid)
    if total_messages > SUMMARIZE_IF_TOTAL_MESSAGES_OVER:
        unsummarized_older = session_store.get_unsummarized_older_messages(
            sid,
            keep_recent=RECENT_VERBATIM_MESSAGES,
            last_summarized_message_id=last_summarized_message_id,
        )
        if len(unsummarized_older) >= SUMMARY_BATCH_SIZE:
            updated_summary = rag.summarize_history(
                existing_summary=summary_text,
                older_messages=unsummarized_older,
            )
            if updated_summary.strip():
                summary_text = updated_summary.strip()
                newest_id = int(unsummarized_older[-1]["id"])
                session_store.upsert_summary(
                    sid,
                    summary_text=summary_text,
                    last_summarized_message_id=newest_id,
                )

    recent_messages = session_store.get_recent_messages(
        sid,
        limit=RECENT_VERBATIM_MESSAGES + 1,
    )
    if recent_messages:
        last = recent_messages[-1]
        if (
            str(last.get("role", "")).lower() == "user"
            and str(last.get("content", "")).strip() == request.question.strip()
        ):
            recent_messages = recent_messages[:-1]

    result = rag.run(
        request.question,
        conversation_summary=summary_text or None,
        recent_messages=[
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in recent_messages
        ],
        trace_callback=trace_callback,
    )

    session_store.add_message(sid, "assistant", str(result.get("answer", "")))
    return result, sid


@app.on_event("startup")
def startup() -> None:
    global session_store
    db_path = resolve_repo_path("data/session/chat_sessions.sqlite3")
    session_store = SessionStore(db_path)
    session_store.initialize(reset=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    try:
        trace_items: list[dict[str, Any]] = []
        trace_callback = (
            _trace_emitter(collector=trace_items) if request.include_trace else None
        )
        result, sid = _run_query(request, trace_callback=trace_callback)
        return QueryResponse(
            **result,
            session_id=sid,
            trace=trace_items if request.include_trace else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query/stream")
def query_rag_stream(request: QueryRequest) -> StreamingResponse:
    stream_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def worker() -> None:
        try:
            emitter = _trace_emitter(stream_queue=stream_queue)
            result, sid = _run_query(request, trace_callback=emitter)
            stream_queue.put(
                {
                    "type": "completed",
                    "payload": {
                        "result": result,
                        "session_id": sid,
                    },
                }
            )
        except Exception as exc:
            stream_queue.put(
                {
                    "type": "error",
                    "payload": {"message": str(exc)},
                }
            )
        finally:
            stream_queue.put({"type": "done", "payload": {}})

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        yield _sse("started", {"message": "query_started"})
        while True:
            item = stream_queue.get()
            item_type = item.get("type")
            payload = item.get("payload", {})
            if item_type == "trace":
                yield _sse("trace", payload)
                continue
            if item_type == "completed":
                yield _sse("completed", payload)
                continue
            if item_type == "error":
                yield _sse("error", payload)
                break
            if item_type == "done":
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
