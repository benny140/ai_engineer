import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from backend.app.ollama_client import OllamaClient, OllamaConfig
from backend.app.vectorstores import RetrievedChunk, VectorStore, create_vector_store
from backend.paths import resolve_repo_path


@dataclass
class RuntimeConfig:
    vector_store_provider: str = "chroma"
    persist_dir: str = "data/chroma_db"
    collection_name: str = "rag_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    max_distance: float = 0.5
    fallback_max_distance: float = 0.7
    ollama_max_retries: int = 1
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "granite4.1:3b"
    ollama_timeout_sec: int = 45


@dataclass
class RetrievalPlan:
    retrieval_query: str
    top_k: int
    response_style: str


class PlannerAgent:
    def __init__(
        self,
        ollama_client: OllamaClient,
        default_top_k: int,
        max_retries: int,
    ):
        self.ollama_client = ollama_client
        self.default_top_k = default_top_k
        self.max_retries = max(0, max_retries)

    def create_plan(
        self,
        question: str,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RetrievalPlan:
        system_prompt = (
            "You are a planning assistant for RAG. "
            "Return strict JSON with keys: retrieval_query, top_k, response_style."
        )
        user_prompt = (
            "Create a retrieval plan for this question. "
            "Keep top_k small and between 1 and 8.\n"
            f"Question: {question}\n"
            f"Default top_k: {self.default_top_k}"
        )

        fallback = RetrievalPlan(
            retrieval_query=question,
            top_k=self.default_top_k,
            response_style="concise and grounded",
        )
        started = time.perf_counter()

        if trace_callback:
            trace_callback(
                {
                    "stage": "planner",
                    "event": "planner_started",
                    "default_top_k": self.default_top_k,
                    "question": question,
                    "llm_system_prompt": system_prompt,
                    "llm_user_prompt": user_prompt,
                }
            )

        attempts = self.max_retries + 1
        raw: Any = None
        for attempt in range(1, attempts + 1):
            try:
                raw = self.ollama_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
                    json_output=True,
                )
                break
            except Exception as exc:
                if trace_callback:
                    trace_callback(
                        {
                            "stage": "planner",
                            "event": "planner_retry",
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "error": str(exc),
                        }
                    )
                if attempt == attempts:
                    if trace_callback:
                        trace_callback(
                            {
                                "stage": "planner",
                                "event": "planner_fallback",
                                "reason": "planner_error",
                                "duration_ms": round(
                                    (time.perf_counter() - started) * 1000, 2
                                ),
                            }
                        )
                    return fallback

        if not isinstance(raw, dict):
            if trace_callback:
                trace_callback(
                    {
                        "stage": "planner",
                        "event": "planner_fallback",
                        "reason": "invalid_planner_json",
                        "raw_response": raw,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
            return fallback

        retrieval_query = str(raw.get("retrieval_query", question)).strip() or question
        top_k = int(raw.get("top_k", self.default_top_k))
        top_k = max(1, min(top_k, 8))
        response_style = (
            str(raw.get("response_style", "concise and grounded")).strip()
            or "concise and grounded"
        )

        plan = RetrievalPlan(
            retrieval_query=retrieval_query,
            top_k=top_k,
            response_style=response_style,
        )
        if trace_callback:
            trace_callback(
                {
                    "stage": "planner",
                    "event": "planner_completed",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "raw_response": raw,
                    "plan": asdict(plan),
                }
            )
        return plan


class RetrievalAgent:
    def __init__(self, config: RuntimeConfig):
        self.store: VectorStore = create_vector_store(
            provider=config.vector_store_provider,
            persist_dir=config.persist_dir,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            max_distance=config.max_distance,
            fallback_max_distance=config.fallback_max_distance,
        )

    def retrieve(
        self,
        plan: RetrievalPlan,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[RetrievedChunk]:
        return self.store.retrieve(
            plan.retrieval_query,
            plan.top_k,
            trace_callback=trace_callback,
        )


class ResponseAgent:
    def __init__(self, ollama_client: OllamaClient, max_retries: int):
        self.ollama_client = ollama_client
        self.max_retries = max(0, max_retries)

    def build_answer(
        self,
        question: str,
        plan: RetrievalPlan,
        chunks: list[RetrievedChunk],
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, str]] | None = None,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if not chunks:
            return (
                "I could not find enough relevant context in the local knowledge base "
                "to answer confidently."
            )

        context_lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            context_lines.append(
                f"[{idx}] source={chunk.source_file} row={chunk.row_index} "
                f"chunk={chunk.chunk_index} score={chunk.score:.3f}\n{chunk.text}"
            )

        system_prompt = (
            "You are a grounded response agent. "
            "Only use the provided context. If context is weak, say so. "
            "Use concise citations like [source_file#row_index]."
        )

        prompt_sections: list[str] = [
            f"Preferred style: {plan.response_style}",
        ]
        if conversation_summary:
            prompt_sections.append(
                "Conversation memory summary:\n" + conversation_summary.strip()
            )
        if recent_messages:
            transcript_lines: list[str] = []
            for msg in recent_messages:
                role = str(msg.get("role", "user")).strip().lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                transcript_lines.append(f"{role}: {content}")
            if transcript_lines:
                prompt_sections.append(
                    "Recent conversation:\n" + "\n".join(transcript_lines)
                )

        prompt_sections.append("Context:\n" + "\n\n".join(context_lines))
        prompt_sections.append(f"Current user question:\n{question}")
        user_prompt = "\n\n".join(prompt_sections)
        started = time.perf_counter()

        if trace_callback:
            trace_callback(
                {
                    "stage": "responder",
                    "event": "responder_started",
                    "chunk_count": len(chunks),
                    "llm_system_prompt": system_prompt,
                    "llm_user_prompt": user_prompt,
                }
            )

        attempts = self.max_retries + 1
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                text = self.ollama_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    json_output=False,
                )
                answer = str(text)
                if trace_callback:
                    trace_callback(
                        {
                            "stage": "responder",
                            "event": "responder_completed",
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "duration_ms": round(
                                (time.perf_counter() - started) * 1000, 2
                            ),
                            "answer": answer,
                            "answer_preview": answer[:280],
                        }
                    )
                return answer
            except Exception as exc:
                last_error = str(exc)
                if trace_callback:
                    trace_callback(
                        {
                            "stage": "responder",
                            "event": "responder_retry",
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "error": str(exc),
                            "elapsed_ms": round(
                                (time.perf_counter() - started) * 1000, 2
                            ),
                        }
                    )
                if attempt < attempts:
                    continue

        source_tags = ", ".join(
            [f"{c.source_file}#row{c.row_index}" for c in chunks[:5]]
        )
        if trace_callback:
            trace_callback(
                {
                    "stage": "responder",
                    "event": "responder_failed",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": last_error,
                    "source_tags": source_tags,
                }
            )
        return (
            "Retrieved context successfully, but the local Ollama response step "
            f"timed out or failed: {last_error}. "
            f"Try increasing ollama_timeout_sec. Sources: {source_tags}"
        )


class MultiAgentRAG:
    def __init__(self, config: RuntimeConfig):
        ollama_client = OllamaClient(
            OllamaConfig(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                timeout_sec=config.ollama_timeout_sec,
            )
        )
        self.planner = PlannerAgent(
            ollama_client, config.top_k, config.ollama_max_retries
        )
        self.retriever = RetrievalAgent(config)
        self.responder = ResponseAgent(ollama_client, config.ollama_max_retries)

    def summarize_history(
        self,
        existing_summary: str,
        older_messages: list[dict[str, Any]],
    ) -> str:
        if not older_messages:
            return existing_summary

        lines: list[str] = []
        for msg in older_messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")

        if not lines:
            return existing_summary

        system_prompt = (
            "You summarize conversation history for a coding assistant. "
            "Keep only durable user goals, constraints, decisions, and unresolved tasks. "
            "Be concise and factual."
        )
        user_prompt = (
            "Update the running summary with these older messages. "
            "Return plain text with short bullet points.\n\n"
            "Current summary:\n"
            f"{existing_summary or '(empty)'}\n\n"
            "New older messages:\n" + "\n".join(lines)
        )

        try:
            text = self.responder.ollama_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                json_output=False,
            )
            return str(text).strip() or existing_summary
        except Exception:
            return existing_summary

    def run(
        self,
        question: str,
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, str]] | None = None,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        plan = self.planner.create_plan(question, trace_callback=trace_callback)
        if trace_callback:
            trace_callback(
                {
                    "stage": "retriever",
                    "event": "retriever_started",
                    "query": plan.retrieval_query,
                    "top_k": plan.top_k,
                }
            )
        retrieval_started = time.perf_counter()
        chunks = self.retriever.retrieve(plan, trace_callback=trace_callback)
        if trace_callback:
            trace_callback(
                {
                    "stage": "retriever",
                    "event": "retriever_completed",
                    "duration_ms": round(
                        (time.perf_counter() - retrieval_started) * 1000, 2
                    ),
                    "chunk_count": len(chunks),
                    "chunks": [
                        {
                            "source_file": c.source_file,
                            "row_index": c.row_index,
                            "chunk_index": c.chunk_index,
                            "score": round(c.score, 4),
                            "text_preview": c.text[:300],
                        }
                        for c in chunks
                    ],
                }
            )
        answer = self.responder.build_answer(
            question,
            plan,
            chunks,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            trace_callback=trace_callback,
        )

        sources = [
            {
                "source_file": c.source_file,
                "row_index": c.row_index,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]

        result = {
            "answer": answer,
            "plan": asdict(plan),
            "sources": sources,
        }
        if trace_callback:
            trace_callback(
                {
                    "stage": "pipeline",
                    "event": "pipeline_completed",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "source_count": len(sources),
                }
            )
        return result


def load_runtime_config(
    runtime_config_path: str = "backend/config/rag_runtime_config.json",
    ingest_config_path: str = "backend/ingestion/ingest_config.json",
) -> RuntimeConfig:
    data: dict[str, Any] = {}

    ingest_path = resolve_repo_path(ingest_config_path)
    if ingest_path.exists():
        with ingest_path.open("r", encoding="utf-8") as f:
            ingest = json.load(f)
            data["persist_dir"] = ingest.get("persist_dir", "data/chroma_db")
            data["collection_name"] = ingest.get("collection_name", "rag_docs")
            data["embedding_model"] = ingest.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )

    runtime_path = resolve_repo_path(runtime_config_path)
    if runtime_path.exists():
        with runtime_path.open("r", encoding="utf-8") as f:
            runtime = json.load(f)
            data.update(runtime)

    return RuntimeConfig(
        vector_store_provider=str(data.get("vector_store_provider", "chroma")),
        persist_dir=str(data.get("persist_dir", "data/chroma_db")),
        collection_name=str(data.get("collection_name", "rag_docs")),
        embedding_model=str(
            data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        top_k=int(data.get("top_k", 5)),
        max_distance=float(data.get("max_distance", 0.5)),
        fallback_max_distance=float(data.get("fallback_max_distance", 0.7)),
        ollama_max_retries=int(data.get("ollama_max_retries", 1)),
        ollama_base_url=str(data.get("ollama_base_url", "http://localhost:11434")),
        ollama_model=str(data.get("ollama_model", "granite4.1:3b")),
        ollama_timeout_sec=int(data.get("ollama_timeout_sec", 45)),
    )
