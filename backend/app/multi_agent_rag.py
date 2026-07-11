import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from backend.app.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from backend.app.vectorstores import RetrievedChunk, VectorStore, create_vector_store
from backend.paths import resolve_repo_path


@dataclass
class RuntimeConfig:
    llm_provider: str = "ollama"
    vector_store_provider: str = "chroma"
    persist_dir: str = "data/chroma_db"
    collection_name: str = "rag_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    azure_search_endpoint: str = ""
    azure_search_index_name: str = "rag-docs"
    azure_search_api_key_env: str = "AZURE_SEARCH_API_KEY"
    azure_search_mode: str = "hybrid"
    azure_search_min_score: float = 0.0
    azure_search_fallback_min_score: float = 0.0
    top_k: int = 5
    max_distance: float = 0.5
    fallback_max_distance: float = 0.7
    ollama_max_retries: int = 1
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "granite4.1:3b"
    ollama_timeout_sec: int = 45
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_timeout_sec: int = 60


@dataclass
class RetrievalPlan:
    retrieval_query: str
    top_k: int
    response_style: str
    is_in_scope: bool = True
    scope_reason: str = ""


class PlannerAgent:
    def __init__(
        self,
        llm_client: Any,
        default_top_k: int,
        max_retries: int,
    ):
        self.llm_client = llm_client
        self.default_top_k = default_top_k
        self.max_retries = max(0, max_retries)

    def create_plan(
        self,
        question: str,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RetrievalPlan:
        system_prompt = (
            "You are a planning assistant for a NIST Cybersecurity Framework 2.0 RAG service. "
            "Return strict JSON with keys: retrieval_query, top_k, response_style, is_in_scope, scope_reason. "
            "Set is_in_scope=true for questions about cybersecurity controls, risk, governance, detection, response, recovery, or implementation practices that can be grounded in NIST CSF 2.0. "
            "Treat domain-specific cybersecurity questions (for example API security controls) as in scope. "
            "Set is_in_scope=false only for clearly unrelated topics."
        )
        user_prompt = (
            "Create a retrieval plan for this question. "
            "Keep top_k small and between 1 and 3.\n"
            f"Question: {question}\n"
            f"Default top_k: {self.default_top_k}"
        )

        fallback = RetrievalPlan(
            retrieval_query=question,
            top_k=max(1, min(self.default_top_k, 3)),
            response_style="brief bullets",
            is_in_scope=True,
            scope_reason="fallback_default_in_scope",
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
                raw = self.llm_client.chat(
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
        top_k_raw = raw.get("top_k", self.default_top_k)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            top_k = self.default_top_k
        top_k = max(1, min(top_k, 3))
        response_style = "brief bullets"
        is_in_scope_raw = raw.get("is_in_scope", True)
        if isinstance(is_in_scope_raw, bool):
            is_in_scope = is_in_scope_raw
        elif isinstance(is_in_scope_raw, str):
            is_in_scope = is_in_scope_raw.strip().lower() in {"true", "1", "yes"}
        else:
            is_in_scope = bool(is_in_scope_raw)
        scope_reason = str(raw.get("scope_reason", "")).strip()

        plan = RetrievalPlan(
            retrieval_query=retrieval_query,
            top_k=top_k,
            response_style=response_style,
            is_in_scope=is_in_scope,
            scope_reason=scope_reason,
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
            azure_search_endpoint=config.azure_search_endpoint,
            azure_search_index_name=config.azure_search_index_name,
            azure_search_api_key_env=config.azure_search_api_key_env,
            azure_search_mode=config.azure_search_mode,
            azure_search_min_score=config.azure_search_min_score,
            azure_search_fallback_min_score=config.azure_search_fallback_min_score,
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
    def __init__(self, llm_client: Any, max_retries: int):
        self.llm_client = llm_client
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
            "Give a short answer in at most 5 bullets and at most 120 words total. "
            "List applicable controls first, then one short line on why each applies. "
            "Do not include long explanations unless the user asks for details. "
            "Use concise citations like [source_file#row_index]."
        )

        prompt_sections: list[str] = [
            "Preferred style: brief bullets",
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
                text = self.llm_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
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
            "Retrieved context successfully, but the LLM response step "
            f"timed out or failed: {last_error}. "
            "Try increasing the configured timeout. "
            f"Sources: {source_tags}"
        )


def build_llm_client(config: RuntimeConfig) -> Any:
    provider = config.llm_provider.strip().lower()
    if provider == "azure_openai":
        load_dotenv(resolve_repo_path(".env"))
        return AzureOpenAIClient(
            AzureOpenAIConfig(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
                deployment=os.getenv("DEPLOYMENT_NAME", "").strip(),
                api_version=config.azure_openai_api_version,
                timeout_sec=config.azure_openai_timeout_sec,
            )
        )
    raise ValueError(
        "Unsupported llm_provider for this deployment: "
        f"{config.llm_provider}. Expected: azure_openai"
    )


class MultiAgentRAG:
    def __init__(self, config: RuntimeConfig):
        if config.vector_store_provider.strip().lower() != "azure_search":
            raise ValueError(
                "Unsupported vector_store_provider for this deployment: "
                f"{config.vector_store_provider}. Expected: azure_search"
            )
        llm_client = build_llm_client(config)
        self.planner = PlannerAgent(llm_client, config.top_k, config.ollama_max_retries)
        self.retriever = RetrievalAgent(config)
        self.responder = ResponseAgent(llm_client, config.ollama_max_retries)

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
            text = self.responder.llm_client.chat(
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
        if not plan.is_in_scope:
            if trace_callback:
                trace_callback(
                    {
                        "stage": "pipeline",
                        "event": "scope_rejected",
                        "scope_reason": plan.scope_reason,
                    }
                )
            result = {
                "answer": "This question is not appropriate for this service.",
                "plan": asdict(plan),
                "sources": [],
            }
            if trace_callback:
                trace_callback(
                    {
                        "stage": "pipeline",
                        "event": "pipeline_completed",
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "source_count": 0,
                    }
                )
            return result

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
    ingest_config_path: str = "backend/ingestion/chroma/ingest_config.json",
) -> RuntimeConfig:
    data: dict[str, Any] = {}

    ingest_path = resolve_repo_path(ingest_config_path)
    if not ingest_path.exists():
        ingest_path = resolve_repo_path("backend/ingestion/ingest_config.json")

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
        llm_provider=str(data.get("llm_provider", "ollama")),
        vector_store_provider=str(data.get("vector_store_provider", "chroma")),
        persist_dir=str(data.get("persist_dir", "data/chroma_db")),
        collection_name=str(data.get("collection_name", "rag_docs")),
        embedding_model=str(
            data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        azure_search_endpoint=str(data.get("azure_search_endpoint", "")),
        azure_search_index_name=str(data.get("azure_search_index_name", "rag-docs")),
        azure_search_api_key_env=str(
            data.get("azure_search_api_key_env", "AZURE_SEARCH_API_KEY")
        ),
        azure_search_mode=str(data.get("azure_search_mode", "hybrid")),
        azure_search_min_score=float(data.get("azure_search_min_score", 0.0)),
        azure_search_fallback_min_score=float(
            data.get("azure_search_fallback_min_score", 0.0)
        ),
        top_k=int(data.get("top_k", 5)),
        max_distance=float(data.get("max_distance", 0.5)),
        fallback_max_distance=float(data.get("fallback_max_distance", 0.7)),
        ollama_max_retries=int(data.get("ollama_max_retries", 1)),
        ollama_base_url=str(data.get("ollama_base_url", "http://localhost:11434")),
        ollama_model=str(data.get("ollama_model", "granite4.1:3b")),
        ollama_timeout_sec=int(data.get("ollama_timeout_sec", 45)),
        azure_openai_api_version=str(
            data.get("azure_openai_api_version", "2024-10-21")
        ),
        azure_openai_timeout_sec=int(data.get("azure_openai_timeout_sec", 60)),
    )
