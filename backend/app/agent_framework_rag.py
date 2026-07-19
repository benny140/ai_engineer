import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.app.vectorstores import RetrievedChunk, VectorStore
from backend.app.vectorstores.azure_search_store import AzureSearchVectorStore
from backend.paths import resolve_repo_path

TraceCallback = Callable[[dict[str, Any]], None]


@dataclass
class RuntimeConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    azure_search_endpoint: str = ""
    azure_search_index_name: str = "rag-docs"
    azure_search_api_key_env: str = "AZURE_SEARCH_API_KEY"
    azure_search_mode: str = "hybrid"
    azure_search_min_score: float = 0.0
    azure_search_fallback_min_score: float = 0.0
    top_k: int = 5
    agent_max_retries: int = 1
    azure_openai_api_version: str = "2024-10-21"


class RetrievalPlan(BaseModel):
    retrieval_query: str
    top_k: int = Field(ge=1)
    response_style: str
    is_in_scope: bool = True
    scope_reason: str = ""


@dataclass
class RAGRequest:
    question: str
    conversation_summary: str | None = None
    recent_messages: list[dict[str, str]] | None = None
    trace_callback: TraceCallback | None = None


@dataclass
class PlannedRequest:
    request: RAGRequest
    plan: RetrievalPlan


@dataclass
class RetrievedRequest:
    request: RAGRequest
    plan: RetrievalPlan
    chunks: list[RetrievedChunk]


@dataclass
class RAGResult:
    answer: str
    plan: dict[str, Any]
    sources: list[dict[str, Any]]


class PlannerExecutor(Executor):
    def __init__(self, agent: Any, default_top_k: int, max_retries: int):
        super().__init__(id="planner")
        self.agent = agent
        self.default_top_k = max(1, default_top_k)
        self.max_retries = max(0, max_retries)

    @handler
    async def plan(
        self,
        request: RAGRequest,
        ctx: WorkflowContext[PlannedRequest],
    ) -> None:
        started = time.perf_counter()
        _emit(
            request.trace_callback,
            stage="planner",
            event="planner_started",
            question=request.question,
            default_top_k=self.default_top_k,
        )
        prompt = (
            "Create a retrieval plan for this NIST Cybersecurity Framework 2.0 question. "
            f"Set top_k to exactly {self.default_top_k}. Mark only clearly unrelated "
            f"questions out of scope. Question: {request.question}"
        )

        try:
            response = await _run_agent(
                self.agent,
                prompt,
                self.max_retries,
                options={"response_format": RetrievalPlan},
            )
            plan = response.value
            if not isinstance(plan, RetrievalPlan):
                plan = RetrievalPlan.model_validate_json(response.text)
            plan.top_k = self.default_top_k
            if not plan.scope_reason:
                plan.scope_reason = "in_scope" if plan.is_in_scope else "out_of_scope"
            _emit(
                request.trace_callback,
                stage="planner",
                event="planner_completed",
                duration_ms=_duration_ms(started),
                plan=plan.model_dump(),
            )
        except Exception as exc:
            plan = RetrievalPlan(
                retrieval_query=request.question,
                top_k=self.default_top_k,
                response_style="brief bullets",
                is_in_scope=False,
                scope_reason="planner_error",
            )
            _emit(
                request.trace_callback,
                stage="planner",
                event="planner_rejected",
                duration_ms=_duration_ms(started),
                reason="planner_error",
                error=str(exc),
                plan=plan.model_dump(),
            )

        await ctx.send_message(PlannedRequest(request=request, plan=plan))


class RetrievalExecutor(Executor):
    def __init__(self, store: VectorStore):
        super().__init__(id="retriever")
        self.store = store

    @handler
    async def retrieve(
        self,
        message: PlannedRequest,
        ctx: WorkflowContext[RetrievedRequest],
    ) -> None:
        request = message.request
        plan = message.plan
        if not plan.is_in_scope:
            _emit(
                request.trace_callback,
                stage="pipeline",
                event="scope_rejected",
                scope_reason=plan.scope_reason,
            )
            await ctx.send_message(
                RetrievedRequest(request=request, plan=plan, chunks=[])
            )
            return

        started = time.perf_counter()
        _emit(
            request.trace_callback,
            stage="retriever",
            event="retriever_started",
            query=plan.retrieval_query,
            top_k=plan.top_k,
        )
        chunks = await asyncio.to_thread(
            self.store.retrieve,
            plan.retrieval_query,
            plan.top_k,
            request.trace_callback,
        )
        _emit(
            request.trace_callback,
            stage="retriever",
            event="retriever_completed",
            duration_ms=_duration_ms(started),
            chunk_count=len(chunks),
            chunks=[_chunk_source(chunk, include_text=True) for chunk in chunks],
        )
        await ctx.send_message(
            RetrievedRequest(request=request, plan=plan, chunks=chunks)
        )


class ResponderExecutor(Executor):
    def __init__(self, agent: Any, max_retries: int):
        super().__init__(id="responder")
        self.agent = agent
        self.max_retries = max(0, max_retries)

    @handler
    async def respond(
        self,
        message: RetrievedRequest,
        ctx: WorkflowContext[Any, RAGResult],
    ) -> None:
        started = time.perf_counter()
        request = message.request
        plan = message.plan

        if not plan.is_in_scope:
            answer = (
                f"Unable to answer: {plan.scope_reason or 'out_of_scope_for_service'}. "
                "This service only handles NIST CSF 2.0 cybersecurity topics."
            )
        elif not message.chunks:
            answer = (
                "Unable to answer: no_related_topic_found. "
                "No sufficiently relevant context was retrieved for this question."
            )
        else:
            prompt = _response_prompt(request, plan, message.chunks)
            _emit(
                request.trace_callback,
                stage="responder",
                event="responder_started",
                chunk_count=len(message.chunks),
            )
            try:
                response = await _run_agent(
                    self.agent,
                    prompt,
                    self.max_retries,
                )
                answer = response.text.strip()
                if not answer:
                    raise RuntimeError("Agent Framework responder returned an empty response")
                _emit(
                    request.trace_callback,
                    stage="responder",
                    event="responder_completed",
                    duration_ms=_duration_ms(started),
                    answer_preview=answer[:280],
                )
            except Exception as exc:
                answer = (
                    "Unable to answer: response_generation_failed. "
                    f"Reason: {exc}"
                )
                _emit(
                    request.trace_callback,
                    stage="responder",
                    event="responder_failed",
                    duration_ms=_duration_ms(started),
                    error=str(exc),
                )

        result = RAGResult(
            answer=answer,
            plan=plan.model_dump(),
            sources=[_chunk_source(chunk) for chunk in message.chunks],
        )
        _emit(
            request.trace_callback,
            stage="pipeline",
            event="pipeline_completed",
            duration_ms=_duration_ms(started),
            source_count=len(result.sources),
        )
        await ctx.yield_output(result)


class AgentFrameworkRAG:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        planner_agent: Any | None = None,
        responder_agent: Any | None = None,
        summarizer_agent: Any | None = None,
        store: VectorStore | None = None,
    ):
        self.config = config
        if planner_agent is None or responder_agent is None or summarizer_agent is None:
            client = _build_chat_client(config)
            planner_agent = planner_agent or client.as_agent(
                name="NISTPlanner",
                instructions=(
                    "You plan retrieval for a NIST CSF 2.0 RAG service. Return the "
                    "requested structured output. Keep retrieval queries concise and mark "
                    "only clearly unrelated questions out of scope."
                ),
            )
            responder_agent = responder_agent or client.as_agent(
                name="GroundedResponder",
                instructions=(
                    "Answer only from the supplied context. Put applicable controls first, "
                    "explain their rationale, cite sources as [source_file#row_index], and "
                    "state when context is weak."
                ),
            )
            summarizer_agent = summarizer_agent or client.as_agent(
                name="ConversationSummarizer",
                instructions=(
                    "Summarize conversation history into concise bullets containing only "
                    "durable goals, constraints, decisions, and unresolved tasks."
                ),
            )

        self.summarizer_agent = summarizer_agent
        vector_store = store or AzureSearchVectorStore(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            api_key_env=config.azure_search_api_key_env,
            embedding_model=config.embedding_model,
            search_mode=config.azure_search_mode,
            min_score=config.azure_search_min_score,
            fallback_min_score=config.azure_search_fallback_min_score,
        )
        planner = PlannerExecutor(
            planner_agent,
            config.top_k,
            config.agent_max_retries,
        )
        retriever = RetrievalExecutor(vector_store)
        responder = ResponderExecutor(responder_agent, config.agent_max_retries)
        self.workflow = (
            WorkflowBuilder(start_executor=planner, output_from=[responder])
            .add_edge(planner, retriever)
            .add_edge(retriever, responder)
            .build()
        )

    async def run_async(
        self,
        question: str,
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, str]] | None = None,
        trace_callback: TraceCallback | None = None,
    ) -> dict[str, Any]:
        result = await self.workflow.run(
            RAGRequest(
                question=question,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
                trace_callback=trace_callback,
            )
        )
        outputs = result.get_outputs()
        if not outputs or not isinstance(outputs[-1], RAGResult):
            raise RuntimeError("Agent Framework workflow returned no RAG result")
        return asdict(outputs[-1])

    def run(
        self,
        question: str,
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, str]] | None = None,
        trace_callback: TraceCallback | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.run_async(
                question,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
                trace_callback=trace_callback,
            )
        )

    def summarize_history(
        self,
        existing_summary: str,
        older_messages: list[dict[str, Any]],
    ) -> str:
        return asyncio.run(
            self.summarize_history_async(existing_summary, older_messages)
        )

    async def summarize_history_async(
        self,
        existing_summary: str,
        older_messages: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"{str(message.get('role', 'user')).lower()}: {str(message.get('content', '')).strip()}"
            for message in older_messages
            if str(message.get("content", "")).strip()
        ]
        if not lines:
            return existing_summary
        prompt = (
            f"Current summary:\n{existing_summary or '(empty)'}\n\n"
            "New older messages:\n" + "\n".join(lines)
        )
        try:
            response = await _run_agent(
                self.summarizer_agent,
                prompt,
                self.config.agent_max_retries,
            )
            return response.text.strip() or existing_summary
        except Exception:
            return existing_summary


def _build_chat_client(config: RuntimeConfig) -> OpenAIChatCompletionClient:
    load_dotenv(resolve_repo_path(".env"))
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("DEPLOYMENT_NAME", "").strip()
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY")
    if not deployment:
        raise ValueError("Missing DEPLOYMENT_NAME")
    return OpenAIChatCompletionClient(
        model=deployment,
        azure_endpoint=endpoint,
        api_version=config.azure_openai_api_version,
        api_key=api_key,
    )


async def _run_agent(
    agent: Any,
    prompt: str,
    max_retries: int,
    options: dict[str, Any] | None = None,
) -> Any:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return await agent.run(prompt, options=options)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Agent Framework run failed: {last_error}") from last_error


def _response_prompt(
    request: RAGRequest,
    plan: RetrievalPlan,
    chunks: list[RetrievedChunk],
) -> str:
    sections = [f"Preferred style: {plan.response_style}"]
    if request.conversation_summary:
        sections.append("Conversation summary:\n" + request.conversation_summary.strip())
    if request.recent_messages:
        transcript = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in request.recent_messages
            if str(message.get("content", "")).strip()
        )
        if transcript:
            sections.append("Recent conversation:\n" + transcript)
    context = "\n\n".join(
        f"[{index}] source={chunk.source_file} row={chunk.row_index} "
        f"chunk={chunk.chunk_index} score={chunk.score:.3f}\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    sections.append("Context:\n" + context)
    sections.append("Current user question:\n" + request.question)
    return "\n\n".join(sections)


def _chunk_source(
    chunk: RetrievedChunk,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "source_file": chunk.source_file,
        "row_index": chunk.row_index,
        "chunk_index": chunk.chunk_index,
        "score": round(chunk.score, 4),
    }
    if include_text:
        source["text_preview"] = chunk.text[:300]
    return source


def _emit(callback: TraceCallback | None, **event: Any) -> None:
    if callback:
        callback(event)


def _duration_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def load_runtime_config(
    runtime_config_path: str = "backend/config/rag_runtime_config.json",
) -> RuntimeConfig:
    path = resolve_repo_path(runtime_config_path)
    data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    return RuntimeConfig(
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
        agent_max_retries=int(data.get("agent_max_retries", 1)),
        azure_openai_api_version=str(
            data.get("azure_openai_api_version", "2024-10-21")
        ),
    )
