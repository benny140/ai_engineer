import os
from typing import Any, Callable

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.app.vectorstores.base import RetrievedChunk
from backend.paths import resolve_repo_path


class AzureSearchVectorStore:
    def __init__(
        self,
        endpoint: str,
        index_name: str,
        api_key_env: str,
        embedding_model: str,
        search_mode: str,
        min_score: float,
        fallback_min_score: float,
    ):
        load_dotenv(resolve_repo_path(".env"))
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")

        self.search_mode = search_mode.strip().lower() or "hybrid"
        self.min_score = min_score
        self.fallback_min_score = fallback_min_score
        self.query_embedder = SentenceTransformer(embedding_model)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )

    def retrieve(
        self,
        query_text: str,
        top_k: int,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self.query_embedder.encode(query_text).tolist()

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="text_vector",
        )

        search_text = query_text if self.search_mode == "hybrid" else None
        results = self.search_client.search(
            search_text=search_text,
            vector_queries=[vector_query],
            top=top_k,
            select=[
                "text",
                "source_file",
                "row_index",
                "parent_id",
                "chunk_index",
            ],
        )

        chunks: list[RetrievedChunk] = []
        best_chunk: RetrievedChunk | None = None
        best_score: float | None = None
        score_preview: list[float] = []

        for result in results:
            score = float(result.get("@search.score", 0.0))
            score_preview.append(round(score, 4))
            candidate = RetrievedChunk(
                text=str(result.get("text", "")),
                score=score,
                source_file=str(result.get("source_file", "unknown")),
                row_index=int(result.get("row_index", -1)),
                parent_id=str(result.get("parent_id", "")),
                chunk_index=int(result.get("chunk_index", -1)),
            )

            if best_score is None or score > best_score:
                best_score = score
                best_chunk = candidate

            if score >= self.min_score:
                chunks.append(candidate)

        if trace_callback:
            trace_callback(
                {
                    "stage": "retriever",
                    "event": "vector_search_result",
                    "provider": "azure_search",
                    "search_mode": self.search_mode,
                    "top_k": top_k,
                    "score_preview": score_preview[: min(8, len(score_preview))],
                    "strict_min_score": self.min_score,
                    "strict_pass_count": len(chunks),
                    "candidate_count": len(score_preview),
                }
            )

        if chunks:
            return chunks

        if (
            best_chunk is not None
            and best_score is not None
            and best_score >= self.fallback_min_score
        ):
            if trace_callback:
                trace_callback(
                    {
                        "stage": "retriever",
                        "event": "fallback_chunk_selected",
                        "fallback_min_score": self.fallback_min_score,
                        "best_score": round(best_score, 4),
                        "source_file": best_chunk.source_file,
                        "row_index": best_chunk.row_index,
                        "chunk_index": best_chunk.chunk_index,
                    }
                )
            return [best_chunk]

        if trace_callback:
            trace_callback(
                {
                    "stage": "retriever",
                    "event": "no_chunks_after_filtering",
                    "strict_min_score": self.min_score,
                    "fallback_min_score": self.fallback_min_score,
                }
            )

        return []