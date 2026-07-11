import chromadb
from chromadb.utils import embedding_functions
from typing import Any, Callable

from backend.app.vectorstores.base import RetrievedChunk
from backend.paths import resolve_repo_path


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_model: str,
        max_distance: float,
        fallback_max_distance: float,
    ):
        self.max_distance = max_distance
        self.fallback_max_distance = fallback_max_distance
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        client = chromadb.PersistentClient(path=str(resolve_repo_path(persist_dir)))
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )

    def retrieve(
        self,
        query_text: str,
        top_k: int,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        best_chunk: RetrievedChunk | None = None
        best_distance: float | None = None
        strict_pass_count = 0
        for doc, meta, distance in zip(docs, metas, distances):
            meta = meta or {}
            dist = float(distance)
            candidate = RetrievedChunk(
                text=str(doc),
                score=1.0 - dist,
                source_file=str(meta.get("source_file", "unknown")),
                row_index=int(meta.get("row_index", -1)),
                parent_id=str(meta.get("parent_id", "")),
                chunk_index=int(meta.get("chunk_index", -1)),
            )
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_chunk = candidate
            if dist > self.max_distance:
                continue
            chunks.append(candidate)
            strict_pass_count += 1

        if trace_callback:
            preview = [round(float(d), 4) for d in distances[: min(8, len(distances))]]
            trace_callback(
                {
                    "stage": "retriever",
                    "event": "vector_search_result",
                    "provider": "chroma",
                    "top_k": top_k,
                    "distance_preview": preview,
                    "strict_max_distance": self.max_distance,
                    "strict_pass_count": strict_pass_count,
                    "candidate_count": len(distances),
                }
            )

        if chunks:
            return chunks

        if (
            best_chunk is not None
            and best_distance is not None
            and best_distance <= self.fallback_max_distance
        ):
            if trace_callback:
                trace_callback(
                    {
                        "stage": "retriever",
                        "event": "fallback_chunk_selected",
                        "fallback_max_distance": self.fallback_max_distance,
                        "best_distance": round(best_distance, 4),
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
                    "strict_max_distance": self.max_distance,
                    "fallback_max_distance": self.fallback_max_distance,
                }
            )

        return chunks
