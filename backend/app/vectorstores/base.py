from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str
    row_index: int
    parent_id: str
    chunk_index: int


class VectorStore(Protocol):
    def retrieve(
        self,
        query_text: str,
        top_k: int,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[RetrievedChunk]: ...
