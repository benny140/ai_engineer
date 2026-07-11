from backend.app.vectorstores.base import VectorStore
from backend.app.vectorstores.chroma_store import ChromaVectorStore


def create_vector_store(
    provider: str,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    max_distance: float,
    fallback_max_distance: float,
) -> VectorStore:
    normalized = provider.strip().lower()
    if normalized == "chroma":
        return ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            max_distance=max_distance,
            fallback_max_distance=fallback_max_distance,
        )
    raise ValueError(
        f"Unsupported vector store provider: {provider}. Supported: chroma"
    )
