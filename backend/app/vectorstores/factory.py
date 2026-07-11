from backend.app.vectorstores.azure_search_store import AzureSearchVectorStore
from backend.app.vectorstores.base import VectorStore
from backend.app.vectorstores.chroma_store import ChromaVectorStore


def create_vector_store(
    provider: str,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    max_distance: float,
    fallback_max_distance: float,
    azure_search_endpoint: str = "",
    azure_search_index_name: str = "",
    azure_search_api_key_env: str = "AZURE_SEARCH_API_KEY",
    azure_search_mode: str = "hybrid",
    azure_search_min_score: float = 0.0,
    azure_search_fallback_min_score: float = 0.0,
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
    if normalized == "azure_search":
        return AzureSearchVectorStore(
            endpoint=azure_search_endpoint,
            index_name=azure_search_index_name,
            api_key_env=azure_search_api_key_env,
            embedding_model=embedding_model,
            search_mode=azure_search_mode,
            min_score=azure_search_min_score,
            fallback_min_score=azure_search_fallback_min_score,
        )
    raise ValueError(
        "Unsupported vector store provider: "
        f"{provider}. Supported: chroma, azure_search"
    )
