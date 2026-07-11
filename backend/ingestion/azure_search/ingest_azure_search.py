import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from backend.ingestion.common import (
    BaseIngestConfig,
    build_jsonl,
    inspect_sources,
    print_report,
    resolve_common_config,
    write_manifest,
)
from backend.paths import resolve_repo_path


@dataclass
class IngestConfig(BaseIngestConfig):
    index_name: str
    vector_dimensions: int


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    common = resolve_common_config(raw)
    return IngestConfig(
        index_name=str(raw.get("index_name", "rag-docs")),
        vector_dimensions=int(raw.get("vector_dimensions", 384)),
        **common,
    )


def require_env_var(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_index(index_client: SearchIndexClient, cfg: IngestConfig) -> None:
    try:
        index_client.get_index(cfg.index_name)
        index_client.delete_index(cfg.index_name)
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="row_index", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="fields_used", type=SearchFieldDataType.String),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_count", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=cfg.vector_dimensions,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw",
            )
        ],
    )

    index = SearchIndex(name=cfg.index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)


def ingest_into_azure_search(
    cfg: IngestConfig,
    endpoint: str,
    api_key: str,
) -> int:
    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    build_index(index_client, cfg)

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=cfg.index_name,
        credential=credential,
    )
    embedding_model = SentenceTransformer(cfg.embedding_model)

    documents: list[dict] = []
    ingested = 0

    with cfg.output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            vector = embedding_model.encode(row["text"]).tolist()
            documents.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "source_file": row["source_file"],
                    "row_index": int(row["row_index"]),
                    "fields_used": ",".join(row.get("fields_used", [])),
                    "parent_id": row["parent_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_count": int(row["chunk_count"]),
                    "text_vector": vector,
                }
            )

            if len(documents) >= cfg.batch_size:
                search_client.upload_documents(documents=documents)
                ingested += len(documents)
                documents = []

    if documents:
        search_client.upload_documents(documents=documents)
        ingested += len(documents)

    return ingested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect parquet sources and ingest into Azure AI Search"
    )
    parser.add_argument(
        "--config",
        default="backend/ingestion/azure_search/ingest_azure_search_config.json",
        help="Path to ingestion config JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "ingest"],
        default="ingest",
        help="inspect: profile parquet files; ingest: inspect + transform + full rebuild + ingest",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(resolve_repo_path(".env"))

    args = parse_args()
    cfg = load_config(resolve_repo_path(args.config))

    report = inspect_sources(cfg)
    print_report(report)

    if args.mode == "inspect":
        print(f"\nProfile report written to {cfg.profile_report}")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)
    processed_records, chunks = build_jsonl(cfg, tokenizer)

    endpoint = require_env_var("AZURE_SEARCH_ENDPOINT")
    api_key = require_env_var("AZURE_SEARCH_API_KEY")
    ingested = ingest_into_azure_search(cfg, endpoint=endpoint, api_key=api_key)

    write_manifest(
        cfg,
        len(report["files"]),
        processed_records,
        chunks,
        ingested,
        provider="azure_search",
        provider_target=cfg.index_name,
    )

    print("\nIngestion complete")
    print("=" * 60)
    print(f"JSONL chunks: {cfg.output_jsonl}")
    print(f"Search endpoint: {endpoint}")
    print(f"Index: {cfg.index_name}")
    print(f"Processed records: {processed_records}")
    print(f"Chunks written: {chunks}")
    print(f"Chunks ingested: {ingested}")


if __name__ == "__main__":
    main()