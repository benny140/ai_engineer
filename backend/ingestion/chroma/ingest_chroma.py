import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
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
    persist_dir: Path
    collection_name: str


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    common = resolve_common_config(raw)
    return IngestConfig(
        persist_dir=resolve_repo_path(raw.get("persist_dir", "data/chroma_db")),
        collection_name=str(raw.get("collection_name", "rag_docs")),
        **common,
    )


def reset_db(persist_dir: Path) -> None:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def ingest_into_chroma(cfg: IngestConfig) -> int:
    reset_db(cfg.persist_dir)
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.embedding_model,
    )

    client = chromadb.PersistentClient(path=str(cfg.persist_dir))
    collection = client.create_collection(
        name=cfg.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    ingested = 0

    with cfg.output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            docs.append(row["text"])
            metas.append(
                {
                    "source_file": row["source_file"],
                    "row_index": int(row["row_index"]),
                    "fields_used": ",".join(row.get("fields_used", [])),
                    "parent_id": row["parent_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "chunk_count": int(row["chunk_count"]),
                }
            )

            if len(ids) >= cfg.batch_size:
                collection.add(ids=ids, documents=docs, metadatas=metas)
                ingested += len(ids)
                ids, docs, metas = [], [], []

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)
        ingested += len(ids)

    return ingested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect parquet sources and ingest into local ChromaDB"
    )
    parser.add_argument(
        "--config",
        default="backend/ingestion/chroma/ingest_config.json",
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
    args = parse_args()
    cfg = load_config(resolve_repo_path(args.config))

    report = inspect_sources(cfg)
    print_report(report)

    if args.mode == "inspect":
        print(f"\nProfile report written to {cfg.profile_report}")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)
    processed_records, chunks = build_jsonl(cfg, tokenizer)
    ingested = ingest_into_chroma(cfg)
    write_manifest(
        cfg,
        len(report["files"]),
        processed_records,
        chunks,
        ingested,
        provider="chroma",
        provider_target=cfg.collection_name,
    )

    print("\nIngestion complete")
    print("=" * 60)
    print(f"JSONL chunks: {cfg.output_jsonl}")
    print(f"Chroma persist dir: {cfg.persist_dir}")
    print(f"Collection: {cfg.collection_name}")
    print(f"Processed records: {processed_records}")
    print(f"Chunks written: {chunks}")
    print(f"Chunks ingested: {ingested}")


if __name__ == "__main__":
    main()