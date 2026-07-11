import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from tqdm import tqdm
from transformers import AutoTokenizer

from backend.paths import resolve_repo_path


@dataclass
class IngestConfig:
    input_dir: Path
    output_jsonl: Path
    profile_report: Path
    persist_dir: Path
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    preview_rows: int
    column_overrides: dict[str, list[str]]


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return IngestConfig(
        input_dir=resolve_repo_path(raw.get("input_dir", "data/raw")),
        output_jsonl=resolve_repo_path(
            raw.get("output_jsonl", "data/processed/chunks.jsonl")
        ),
        profile_report=resolve_repo_path(
            raw.get("profile_report", "data/processed/source_profile.json")
        ),
        persist_dir=resolve_repo_path(raw.get("persist_dir", "data/chroma_db")),
        collection_name=raw.get("collection_name", "rag_docs"),
        embedding_model=raw.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(raw.get("chunk_size", 400)),
        chunk_overlap=int(raw.get("chunk_overlap", 60)),
        batch_size=int(raw.get("batch_size", 256)),
        preview_rows=int(raw.get("preview_rows", 3)),
        column_overrides=raw.get("column_overrides", {}),
    )


def parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def pick_columns(
    df: pd.DataFrame, file_name: str, overrides: dict[str, list[str]]
) -> list[str]:
    if file_name in overrides:
        wanted = [c for c in overrides[file_name] if c in df.columns]
        if wanted:
            return wanted

    text_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            non_empty = series.dropna().astype(str).str.strip()
            if (non_empty != "").any():
                text_cols.append(col)
    return text_cols


def normalize_text(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return " ".join(json.dumps(value, ensure_ascii=False).split())
    text = str(value)
    return " ".join(text.split())


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def compose_row_text(
    row: pd.Series, selected_columns: list[str]
) -> tuple[str, list[str]]:
    parts: list[str] = []
    fields_used: list[str] = []

    for col in selected_columns:
        value = row.get(col)
        if is_missing(value):
            continue
        cleaned = normalize_text(value)
        if not cleaned:
            continue
        parts.append(f"{col}: {cleaned}")
        fields_used.append(col)

    return "\n".join(parts), fields_used


def chunk_text_by_tokens(
    tokenizer: AutoTokenizer, text: str, chunk_size: int, overlap: int
) -> list[str]:
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if not input_ids:
        return []

    chunks: list[str] = []
    stride = max(1, chunk_size - overlap)

    for start in range(0, len(input_ids), stride):
        end = start + chunk_size
        token_chunk = input_ids[start:end]
        if not token_chunk:
            continue
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(input_ids):
            break

    return chunks


def make_id(source_file: str, row_index: int, chunk_index: int, text: str) -> str:
    payload = f"{source_file}|{row_index}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def inspect_sources(cfg: IngestConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "input_dir": str(cfg.input_dir),
        "files": [],
    }

    for file_path in parquet_files(cfg.input_dir):
        df = pd.read_parquet(file_path)
        selected = pick_columns(df, file_path.name, cfg.column_overrides)

        sample = df.head(cfg.preview_rows).fillna("")
        sample_records = json.loads(
            sample.to_json(orient="records", force_ascii=False, date_format="iso")
        )

        info = {
            "file": file_path.name,
            "rows": int(len(df)),
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "selected_columns": selected,
            "sample_rows": sample_records,
        }
        report["files"].append(info)

    cfg.profile_report.parent.mkdir(parents=True, exist_ok=True)
    with cfg.profile_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def build_jsonl(cfg: IngestConfig, tokenizer: AutoTokenizer) -> tuple[int, int]:
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_chunks = 0

    with cfg.output_jsonl.open("w", encoding="utf-8") as out_f:
        for file_path in parquet_files(cfg.input_dir):
            df = pd.read_parquet(file_path)
            selected = pick_columns(df, file_path.name, cfg.column_overrides)

            if not selected:
                continue

            for row_index, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Compose {file_path.name}"
            ):
                text, fields_used = compose_row_text(row, selected)
                if not text:
                    continue

                chunks = chunk_text_by_tokens(
                    tokenizer, text, cfg.chunk_size, cfg.chunk_overlap
                )
                if not chunks:
                    continue

                parent_id = hashlib.sha1(
                    f"{file_path.name}|{row_index}".encode("utf-8")
                ).hexdigest()
                chunk_count = len(chunks)

                for chunk_index, chunk in enumerate(chunks):
                    rec_id = make_id(file_path.name, row_index, chunk_index, chunk)
                    line = {
                        "id": rec_id,
                        "text": chunk,
                        "source_file": file_path.name,
                        "row_index": int(row_index),
                        "fields_used": fields_used,
                        "parent_id": parent_id,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    }
                    out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    total_chunks += 1

                total_records += 1

    return total_records, total_chunks


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


def print_report(report: dict[str, Any]) -> None:
    print("\nSource inspection summary")
    print("=" * 60)
    for file_info in report["files"]:
        print(f"File: {file_info['file']}")
        print(f"Rows: {file_info['rows']}")
        print("Columns:")
        for col in file_info["columns"]:
            print(f"  - {col['name']} ({col['dtype']})")
        print(
            f"Selected columns: {', '.join(file_info['selected_columns']) or '(none)'}"
        )
        print("-" * 60)


def write_manifest(
    cfg: IngestConfig,
    profiled_files: int,
    processed_records: int,
    chunks: int,
    ingested: int,
) -> None:
    manifest = {
        "input_dir": str(cfg.input_dir),
        "output_jsonl": str(cfg.output_jsonl),
        "profile_report": str(cfg.profile_report),
        "persist_dir": str(cfg.persist_dir),
        "collection_name": cfg.collection_name,
        "embedding_model": cfg.embedding_model,
        "profiled_files": profiled_files,
        "processed_records": processed_records,
        "chunks_written": chunks,
        "chunks_ingested": ingested,
    }
    manifest_path = cfg.output_jsonl.parent / "ingest_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect parquet sources and ingest into local ChromaDB"
    )
    parser.add_argument(
        "--config",
        default="backend/ingestion/ingest_config.json",
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
    write_manifest(cfg, len(report["files"]), processed_records, chunks, ingested)

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
