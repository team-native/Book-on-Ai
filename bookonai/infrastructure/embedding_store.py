"""Persistent school-book embedding cache management."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.books import (
    book_content_fingerprint,
    dedupe_books,
    get_book_identity_keys,
    make_book_text,
    make_embedding_cache_key,
    normalize_book,
    school_metadata,
)
from ..config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    SCHOOL_BOOK_SOURCE_COLUMNS,
    SCHOOL_BOOK_SOURCE_TABLE,
    SCHOOL_DB_PATH,
    SCHOOL_EMBEDDINGS_DIR,
    SCHOOL_EMBEDDINGS_METADATA_PATH,
    SCHOOL_EMBEDDINGS_NPY_PATH,
    SCHOOL_EMBEDDINGS_PROGRESS_PATH,
    SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY,
)
from .database import load_school_books_from_sqlite, summarize_school_books
from .gemini import get_embedding_with_retries
from ..domain.models import GeminiQuotaError, SchoolEmbeddingStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def embedding_record_from_book(book: dict[str, str], embedding_index: int) -> dict[str, Any]:
    return {
        "embedding_index": embedding_index,
        "identity_key": make_embedding_cache_key(book),
        "content_fingerprint": book_content_fingerprint(book),
        "book": book,
    }


def load_final_embedding_records(
    metadata_path: Path | None = None,
    npy_path: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load finalized metadata and vectors when they match the current model."""

    metadata_path = metadata_path or SCHOOL_EMBEDDINGS_METADATA_PATH
    npy_path = npy_path or SCHOOL_EMBEDDINGS_NPY_PATH
    if not metadata_path.exists() or not npy_path.exists():
        return [], np.empty((0, 0), dtype=np.float32)

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        vectors = np.load(npy_path).astype(np.float32, copy=False)
    except (OSError, json.JSONDecodeError, ValueError):
        return [], np.empty((0, 0), dtype=np.float32)

    if metadata.get("model") != GEMINI_EMBEDDING_MODEL:
        return [], np.empty((0, 0), dtype=np.float32)

    records = metadata.get("books", [])
    if not isinstance(records, list) or vectors.ndim != 2 or len(records) != vectors.shape[0]:
        return [], np.empty((0, 0), dtype=np.float32)
    return records, vectors


def load_partial_embedding_records(
    progress_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Load safely flushed in-progress embeddings so a failed run can resume."""

    progress_path = progress_path or SCHOOL_EMBEDDINGS_PROGRESS_PATH
    if not progress_path.exists():
        return [], []

    records: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    try:
        with progress_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                item = json.loads(line)
                if item.get("model") != GEMINI_EMBEDDING_MODEL:
                    continue

                record = item.get("record")
                embedding = item.get("embedding")
                if isinstance(record, dict) and isinstance(embedding, list) and embedding:
                    records.append(record)
                    vectors.append([float(value) for value in embedding])
    except (OSError, json.JSONDecodeError):
        return [], []
    return records, vectors


def append_partial_embedding(record: dict[str, Any], embedding: list[float]) -> None:
    SCHOOL_EMBEDDINGS_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "model": GEMINI_EMBEDDING_MODEL,
        "record": record,
        "embedding": embedding,
        "saved_at": utc_now_iso(),
    }
    with SCHOOL_EMBEDDINGS_PROGRESS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False))
        file.write("\n")


def merge_embedding_sources() -> tuple[list[dict[str, Any]], list[np.ndarray], int]:
    """Return final + partial records, de-duplicated by identity key."""

    final_records, final_vectors = load_final_embedding_records()
    partial_records, partial_vectors = load_partial_embedding_records()
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    seen: set[str] = set()

    for record, vector in zip(final_records, final_vectors):
        key = str(record.get("identity_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        record["embedding_index"] = len(records)
        records.append(record)
        vectors.append(np.asarray(vector, dtype=np.float32))

    partial_count = 0
    for record, vector in zip(partial_records, partial_vectors):
        key = str(record.get("identity_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        record["embedding_index"] = len(records)
        records.append(record)
        vectors.append(np.asarray(vector, dtype=np.float32))
        partial_count += 1
    return records, vectors, partial_count


def save_embedding_store(
    records: list[dict[str, Any]],
    vectors: list[np.ndarray],
    total_book_count: int,
    duplicate_copy_count: int,
) -> None:
    """Atomically save the canonical .npy vectors and metadata."""

    SCHOOL_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if vectors:
        matrix = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    for index, record in enumerate(records):
        record["embedding_index"] = index

    metadata = {
        "model": GEMINI_EMBEDDING_MODEL,
        **school_metadata(),
        "source_database": str(SCHOOL_DB_PATH),
        "source_table": SCHOOL_BOOK_SOURCE_TABLE,
        "source_columns": SCHOOL_BOOK_SOURCE_COLUMNS,
        "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
        "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
        "total_book_count": total_book_count,
        "unique_book_count": len(records),
        "duplicate_copy_count": duplicate_copy_count,
        "embedding_dtype": "float32",
        "embedding_shape": list(matrix.shape),
        "updated_at": utc_now_iso(),
        "books": records,
    }

    tmp_npy_path = SCHOOL_EMBEDDINGS_NPY_PATH.with_suffix(".npy.tmp")
    tmp_metadata_path = SCHOOL_EMBEDDINGS_METADATA_PATH.with_suffix(".json.tmp")
    with tmp_npy_path.open("wb") as file:
        np.save(file, matrix.astype(np.float32, copy=False))
    with tmp_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    tmp_npy_path.replace(SCHOOL_EMBEDDINGS_NPY_PATH)
    tmp_metadata_path.replace(SCHOOL_EMBEDDINGS_METADATA_PATH)
    if SCHOOL_EMBEDDINGS_PROGRESS_PATH.exists():
        SCHOOL_EMBEDDINGS_PROGRESS_PATH.unlink()


def build_school_embedding_store(
    max_new_embeddings: int | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Precompute embeddings for every unique school book, resuming if needed."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    source_books = load_school_books_from_sqlite()
    unique_books = dedupe_books(source_books)
    records, vectors, partial_count = merge_embedding_sources()
    record_by_key = {
        str(record.get("identity_key")): record
        for record in records
        if record.get("identity_key")
    }
    new_embedding_count = 0
    skipped_existing_count = 0
    stopped_reason = ""
    retry_after_seconds: int | None = None

    for book in unique_books:
        key = make_embedding_cache_key(book)
        if key in record_by_key:
            record_by_key[key]["book"] = book
            record_by_key[key]["content_fingerprint"] = book_content_fingerprint(book)
            skipped_existing_count += 1
            continue

        if max_new_embeddings is not None and new_embedding_count >= max_new_embeddings:
            stopped_reason = f"요청한 실행 제한 {max_new_embeddings}권에 도달했습니다."
            break

        try:
            embedding = get_embedding_with_retries(make_book_text(book))
        except GeminiQuotaError as error:
            stopped_reason = "Gemini quota/rate limit에 도달해서 여기까지 저장하고 종료합니다."
            retry_after_seconds = error.retry_after_seconds
            break
        except RuntimeError as error:
            stopped_reason = f"Gemini/API/network 오류가 반복되어 여기까지 저장하고 종료합니다: {error}"
            break

        vector = np.asarray(embedding, dtype=np.float32)
        record = embedding_record_from_book(book, len(records))
        append_partial_embedding(record, vector.tolist())
        record_by_key[key] = record
        records.append(record)
        vectors.append(vector)
        new_embedding_count += 1

        if new_embedding_count % SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY == 0:
            print(f"임베딩 진행: 신규 {new_embedding_count}권, 기존 {skipped_existing_count}권", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    save_embedding_store(
        records=records,
        vectors=vectors,
        total_book_count=len(source_books),
        duplicate_copy_count=len(source_books) - len(unique_books),
    )
    return {
        "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
        "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
        "progress_file": str(SCHOOL_EMBEDDINGS_PROGRESS_PATH),
        "model": GEMINI_EMBEDDING_MODEL,
        "source_table": SCHOOL_BOOK_SOURCE_TABLE,
        "total_book_count": len(source_books),
        "unique_book_count": len(unique_books),
        "duplicate_copy_count": len(source_books) - len(unique_books),
        "existing_embedding_count": skipped_existing_count,
        "partial_embedding_count_loaded": partial_count,
        "new_embedding_count": new_embedding_count,
        "stored_embedding_count": len(records),
        "remaining_embedding_count": max(len(unique_books) - len(records), 0),
        "stopped_reason": stopped_reason,
        "retry_after_seconds": retry_after_seconds,
    }


def build_identity_index(books: list[dict[str, str]]) -> dict[tuple[str, ...], list[int]]:
    index: dict[tuple[str, ...], list[int]] = {}
    for book_index, book in enumerate(books):
        for key in get_book_identity_keys(book):
            index.setdefault(key, []).append(book_index)
    return index


def load_school_embedding_store() -> SchoolEmbeddingStore:
    """Load finalized and partial precomputed school embeddings."""

    records, vector_list, _partial_count = merge_embedding_sources()
    if not records or not vector_list:
        return SchoolEmbeddingStore([], np.empty((0, 0), dtype=np.float32), {})

    vectors = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vector_list])
    books = [
        normalize_book(record.get("book", {}))
        for record in records
        if isinstance(record.get("book"), dict)
    ]
    if len(books) != vectors.shape[0]:
        return SchoolEmbeddingStore([], np.empty((0, 0), dtype=np.float32), {})

    return SchoolEmbeddingStore(
        books=books,
        vectors=vectors.astype(np.float32, copy=False),
        identity_index=build_identity_index(books),
    )


def get_embedding_store_status() -> dict[str, Any]:
    final_records, final_vectors = load_final_embedding_records()
    partial_records, _ = load_partial_embedding_records()
    try:
        source_summary = summarize_school_books()
    except FileNotFoundError:
        source_summary = {
            "source_table": SCHOOL_BOOK_SOURCE_TABLE,
            "total_book_count": 0,
            "unique_book_count": 0,
            "duplicate_copy_count": 0,
        }

    return {
        "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
        "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
        "progress_file": str(SCHOOL_EMBEDDINGS_PROGRESS_PATH),
        "embedding_file_exists": SCHOOL_EMBEDDINGS_NPY_PATH.exists(),
        "metadata_file_exists": SCHOOL_EMBEDDINGS_METADATA_PATH.exists(),
        "progress_file_exists": SCHOOL_EMBEDDINGS_PROGRESS_PATH.exists(),
        "model": GEMINI_EMBEDDING_MODEL,
        "stored_embedding_count": len(final_records),
        "partial_embedding_count": len(partial_records),
        "embedding_shape": list(final_vectors.shape),
        **source_summary,
    }
