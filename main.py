import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel


# ==============================
# 기본 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, encoding="utf-8-sig")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_MODEL = (
    os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL).strip()
    or DEFAULT_GEMINI_EMBEDDING_MODEL
)
if GEMINI_EMBEDDING_MODEL in {"text-embedding-004", "models/text-embedding-004"}:
    GEMINI_EMBEDDING_MODEL = DEFAULT_GEMINI_EMBEDDING_MODEL
GEMINI_EMBEDDING_TIMEOUT_SECONDS = int(os.getenv("GEMINI_EMBEDDING_TIMEOUT_SECONDS", "60").strip() or "60")
GEMINI_EMBEDDING_MAX_RETRIES = int(os.getenv("GEMINI_EMBEDDING_MAX_RETRIES", "3").strip() or "3")

SCHOOL_NAME = os.getenv("SCHOOL_NAME", "광주소프트웨어마이스터고등학교").strip() or "광주소프트웨어마이스터고등학교"
SCHOOL_PROV_CODE = os.getenv("SCHOOL_PROV_CODE", "F10").strip() or "F10"
SCHOOL_NEIS_CODE = os.getenv("SCHOOL_NEIS_CODE", "F100000120").strip() or "F100000120"
SCHOOL_DB_PATH = BASE_DIR / (os.getenv("SCHOOL_DB_PATH", "book-on.sqlite").strip() or "book-on.sqlite")

SCHOOL_EMBEDDINGS_DIR = BASE_DIR / (
    os.getenv("SCHOOL_EMBEDDINGS_DIR", "school_embeddings").strip()
    or "school_embeddings"
)
SCHOOL_EMBEDDINGS_NPY_PATH = SCHOOL_EMBEDDINGS_DIR / (
    os.getenv("SCHOOL_EMBEDDINGS_NPY_FILENAME", "school_book_embeddings.npy").strip()
    or "school_book_embeddings.npy"
)
SCHOOL_EMBEDDINGS_METADATA_PATH = SCHOOL_EMBEDDINGS_DIR / (
    os.getenv("SCHOOL_EMBEDDINGS_METADATA_FILENAME", "school_book_metadata.json").strip()
    or "school_book_metadata.json"
)
SCHOOL_EMBEDDINGS_PROGRESS_PATH = SCHOOL_EMBEDDINGS_DIR / (
    os.getenv("SCHOOL_EMBEDDINGS_PROGRESS_FILENAME", "school_book_embeddings.partial.jsonl").strip()
    or "school_book_embeddings.partial.jsonl"
)
SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY = int(
    os.getenv("SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY", "25").strip()
    or "25"
)

SCHOOL_BOOK_SOURCE_TABLE = "dls_books"
SCHOOL_BOOK_SOURCE_COLUMNS = [
    "reg_code",
    "title",
    "author",
    "publisher",
    "pub_year",
    "isbn",
    "call_no",
    "class_no",
    "category_code",
    "category_name",
    "status",
    "deleted_at",
]

app = FastAPI()


# ==============================
# 요청 데이터 형식
# ==============================

class UserRecommendRequest(BaseModel):
    user_id: str | None = None
    books: list[dict[str, Any]]
    top_k: int = 5


@dataclass
class SchoolEmbeddingStore:
    books: list[dict[str, str]]
    vectors: np.ndarray
    identity_index: dict[tuple[str, ...], list[int]]


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini reports rate or quota exhaustion."""

    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


# ==============================
# 책 데이터 정리
# ==============================

def get_value(data: dict[str, Any], *keys: str, default: str = "") -> str:
    """여러 후보 필드명 중 값이 들어있는 첫 번째 값을 가져온다."""

    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip() != "":
            return html.unescape(str(value).strip())

    return default


def normalize_book(raw_book: dict[str, Any]) -> dict[str, str]:
    """앱 입력값이나 SQLite 행을 추천 계산용 공통 형식으로 바꾼다."""

    category_name = get_value(raw_book, "category_name", "categoryName", "category", "분류명")
    material_type = get_value(raw_book, "material_type", "materialType", "pubFormCodeDesc", "자료유형", "자료형태")

    return {
        "source_id": get_value(raw_book, "source_id", "reg_code", "regCode", "book_id", "id"),
        "title": get_value(
            raw_book,
            "title", "bookTitle", "book_title", "TITLE",
            "서명", "도서명", "자료명", "본서명", "책제목",
        ),
        "author": get_value(
            raw_book,
            "author", "authors", "writer", "AUTHOR",
            "저자", "저작자", "글쓴이",
        ),
        "publisher": get_value(raw_book, "publisher", "PUBLISHER", "출판사", "발행자"),
        "publish_year": get_value(
            raw_book,
            "publish_year", "publishYear", "pubYear", "pub_year", "PUB_YEAR",
            "출판년도", "발행년도", "발행년",
        ),
        "isbn": get_value(raw_book, "ISBN", "isbn", "isbn13"),
        "class_no": get_value(raw_book, "class_no", "classNo", "classNoText", "KDC", "kdc", "분류기호"),
        "call_no": get_value(raw_book, "call_no", "callNo", "청구기호"),
        "category_code": get_value(raw_book, "category_code", "categoryCode", "category_id", "분류코드"),
        "category_name": category_name,
        "material_type": material_type or category_name,
        "register_no": get_value(raw_book, "register_no", "registerNo", "regNo", "reg_code", "regCode"),
        "status": get_value(raw_book, "status", "loan_status", "loanStatus", "status_desc"),
    }


def make_book_text(book: dict[str, str]) -> str:
    """Gemini 임베딩에 넣을 책 정보 텍스트를 만든다."""

    return f"""
title: {book["title"]}
author: {book["author"]}
publisher: {book["publisher"]}
publish_year: {book["publish_year"]}
isbn: {book["isbn"]}
call_no: {book["call_no"]}
class_no: {book["class_no"]}
category_code: {book["category_code"]}
category_name: {book["category_name"]}
material_type: {book["material_type"]}
""".strip()


def normalize_identity_text(value: str) -> str:
    """Book identity matching should ignore spacing and letter case."""

    return " ".join(value.casefold().split())


def normalize_title_text(value: str) -> str:
    """Create a strict title fingerprint for recommendation de-duplication."""

    normalized = html.unescape(value).casefold()
    normalized = re.sub(r"\([^)]*\)|\[[^\]]*\]|<[^>]*>", "", normalized)
    normalized = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
    return normalized


def normalize_isbn(value: str) -> str:
    """Keep ISBN digits/X only so hyphen formatting does not matter."""

    return re.sub(r"[^0-9xX]", "", value).casefold()


def get_book_identity_key(book: dict[str, str]) -> tuple[str, ...]:
    """Return the primary stable key for one logical book."""

    keys = get_book_identity_keys(book)
    return keys[0] if keys else ("unknown",)


def get_book_identity_keys(book: dict[str, str]) -> list[tuple[str, ...]]:
    """Return stable keys that identify the same logical book."""

    isbn = normalize_isbn(book.get("isbn", ""))
    title = normalize_title_text(book.get("title", ""))
    author = normalize_identity_text(book.get("author", ""))
    publisher = normalize_identity_text(book.get("publisher", ""))

    if isbn:
        return [("isbn", isbn)]
    if title:
        return [("book", title, author, publisher)]

    register_no = normalize_identity_text(book.get("register_no", ""))
    if register_no:
        return [("register_no", register_no)]

    return []


def make_embedding_cache_key(book: dict[str, str]) -> str:
    """Return a string key for stored school book embeddings."""

    return "|".join(get_book_identity_key(book))


def book_content_fingerprint(book: dict[str, str]) -> str:
    """Hash the text used for embedding so stale vectors can be detected."""

    return hashlib.sha256(make_book_text(book).encode("utf-8")).hexdigest()


def dedupe_books(books: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the first copy of each logical book."""

    unique_books: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()

    for book in books:
        keys = get_book_identity_keys(book)
        if not keys or any(key in seen for key in keys):
            continue

        seen.update(keys)
        unique_books.append(book)

    return unique_books


def school_metadata() -> dict[str, str]:
    return {
        "school_name": SCHOOL_NAME,
        "school_prov_code": SCHOOL_PROV_CODE,
        "school_neis_code": SCHOOL_NEIS_CODE,
    }


# ==============================
# SQLite 학교 도서 읽기
# ==============================

def connect_school_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the BookOn SQLite database in read-only mode."""

    db_path = db_path or SCHOOL_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite 파일을 찾을 수 없습니다: {db_path}")

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def get_sqlite_table_info(db_path: Path | None = None) -> dict[str, Any]:
    """Return the actual source table/column summary used by embeddings."""

    connection = connect_school_db(db_path)
    try:
        columns = connection.execute(f"PRAGMA table_info({SCHOOL_BOOK_SOURCE_TABLE})").fetchall()
        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {SCHOOL_BOOK_SOURCE_TABLE}
            WHERE deleted_at IS NULL
              AND COALESCE(TRIM(title), '') != ''
            """
        ).fetchone()[0]
    finally:
        connection.close()

    return {
        "table": SCHOOL_BOOK_SOURCE_TABLE,
        "columns": [dict(column) for column in columns],
        "active_book_rows": int(row_count),
        "used_columns": SCHOOL_BOOK_SOURCE_COLUMNS,
    }


def load_school_books_from_sqlite(db_path: Path | None = None) -> list[dict[str, str]]:
    """Read all active school books from book-on.sqlite without modifying it."""

    select_columns = ", ".join(SCHOOL_BOOK_SOURCE_COLUMNS)
    query = f"""
        SELECT {select_columns}
        FROM {SCHOOL_BOOK_SOURCE_TABLE}
        WHERE deleted_at IS NULL
          AND COALESCE(TRIM(title), '') != ''
        ORDER BY title, author, publisher, reg_code
    """

    connection = connect_school_db(db_path)
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()

    books = []
    for row in rows:
        raw_book = dict(row)
        raw_book["source_id"] = raw_book.get("reg_code") or ""
        books.append(normalize_book(raw_book))

    return books


def summarize_school_books(db_path: Path | None = None) -> dict[str, Any]:
    """Return source row and logical book counts."""

    books = load_school_books_from_sqlite(db_path)
    unique_books = dedupe_books(books)
    return {
        "source_table": SCHOOL_BOOK_SOURCE_TABLE,
        "source_columns": SCHOOL_BOOK_SOURCE_COLUMNS,
        "total_book_count": len(books),
        "unique_book_count": len(unique_books),
        "duplicate_copy_count": len(books) - len(unique_books),
    }


# ==============================
# Gemini 임베딩
# ==============================

def get_gemini_embedding(text: str) -> list[float]:
    """Gemini Embedding API로 텍스트 하나를 벡터로 변환한다."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    model_path = GEMINI_EMBEDDING_MODEL
    if not model_path.startswith("models/"):
        model_path = f"models/{model_path}"

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{model_path}:embedContent"
        f"?key={urllib.parse.quote(GEMINI_API_KEY)}"
    )
    payload = {
        "model": model_path,
        "content": {
            "parts": [
                {"text": text}
            ]
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json;charset=UTF-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=GEMINI_EMBEDDING_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        if error.code == 429:
            raise GeminiQuotaError(
                f"Gemini Embedding API quota/rate limit 초과: {error_body}",
                retry_after_seconds=parse_retry_after_seconds(error_body),
            ) from error
        raise RuntimeError(f"Gemini Embedding API 오류: {error.code} {error_body}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Gemini Embedding API 호출 실패: {error}") from error

    values = result.get("embedding", {}).get("values", [])
    if not values and result.get("embeddings"):
        values = result["embeddings"][0].get("values", [])
    if not values:
        raise RuntimeError(f"Gemini Embedding API 응답에 embedding 값이 없습니다: {result}")

    return [float(value) for value in values]


def parse_retry_after_seconds(error_body: str) -> int | None:
    """Extract Gemini retry delay from a quota error body when present."""

    try:
        data = json.loads(error_body)
    except json.JSONDecodeError:
        data = {}

    details = data.get("error", {}).get("details", []) if isinstance(data, dict) else []
    for detail in details:
        if not isinstance(detail, dict):
            continue

        retry_delay = detail.get("retryDelay")
        if isinstance(retry_delay, str):
            match = re.match(r"^(\d+)(?:\.\d+)?s$", retry_delay.strip())
            if match:
                return int(match.group(1))

    match = re.search(r"retry in ([0-9.]+)s", error_body, flags=re.IGNORECASE)
    if match:
        return int(float(match.group(1)))

    return None


def get_embedding_with_retries(text: str) -> list[float]:
    """Embed one text with bounded retry/backoff for precompute jobs."""

    for attempt in range(1, GEMINI_EMBEDDING_MAX_RETRIES + 1):
        try:
            return get_gemini_embedding(text)
        except GeminiQuotaError:
            raise
        except RuntimeError as error:
            if attempt >= GEMINI_EMBEDDING_MAX_RETRIES:
                raise

            print(
                f"Gemini 임베딩 호출 실패, 재시도 {attempt}/{GEMINI_EMBEDDING_MAX_RETRIES}: {error}",
                flush=True,
            )
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("Gemini Embedding API 호출 재시도에 실패했습니다.")


# ==============================
# 사전 임베딩 저장소
# ==============================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def embedding_record_from_book(book: dict[str, str], embedding_index: int) -> dict[str, Any]:
    key = make_embedding_cache_key(book)
    return {
        "embedding_index": embedding_index,
        "identity_key": key,
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

    records, vector_list, partial_count = merge_embedding_sources()
    if not records or not vector_list:
        return SchoolEmbeddingStore(
            books=[],
            vectors=np.empty((0, 0), dtype=np.float32),
            identity_index={},
        )

    vectors = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vector_list])
    books = [
        normalize_book(record.get("book", {}))
        for record in records
        if isinstance(record.get("book"), dict)
    ]

    if len(books) != vectors.shape[0]:
        return SchoolEmbeddingStore(
            books=[],
            vectors=np.empty((0, 0), dtype=np.float32),
            identity_index={},
        )

    return SchoolEmbeddingStore(
        books=books,
        vectors=vectors.astype(np.float32, copy=False),
        identity_index=build_identity_index(books),
    )


def get_embedding_store_status() -> dict[str, Any]:
    final_records, final_vectors = load_final_embedding_records()
    partial_records, _ = load_partial_embedding_records()
    source_summary: dict[str, Any]
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


# ==============================
# 추천 계산
# ==============================

def cosine_similarity_matrix(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity against all school vectors with NumPy."""

    if vectors.size == 0 or query_vector.size == 0:
        return np.empty((0,), dtype=np.float32)

    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros((vectors.shape[0],), dtype=np.float32)

    denominator = vector_norms * query_norm
    scores = np.zeros((vectors.shape[0],), dtype=np.float32)
    valid = denominator > 0
    scores[valid] = vectors[valid] @ query_vector / denominator[valid]
    return scores.astype(np.float32, copy=False)


def normalize_class_no(class_no: str) -> str:
    """Keep only the meaningful KDC/DDC-like classification characters."""

    return "".join(char for char in class_no if char.isalnum())


def class_similarity(a: str, b: str) -> float:
    """Compare broad genre/topic by classification prefix."""

    class_a = normalize_class_no(a)
    class_b = normalize_class_no(b)

    if not class_a or not class_b:
        return 0.0
    if class_a == class_b:
        return 1.0
    if class_a[:3] == class_b[:3]:
        return 0.9
    if class_a[:2] == class_b[:2]:
        return 0.75
    if class_a[:1] == class_b[:1]:
        return 0.55

    return 0.0


def category_similarity(a: str, b: str) -> float:
    category_a = normalize_identity_text(a)
    category_b = normalize_identity_text(b)
    if not category_a or not category_b:
        return 0.0
    return 1.0 if category_a == category_b else 0.0


def material_similarity(a: str, b: str) -> float:
    """Compare book format/material type."""

    material_a = normalize_identity_text(a)
    material_b = normalize_identity_text(b)

    if not material_a or not material_b:
        return 0.0

    return 1.0 if material_a == material_b else 0.0


def metadata_similarity(candidate: dict[str, str], user_books: list[dict[str, str]]) -> float:
    """Use genre/category/format similarity as a small ranking signal."""

    best_score = 0.0

    for user_book in user_books:
        score = (
            class_similarity(candidate.get("class_no", ""), user_book.get("class_no", "")) * 0.45
            + category_similarity(candidate.get("category_name", ""), user_book.get("category_name", "")) * 0.35
            + material_similarity(candidate.get("material_type", ""), user_book.get("material_type", "")) * 0.20
        )
        best_score = max(best_score, score)

    return best_score


def is_already_read(candidate: dict[str, str], user_books: list[dict[str, str]]) -> bool:
    """사용자가 이미 입력한 책은 추천에서 제외한다."""

    candidate_keys = set(get_book_identity_keys(candidate))
    candidate_title = normalize_title_text(candidate.get("title", ""))
    candidate_author = normalize_identity_text(candidate.get("author", ""))

    for user_book in user_books:
        if candidate_keys & set(get_book_identity_keys(user_book)):
            return True

        user_title = normalize_title_text(user_book.get("title", ""))
        user_author = normalize_identity_text(user_book.get("author", ""))
        if candidate_title and candidate_title == user_title:
            if not candidate_author or not user_author or candidate_author == user_author:
                return True

    return False


def find_user_book_vector_indexes(
    user_books: list[dict[str, str]],
    store: SchoolEmbeddingStore,
) -> list[int]:
    """Find saved embeddings for the user's read books."""

    indexes: list[int] = []
    seen: set[int] = set()
    title_fallback: dict[str, list[int]] = {}

    for index, book in enumerate(store.books):
        title = normalize_title_text(book.get("title", ""))
        if title:
            title_fallback.setdefault(title, []).append(index)

    for user_book in user_books:
        matched_indexes: list[int] = []
        for key in get_book_identity_keys(user_book):
            matched_indexes.extend(store.identity_index.get(key, []))

        if not matched_indexes:
            matched_indexes.extend(title_fallback.get(normalize_title_text(user_book.get("title", "")), []))

        for index in matched_indexes:
            if index not in seen:
                seen.add(index)
                indexes.append(index)
                break

    return indexes


def recommend_books_for_user(raw_user_books: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    """요청으로 받은 사용자별 책 목록을 기준으로 학교 도서관 책만 추천한다."""

    user_books = [
        normalize_book(book)
        for book in raw_user_books
    ]
    user_books = [
        book
        for book in user_books
        if book["title"] != ""
    ]
    user_books = dedupe_books(user_books)

    if not user_books:
        return {
            "message": "사용자 책 목록이 비어 있습니다.",
            "recommended_books": [],
        }

    store = load_school_embedding_store()
    if not store.books:
        return {
            "message": "학교 도서 임베딩 파일이 없습니다. 먼저 사전 임베딩을 실행하세요.",
            "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
            "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
            "recommended_books": [],
        }

    user_vector_indexes = find_user_book_vector_indexes(user_books, store)
    if not user_vector_indexes:
        return {
            "message": "사용자가 읽은 책의 저장된 임베딩을 찾지 못했습니다.",
            "school_name": SCHOOL_NAME,
            "user_book_count": len(user_books),
            "candidate_book_count": len(store.books),
            "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
            "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
            "gemini_calls_per_recommendation": 0,
            "recommended_books": [],
        }

    user_book_vectors = store.vectors[user_vector_indexes]
    user_preference_vector = np.mean(user_book_vectors, axis=0).astype(np.float32, copy=False)
    vector_scores = cosine_similarity_matrix(store.vectors, user_preference_vector)

    recommendations = []
    recommended_keys: set[tuple[str, ...]] = set()
    for index, candidate in enumerate(store.books):
        if index in user_vector_indexes or is_already_read(candidate, user_books):
            continue

        candidate_keys = get_book_identity_keys(candidate)
        if any(key in recommended_keys for key in candidate_keys):
            continue

        vector_score = float(vector_scores[index])
        metadata_score = metadata_similarity(candidate, user_books)
        score = vector_score * 0.75 + metadata_score * 0.25
        recommendations.append({
            "title": candidate["title"],
            "author": candidate["author"],
            "publisher": candidate["publisher"],
            "publish_year": candidate["publish_year"],
            "isbn": candidate["isbn"],
            "class_no": candidate["class_no"],
            "call_no": candidate["call_no"],
            "material_type": candidate["material_type"],
            "category_code": candidate["category_code"],
            "category_name": candidate["category_name"],
            "score": round(score, 4),
            "vector_score": round(vector_score, 4),
            "metadata_score": round(metadata_score, 4),
        })
        recommended_keys.update(candidate_keys)

    recommendations.sort(
        key=lambda book: book["score"],
        reverse=True,
    )

    return {
        "school_name": SCHOOL_NAME,
        "school_prov_code": SCHOOL_PROV_CODE,
        "school_neis_code": SCHOOL_NEIS_CODE,
        "user_book_count": len(user_books),
        "user_embedding_match_count": len(user_vector_indexes),
        "candidate_book_count": len(store.books),
        "embedding_cache_path": str(SCHOOL_EMBEDDINGS_NPY_PATH),
        "embedding_file": str(SCHOOL_EMBEDDINGS_NPY_PATH),
        "metadata_file": str(SCHOOL_EMBEDDINGS_METADATA_PATH),
        "candidate_embedding_cache_hits": len(store.books),
        "candidate_embedding_cache_misses": 0,
        "gemini_candidate_calls_saved": len(store.books),
        "gemini_calls_per_recommendation": 0,
        "recommended_books": recommendations[:max(1, top_k)],
    }


# ==============================
# API
# ==============================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend/user")
def recommend_for_user(request: UserRecommendRequest):
    """앱에서 사용자별 책 목록을 보내면 사용자별 추천 결과를 반환한다."""

    result = recommend_books_for_user(
        raw_user_books=request.books,
        top_k=request.top_k,
    )
    result["user_id"] = request.user_id
    return result


# ==============================
# CLI
# ==============================

def parse_book_input(line: str) -> dict[str, str]:
    """터미널 입력 문자열을 책 정보 딕셔너리로 변환한다."""

    if "|" in line:
        parts = [part.strip() for part in line.split("|")]
    else:
        parts = [part.strip() for part in line.split(",")]

    return {
        "title": parts[0] if len(parts) > 0 else "",
        "author": parts[1] if len(parts) > 1 else "",
        "publisher": parts[2] if len(parts) > 2 else "",
        "publish_year": "",
        "isbn": "",
        "class_no": "",
        "call_no": "",
        "category_code": "",
        "category_name": "",
        "material_type": "",
        "register_no": "",
    }


def collect_books_from_terminal() -> list[dict[str, str]]:
    """사용자가 터미널에서 여러 책을 입력하도록 한다."""

    print("책 추천 테스트 모드")
    print("책 형식: 제목|저자|출판사")
    print("입력을 마치려면 'done' 또는 빈 줄을 입력하세요.")

    books: list[dict[str, str]] = []

    while True:
        line = input("책 입력 > ").strip()

        if not line or line.lower() in {"done", "exit", "quit", "q"}:
            break

        book = parse_book_input(line)

        if not book["title"]:
            print("제목이 비어 있어서 입력을 건너뜁니다.")
            continue

        books.append(book)

    return books


def run_cli() -> None:
    """터미널에서 입력한 책 취향을 기준으로 추천 결과를 출력한다."""

    books = collect_books_from_terminal()

    if not books:
        print("입력된 책이 없습니다. 추천을 생성할 수 없습니다.")
        return

    while True:
        try:
            top_k_input = input("추천 받을 책 수 (기본 5) > ").strip()
            top_k = int(top_k_input) if top_k_input else 5
            if top_k > 0:
                break
        except ValueError:
            print("숫자를 입력해 주세요.")

    try:
        result = recommend_books_for_user(books, top_k)

        print(f"\n{SCHOOL_NAME} 도서관 책 목록 기반 추천 결과")
        print("-" * 40)
        print(f"후보 책 수: {result.get('candidate_book_count', 0)}권")
        print(f"추천 1회당 Gemini 호출: {result.get('gemini_calls_per_recommendation', 0)}회")

        for index, book in enumerate(result.get("recommended_books", []), start=1):
            author = book["author"] or "미상"
            publisher = book["publisher"] or "미상"
            publish_year = book["publish_year"] or "미상"
            print(f"{index}. {book['title']} - {author} ({publisher}, {publish_year})")
            print(f"   유사도: {book['score']:.4f}")

        if not result.get("recommended_books"):
            print(result.get("message", "추천 결과가 없습니다."))
    except Exception as error:
        print(f"추천 처리 중 오류가 발생했습니다: {error}")


def print_precompute_result(result: dict[str, Any]) -> None:
    print("학교 도서 임베딩 사전 계산 완료")
    print(f"SQLite 테이블: {result['source_table']}")
    print(f"전체 책 수: {result['total_book_count']}")
    print(f"중복 제거 후 책 수: {result['unique_book_count']}")
    print(f"중복 복본 수: {result['duplicate_copy_count']}")
    print(f"기존 임베딩: {result['existing_embedding_count']}")
    print(f"새 임베딩: {result['new_embedding_count']}")
    print(f"저장된 임베딩: {result['stored_embedding_count']}")
    print(f"남은 임베딩: {result.get('remaining_embedding_count', 0)}")
    print(f"벡터 파일: {result['embedding_file']}")
    print(f"메타데이터 파일: {result['metadata_file']}")
    if result.get("stopped_reason"):
        print(f"중단 이유: {result['stopped_reason']}")
    if result.get("retry_after_seconds") is not None:
        print(f"권장 재시도 대기: {result['retry_after_seconds']}초")


if __name__ == "__main__":
    if "--precompute-school-embeddings" in sys.argv or "--build-cache" in sys.argv:
        try:
            print_precompute_result(build_school_embedding_store())
        except Exception as error:
            print(f"학교 도서 임베딩 생성 중 오류가 발생했습니다: {error}")
            sys.exit(1)
    elif "--embedding-status" in sys.argv:
        print(json.dumps(get_embedding_store_status(), ensure_ascii=False, indent=2))
    elif "--sqlite-summary" in sys.argv:
        print(json.dumps(summarize_school_books(), ensure_ascii=False, indent=2))
    else:
        run_cli()
