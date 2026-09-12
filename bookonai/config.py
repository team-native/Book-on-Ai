"""Application settings loaded from the project environment."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
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

GEMINI_EMBEDDING_TIMEOUT_SECONDS = int(
    os.getenv("GEMINI_EMBEDDING_TIMEOUT_SECONDS", "60").strip() or "60"
)
GEMINI_EMBEDDING_MAX_RETRIES = int(
    os.getenv("GEMINI_EMBEDDING_MAX_RETRIES", "3").strip() or "3"
)

SCHOOL_NAME = (
    os.getenv("SCHOOL_NAME", "광주소프트웨어마이스터고등학교").strip()
    or "광주소프트웨어마이스터고등학교"
)
SCHOOL_PROV_CODE = os.getenv("SCHOOL_PROV_CODE", "F10").strip() or "F10"
SCHOOL_NEIS_CODE = os.getenv("SCHOOL_NEIS_CODE", "F100000120").strip() or "F100000120"
SCHOOL_DB_PATH = BASE_DIR / (
    os.getenv("SCHOOL_DB_PATH", "book-on.sqlite").strip() or "book-on.sqlite"
)

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
    os.getenv("SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY", "25").strip() or "25"
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

