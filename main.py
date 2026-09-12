"""BookOn AI application entry point.

The implementation is organized under ``bookonai`` by responsibility. The
imports below intentionally re-export the established names so existing
commands such as ``from main import build_school_embedding_store`` continue to
work.
"""

from bookonai.interfaces.api import app, health_check, recommend_for_user
from bookonai.domain.books import (
    book_content_fingerprint,
    dedupe_books,
    get_book_identity_key,
    get_book_identity_keys,
    get_value,
    make_book_text,
    make_embedding_cache_key,
    normalize_book,
    normalize_identity_text,
    normalize_isbn,
    normalize_title_text,
    school_metadata,
)
from bookonai.config import (
    BASE_DIR,
    ENV_PATH,
    GEMINI_API_KEY,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_TIMEOUT_SECONDS,
    GEMINI_EMBEDDING_MAX_RETRIES,
    SCHOOL_NAME,
    SCHOOL_PROV_CODE,
    SCHOOL_NEIS_CODE,
    SCHOOL_DB_PATH,
    SCHOOL_EMBEDDINGS_DIR,
    SCHOOL_EMBEDDINGS_NPY_PATH,
    SCHOOL_EMBEDDINGS_METADATA_PATH,
    SCHOOL_EMBEDDINGS_PROGRESS_PATH,
    SCHOOL_EMBEDDING_PROGRESS_FLUSH_EVERY,
    SCHOOL_BOOK_SOURCE_TABLE,
    SCHOOL_BOOK_SOURCE_COLUMNS,
)
from bookonai.interfaces.cli import (
    collect_books_from_terminal,
    parse_book_input,
    print_precompute_result,
    run_cli,
    run_command_line,
)
from bookonai.infrastructure.database import (
    connect_school_db,
    get_sqlite_table_info,
    load_school_books_from_sqlite,
    summarize_school_books,
)
from bookonai.infrastructure.embedding_store import (
    append_partial_embedding,
    build_identity_index,
    build_school_embedding_store,
    embedding_record_from_book,
    get_embedding_store_status,
    load_final_embedding_records,
    load_partial_embedding_records,
    load_school_embedding_store,
    merge_embedding_sources,
    save_embedding_store,
    utc_now_iso,
)
from bookonai.infrastructure.gemini import (
    get_embedding_with_retries,
    get_gemini_embedding,
    parse_retry_after_seconds,
)
from bookonai.domain.models import GeminiQuotaError, SchoolEmbeddingStore, UserRecommendRequest
from bookonai.domain.recommendations import (
    category_similarity,
    class_similarity,
    cosine_similarity_matrix,
    find_user_book_vector_indexes,
    is_already_read,
    material_similarity,
    metadata_similarity,
    normalize_class_no,
    recommend_books_for_user,
)


if __name__ == "__main__":
    run_command_line()
