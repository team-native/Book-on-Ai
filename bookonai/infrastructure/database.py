"""Read-only access to the school library SQLite database."""

import sqlite3
from pathlib import Path
from typing import Any

from ..domain.books import dedupe_books, normalize_book
from ..config import SCHOOL_BOOK_SOURCE_COLUMNS, SCHOOL_BOOK_SOURCE_TABLE, SCHOOL_DB_PATH


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
