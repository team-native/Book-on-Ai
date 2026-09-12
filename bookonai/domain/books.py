"""Book normalization, identity matching, and de-duplication helpers."""

import hashlib
import html
import re
from typing import Any

from ..config import SCHOOL_NAME, SCHOOL_NEIS_CODE, SCHOOL_PROV_CODE


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
    material_type = get_value(
        raw_book, "material_type", "materialType", "pubFormCodeDesc", "자료유형", "자료형태"
    )

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
