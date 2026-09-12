"""Recommendation scoring and candidate filtering."""

from typing import Any

import numpy as np

from .books import (
    dedupe_books,
    get_book_identity_keys,
    normalize_book,
    normalize_identity_text,
    normalize_title_text,
)
from ..config import (
    SCHOOL_EMBEDDINGS_METADATA_PATH,
    SCHOOL_EMBEDDINGS_NPY_PATH,
    SCHOOL_NAME,
    SCHOOL_NEIS_CODE,
    SCHOOL_PROV_CODE,
)
from ..infrastructure.embedding_store import load_school_embedding_store
from .models import SchoolEmbeddingStore


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

    normalized_books = [normalize_book(book) for book in raw_user_books]
    user_books = dedupe_books([book for book in normalized_books if book["title"] != ""])

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

    recommendations.sort(key=lambda book: book["score"], reverse=True)
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
