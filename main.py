import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
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

load_dotenv(dotenv_path=ENV_PATH)

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

READ365_SEARCH_API_URL = "https://read365.edunet.net/alpasq/api/search"
SCHOOL_PROV_CODE = os.getenv("SCHOOL_PROV_CODE", "F10").strip() or "F10"
SCHOOL_NEIS_CODE = os.getenv("SCHOOL_NEIS_CODE", "F100000120").strip() or "F100000120"
SCHOOL_NAME = os.getenv("SCHOOL_NAME", "광주소프트웨어마이스터고등학교").strip() or "광주소프트웨어마이스터고등학교"
READ365_MAX_PAGES_PER_KEYWORD = int(os.getenv("READ365_MAX_PAGES_PER_KEYWORD", "3").strip() or "3")
READ365_CANDIDATE_LIMIT = int(os.getenv("READ365_CANDIDATE_LIMIT", "80").strip() or "80")

app = FastAPI()


# ==============================
# 요청 데이터 형식
# ==============================

class UserRecommendRequest(BaseModel):
    user_id: str | None = None
    books: list[dict[str, Any]]
    top_k: int = 5


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
    """앱 입력값이나 독서로 응답을 추천 계산용 공통 형식으로 바꾼다."""

    return {
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
        "publisher": get_value(
            raw_book,
            "publisher", "PUBLISHER",
            "출판사", "발행자",
        ),
        "publish_year": get_value(
            raw_book,
            "publish_year", "publishYear", "pubYear", "PUB_YEAR",
            "출판년도", "발행년도", "발행년",
        ),
        "isbn": get_value(raw_book, "ISBN", "isbn", "isbn13"),
        "class_no": get_value(raw_book, "class_no", "classNo", "classNoText", "KDC", "kdc", "분류기호"),
        "call_no": get_value(raw_book, "call_no", "callNo", "청구기호"),
        "material_type": get_value(raw_book, "material_type", "materialType", "pubFormCodeDesc", "자료유형", "자료형태"),
        "register_no": get_value(raw_book, "register_no", "registerNo", "regNo", "등록번호"),
    }


def make_book_text(book: dict[str, str]) -> str:
    """Gemini 임베딩에 넣을 책 정보 텍스트를 만든다."""

    return f"""
제목: {book["title"]}
저자: {book["author"]}
출판사: {book["publisher"]}
출판년도: {book["publish_year"]}
ISBN: {book["isbn"]}
분류기호: {book["class_no"]}
청구기호: {book["call_no"]}
자료유형: {book["material_type"]}
""".strip()


# ==============================
# 독서로 학교 도서관 검색
# ==============================

def normalize_identity_text(value: str) -> str:
    """Book identity matching should ignore spacing and letter case."""

    return " ".join(value.casefold().split())


def normalize_title_text(value: str) -> str:
    """Create a strict title fingerprint for recommendation de-duplication."""

    normalized = html.unescape(value).casefold()
    normalized = re.sub(r"\([^)]*\)|\[[^\]]*\]|<[^>]*>", "", normalized)
    normalized = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
    return normalized


def get_book_identity_key(book: dict[str, str]) -> tuple[str, ...]:
    """Return a stable key for removing duplicate copies of the same book."""

    keys = get_book_identity_keys(book)
    return keys[0] if keys else ("unknown",)


def get_book_identity_keys(book: dict[str, str]) -> list[tuple[str, ...]]:
    """Return all stable keys that can identify the same book."""

    title = normalize_title_text(book.get("title", ""))
    isbn = normalize_identity_text(book.get("isbn", "").replace("-", ""))
    keys: list[tuple[str, ...]] = []

    if title:
        keys.append(("title", title))
    if isbn:
        keys.append(("isbn", isbn))

    return keys


def dedupe_books(books: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the first copy of each book."""

    unique_books: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()

    for book in books:
        keys = get_book_identity_keys(book)
        if any(key in seen for key in keys):
            continue

        seen.update(keys)
        unique_books.append(book)

    return unique_books


def make_book_text(book: dict[str, str]) -> str:
    """Build embedding text with genre and format weighted above author."""

    return f"""
title: {book["title"]}
topic_title: {book["title"]}
publisher: {book["publisher"]}
publish_year: {book["publish_year"]}
classification: {book["class_no"]}
genre_classification: {book["class_no"]}
material_type: {book["material_type"]}
book_format: {book["material_type"]}
""".strip()


def read365_school_search(search_keyword: str, page: int = 1) -> list[dict[str, Any]]:
    """독서로에서 현재 학교 도서관 소장 도서를 검색한다."""

    keyword = search_keyword.strip()
    if not keyword:
        return []

    payload = {
        "searchKeyword": keyword,
        "searchType": "",
        "provCode": SCHOOL_PROV_CODE,
        "neisCode": [SCHOOL_NEIS_CODE],
        "page": str(page),
        "display": "10",
    }

    request = urllib.request.Request(
        READ365_SEARCH_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json;charset=UTF-8",
            "Accept": "application/json",
            "Referer": "https://read365.edunet.net/High/",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))

    if result.get("status") != "OK":
        return []

    data = result.get("data") or {}
    books = data.get("bookList") or []
    return [book for book in books if isinstance(book, dict)]


def make_search_keywords(user_books: list[dict[str, str]]) -> list[str]:
    """사용자 책 목록과 넓은 주제 검색어를 합쳐 후보 수집용 검색어를 만든다."""

    keywords: list[str] = []

    for book in user_books:
        title = book.get("title", "").strip()
        author = book.get("author", "").strip()

        if title:
            keywords.append(title)
            keywords.extend([word for word in title.replace(":", " ").split() if len(word) >= 2])
        if author:
            keywords.append(author)

    # 독서로 검색 API는 빈 검색어로 전체 장서를 주지 않으므로,
    # 특정 분야에 치우치지 않게 넓은 주제어로 학교 도서관 후보를 모은다.
    keywords.extend([
        "소설",
        "문학",
        "시",
        "에세이",
        "인문",
        "철학",
        "역사",
        "사회",
        "경제",
        "정치",
        "심리",
        "과학",
        "수학",
        "예술",
        "음악",
        "미술",
        "진로",
        "자기계발",
        "공부",
        "여행",
        "건강",
        "환경",
        "프로그래밍",
        "알고리즘",
        "데이터",
        "인공지능",
        "컴퓨터",
        "책",
        "가",
    ])

    unique_keywords = []
    seen = set()
    for keyword in keywords:
        clean_keyword = keyword.strip()
        if clean_keyword and clean_keyword not in seen:
            seen.add(clean_keyword)
            unique_keywords.append(clean_keyword)

    return unique_keywords


def load_school_candidate_books(user_books: list[dict[str, str]]) -> list[dict[str, Any]]:
    """광주소프트웨어마이스터고등학교 도서관 안에 있는 책만 후보로 모은다."""

    candidates: list[dict[str, Any]] = []
    seen = set()

    for keyword in make_search_keywords(user_books):
        for page in range(1, READ365_MAX_PAGES_PER_KEYWORD + 1):
            for raw_book in read365_school_search(keyword, page):
                school_name = str(raw_book.get("schoolName", ""))
                neis_code = str(raw_book.get("neisCode", ""))

                if SCHOOL_NAME not in school_name or neis_code != SCHOOL_NEIS_CODE:
                    continue

                key = (
                    raw_book.get("isbn") or "",
                    raw_book.get("title") or "",
                    raw_book.get("author") or "",
                    raw_book.get("regNo") or "",
                )
                if key in seen:
                    continue

                seen.add(key)
                candidates.append(raw_book)

                if len(candidates) >= READ365_CANDIDATE_LIMIT:
                    return candidates

    return candidates


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
        raise RuntimeError(f"Gemini Embedding API 오류: {error.code} {error_body}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Gemini Embedding API 호출 실패: {error}") from error

    values = result.get("embedding", {}).get("values", [])
    if not values and result.get("embeddings"):
        values = result["embeddings"][0].get("values", [])
    if not values:
        raise RuntimeError(f"Gemini Embedding API 응답에 embedding 값이 없습니다: {result}")

    return [float(value) for value in values]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 Gemini 임베딩 벡터로 변환한다."""

    embeddings: list[list[float]] = []

    for text in texts:
        if not text.strip():
            continue

        for attempt in range(1, GEMINI_EMBEDDING_MAX_RETRIES + 1):
            try:
                embeddings.append(get_gemini_embedding(text))
                break
            except RuntimeError:
                if attempt >= GEMINI_EMBEDDING_MAX_RETRIES:
                    raise

                time.sleep(min(2 ** (attempt - 1), 8))

    return embeddings


# ==============================
# 추천 계산
# ==============================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터가 얼마나 비슷한지 계산한다."""

    vector_a = np.array(a)
    vector_b = np.array(b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


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


def material_similarity(a: str, b: str) -> float:
    """Compare book format/material type."""

    material_a = normalize_identity_text(a)
    material_b = normalize_identity_text(b)

    if not material_a or not material_b:
        return 0.0

    return 1.0 if material_a == material_b else 0.0


def metadata_similarity(candidate: dict[str, str], user_books: list[dict[str, str]]) -> float:
    """Use genre and format similarity so author alone does not dominate."""

    best_score = 0.0

    for user_book in user_books:
        score = (
            class_similarity(candidate.get("class_no", ""), user_book.get("class_no", "")) * 0.8
            + material_similarity(candidate.get("material_type", ""), user_book.get("material_type", "")) * 0.2
        )
        best_score = max(best_score, score)

    return best_score


def recommendation_score(
    vector_score: float,
    candidate: dict[str, str],
    user_books: list[dict[str, str]],
) -> float:
    """Blend semantic similarity with genre/format metadata."""

    meta_score = metadata_similarity(candidate, user_books)
    return vector_score * 0.75 + meta_score * 0.25


def is_already_read(candidate: dict[str, str], user_books: list[dict[str, str]]) -> bool:
    """사용자가 이미 입력한 책은 추천에서 제외한다."""

    candidate_keys = set(get_book_identity_keys(candidate))
    candidate_isbn = candidate.get("isbn", "")
    candidate_title = candidate.get("title", "")
    candidate_author = candidate.get("author", "")

    for user_book in user_books:
        if candidate_keys & set(get_book_identity_keys(user_book)):
            return True

        user_isbn = user_book.get("isbn", "")
        user_title = user_book.get("title", "")
        user_author = user_book.get("author", "")

        if candidate_isbn and user_isbn and candidate_isbn == user_isbn:
            return True

        if candidate_title == user_title and candidate_author == user_author:
            return True

    return False


def recommend_books_for_user(raw_user_books: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    """요청으로 받은 사용자별 책 목록을 기준으로 학교 도서관 책만 추천한다."""

    if not GEMINI_API_KEY:
        return {
            "message": "GEMINI_API_KEY가 설정되지 않았습니다.",
            "recommended_books": [],
        }

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

    raw_candidate_books = load_school_candidate_books(user_books)
    candidate_books = [
        normalize_book(book)
        for book in raw_candidate_books
    ]
    candidate_books = [
        book
        for book in candidate_books
        if book["title"] != ""
    ]
    candidate_books = dedupe_books(candidate_books)

    if not candidate_books:
        return {
            "message": f"{SCHOOL_NAME} 도서관 후보 책 목록을 가져오지 못했습니다.",
            "recommended_books": [],
        }

    user_book_vectors = get_embeddings([make_book_text(book) for book in user_books])
    candidate_book_vectors = get_embeddings([make_book_text(book) for book in candidate_books])
    user_preference_vector = np.mean(user_book_vectors, axis=0).tolist()

    recommendations = []
    recommended_keys: set[tuple[str, ...]] = set()
    for candidate, candidate_vector in zip(candidate_books, candidate_book_vectors):
        if is_already_read(candidate, user_books):
            continue

        candidate_keys = get_book_identity_keys(candidate)
        if any(key in recommended_keys for key in candidate_keys):
            continue

        vector_score = cosine_similarity(user_preference_vector, candidate_vector)
        score = recommendation_score(vector_score, candidate, user_books)
        recommendations.append({
            "title": candidate["title"],
            "author": candidate["author"],
            "publisher": candidate["publisher"],
            "publish_year": candidate["publish_year"],
            "isbn": candidate["isbn"],
            "class_no": candidate["class_no"],
            "call_no": candidate["call_no"],
            "material_type": candidate["material_type"],
            "score": round(score, 4),
            "vector_score": round(vector_score, 4),
            "metadata_score": round(metadata_similarity(candidate, user_books), 4),
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
        "candidate_book_count": len(candidate_books),
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
# CLI 테스트
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
        "publisher": "",
        "publish_year": "",
        "isbn": "",
        "class_no": "",
        "call_no": "",
        "material_type": "",
        "register_no": "",
    }


def collect_books_from_terminal() -> list[dict[str, str]]:
    """사용자가 터미널에서 여러 책을 입력하도록 한다."""

    print("책 추천 테스트 모드")
    print("책 형식: 제목|저자")
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

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

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
        print(f"임베딩 모델: Gemini {GEMINI_EMBEDDING_MODEL}")

        for index, book in enumerate(result.get("recommended_books", []), start=1):
            author = book["author"] or "미상"
            publisher = book["publisher"] or "미상"
            publish_year = book["publish_year"] or "미상"
            print(f"{index}. {book['title']} - {author} ({publisher}, {publish_year})")
            print(f"   유사도: {book['score']:.4f}")
    except Exception as error:
        print(f"추천 처리 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    run_cli()
