"""Interactive CLI and command-line entry points."""

import json
import sys
from typing import Any

from ..config import SCHOOL_NAME
from ..infrastructure.database import summarize_school_books
from ..infrastructure.embedding_store import build_school_embedding_store, get_embedding_store_status
from ..domain.recommendations import recommend_books_for_user


def parse_book_input(line: str) -> dict[str, str]:
    """터미널 입력 문자열을 책 정보 딕셔너리로 변환한다."""

    parts = [part.strip() for part in (line.split("|") if "|" in line else line.split(","))]
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


def run_command_line(argv: list[str] | None = None) -> None:
    """Run the same command-line modes historically exposed by main.py."""

    argv = sys.argv[1:] if argv is None else argv
    if "--precompute-school-embeddings" in argv or "--build-cache" in argv:
        try:
            print_precompute_result(build_school_embedding_store())
        except Exception as error:
            print(f"학교 도서 임베딩 생성 중 오류가 발생했습니다: {error}")
            raise SystemExit(1) from error
    elif "--embedding-status" in argv:
        print(json.dumps(get_embedding_store_status(), ensure_ascii=False, indent=2))
    elif "--sqlite-summary" in argv:
        print(json.dumps(summarize_school_books(), ensure_ascii=False, indent=2))
    else:
        run_cli()
