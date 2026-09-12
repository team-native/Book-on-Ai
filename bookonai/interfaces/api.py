"""FastAPI application and HTTP routes."""

from fastapi import FastAPI

from ..domain.models import UserRecommendRequest
from ..domain.recommendations import recommend_books_for_user


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend/user")
def recommend_for_user(request: UserRecommendRequest):
    """앱에서 사용자별 책 목록을 보내면 사용자별 추천 결과를 반환한다."""

    result = recommend_books_for_user(raw_user_books=request.books, top_k=request.top_k)
    result["user_id"] = request.user_id
    return result
