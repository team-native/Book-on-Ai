"""Shared request, store, and exception models for the domain layer."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel


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
