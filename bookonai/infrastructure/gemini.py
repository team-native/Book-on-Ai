"""Gemini embedding API integration and retry handling."""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from ..config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MAX_RETRIES,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_TIMEOUT_SECONDS,
)
from ..domain.models import GeminiQuotaError


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
        "content": {"parts": [{"text": text}]},
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
