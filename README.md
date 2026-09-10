# Book-on-Ai

학교 도서관의 전체 도서를 미리 임베딩하고, 사용자의 독서 기록과 벡터 유사도를 계산해 도서를 추천하는 FastAPI 서버입니다. 일반 추천 요청에서는 Gemini API를 호출하지 않습니다.

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

사전 임베딩을 실행할 때만 `.env`의 `GEMINI_API_KEY`에 실제 키를 설정합니다. `.env`는 Git에 포함되지 않습니다.

## 사전 임베딩

프로젝트 루트에 `book-on.sqlite`를 둔 뒤 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\precompute_school_embeddings.py --sleep 1
```

호출 수를 제한해서 나누어 실행할 수도 있습니다. 이미 완료한 책은 건너뛰므로 같은 명령을 다시 실행하면 이어서 처리합니다.

```powershell
.\.venv\Scripts\python.exe scripts\precompute_school_embeddings.py --limit 900 --sleep 1
```

상태 확인:

```powershell
.\.venv\Scripts\python.exe main.py --embedding-status
```

생성되는 파일은 다음과 같습니다. 파일 크기 때문에 `school_embeddings/`는 Git에 포함되지 않으며 배포 서버에 별도로 전달해야 합니다.

```text
school_embeddings/school_book_embeddings.npy
school_embeddings/school_book_metadata.json
```

## API 실행

```powershell
.\.venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000
```

- 상태 확인: `GET /health`
- API 문서: `GET /docs`
- 사용자 추천: `POST /recommend/user`

요청 예시:

```json
{
  "user_id": "user-123",
  "top_k": 5,
  "books": [
    {
      "isbn": "9788936434267",
      "title": "아몬드",
      "author": "손원평",
      "publisher": "창비",
      "category_name": "문학",
      "class_no": "813.7"
    }
  ]
}
```

책은 ISBN으로 우선 매칭하고, ISBN이 없으면 제목, 저자, 출판사를 사용합니다. 응답의 `recommended_books`에 유사도가 높은 학교 소장 도서가 반환됩니다.

## 배포 파일

추천 서버에는 애플리케이션 코드와 같은 시점에 생성된 `.npy`, metadata 파일이 필요합니다. 신규 도서를 임베딩하는 서버가 아니라면 `GEMINI_API_KEY`와 `book-on.sqlite`는 필요하지 않습니다.
