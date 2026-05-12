# NCERT Class 6 Intelligent Study Assistant Backend

This backend provides the AI layer for the Next.js frontend using FastAPI, LangChain, Chroma, Hugging Face embeddings, and local Ollama models.

## What it does

- Answers student questions using only official NCERT Class 6 textbook content.
- Generates flashcards, important Q&A, and quiz questions from the retrieved textbook context.
- Exposes a clean JSON API for the frontend.
- Supports local ingestion of NCERT PDFs or text files into a persisted Chroma vector store.

## Folder layout

- `backend/app/` contains the FastAPI application.
- `backend/data/ncert/` is where official textbook files should be placed.
- `backend/storage/chroma/` stores the persisted vector database.

## Expected content layout

Place files using a subject-first layout such as:

```text
backend/data/ncert/class-6/science/chapter-1.pdf
backend/data/ncert/class-6/mathematics/chapter-2.pdf
```

You can also use chapter folders:

```text
backend/data/ncert/class-6/science/food-where-does-it-come-from/notes.pdf
```

## Environment variables

Copy `backend/.env.example` to `backend/.env` and adjust values if needed.

## Run locally

Install dependencies from the `backend/` directory, then start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

## Ingest official NCERT files

After placing the official textbook files under `backend/data/ncert/`, run:

```bash
python -m app.scripts.ingest_ncert --class-num 6 --source-dir data/ncert/class-6
```

## API endpoints

- `GET /api/v1/health`
- `GET /api/v1/catalog?class_num=6`
- `POST /api/v1/chat`
- `POST /api/v1/flashcards`
- `POST /api/v1/questions`
- `POST /api/v1/quiz`
- `POST /api/v1/ingest`

## Notes

- The backend is strict: if the official NCERT corpus does not contain the answer, it says so instead of inventing facts.
- Ollama must be running locally and the selected model must be available in the Ollama registry.
