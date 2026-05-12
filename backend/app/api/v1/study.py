from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.core.deps import get_container
from app.schemas import (
    ChapterSummaryRequest,
    CatalogResponse,
    ChatRequest,
    ChatResponse,
    FlashcardRequest,
    FlashcardResponse,
    IngestRequest,
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
    QuizRequest,
    QuizResponse,
)

router = APIRouter()


@router.get("/catalog", response_model=CatalogResponse)
async def catalog(request: Request, class_num: int = 6) -> CatalogResponse:
    container = get_container(request)
    return await run_in_threadpool(container.catalog.get_catalog, class_num)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    container = get_container(request)
    return await run_in_threadpool(container.rag.answer_question, payload)


@router.post("/summary", response_model=ChatResponse)
async def chapter_summary(request: Request, payload: ChapterSummaryRequest) -> ChatResponse:
    container = get_container(request)
    return await run_in_threadpool(container.rag.summarize_chapter, payload.class_num, payload.subject, payload.chapter, payload.top_k)


@router.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(request: Request, payload: FlashcardRequest) -> FlashcardResponse:
    container = get_container(request)
    return await run_in_threadpool(container.rag.generate_flashcards, payload)


@router.post("/questions", response_model=QuestionResponse)
async def questions(request: Request, payload: QuestionRequest) -> QuestionResponse:
    container = get_container(request)
    return await run_in_threadpool(container.rag.generate_questions, payload)


@router.post("/quiz", response_model=QuizResponse)
async def quiz(request: Request, payload: QuizRequest) -> QuizResponse:
    container = get_container(request)
    # Override quiz_type to enforce MCQ-only functionality
    payload.quiz_type = "mcq"
    return await run_in_threadpool(container.rag.generate_quiz, payload)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, payload: IngestRequest) -> IngestResponse:
    container = get_container(request)
    return await run_in_threadpool(container.ingest.ingest_directory, payload)
