from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    app_name: str
    class_num: int
    documents_indexed: int
    ollama_base_url: str
    ollama_model: str
    generation_model: str
    embedding_model_name: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str


class SourceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    file_name: str
    class_num: int | None = None
    subject: str | None = None
    chapter: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    relevance_score: float | None = None
    excerpt: str | None = None
    topic: str | None = None
    exercise_section: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    question: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=12)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    not_in_textbook: bool = False
    retrieved_documents: int = 0
    source_scope: str | None = None


class ChapterSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)


class FlashcardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    focus_area: str | None = None
    count: int = Field(default=10, ge=5, le=20)
    top_k: int = Field(default=6, ge=1, le=12)


class FlashcardItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    front: str
    back: str
    explanation: str
    type: Literal["definition", "concept", "formula", "application", "misconception"] = "concept"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    blooms_level: Literal["remember", "understand", "apply"] = "understand"


class FlashcardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flashcards: list[FlashcardItem] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    not_in_textbook: bool = False


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    count: int = Field(default=10, ge=5, le=20)
    top_k: int = Field(default=6, ge=1, le=12)


class QuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    explanation: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    type: str = "conceptual"
    blooms_level: Literal["understand", "apply", "analyze"] = "understand"

class AssertionReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    count: int = Field(default=8, ge=4, le=15)
    top_k: int = Field(default=6, ge=1, le=12)


class AssertionReasonItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion: str
    reason: str
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str


class AssertionReasonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion_reason_questions: list[AssertionReasonItem] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    not_in_textbook: bool = False


class QuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionItem] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    not_in_textbook: bool = False


class QuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    quiz_type: Literal["mcq", "mixed", "true_false", "short_answer"] = "mcq"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    count: int = Field(default=10, ge=5, le=20)
    top_k: int = Field(default=6, ge=1, le=12)


class QuizOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    text: str


class QuizItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    question_type: Literal["mcq", "true_false", "short_answer"]
    options: list[QuizOption] = Field(default_factory=list)
    correct_answer: str
    explanation: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    blooms_level: Literal["remember", "understand", "apply", "analyze"] = "understand"


class QuizResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz_title: str
    questions: list[QuizItem] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    not_in_textbook: bool = False


class CatalogChapter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    file_path: str | None = None


class CatalogSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    chapter_count: int = 0
    chapters: list[CatalogChapter] = Field(default_factory=list)


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_num: int
    subjects: list[CatalogSubject] = Field(default_factory=list)
    source_dir: str
    note: str | None = None


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: str | None = None
    class_num: int = Field(default=6, ge=1, le=12)
    subject: str | None = None
    chapter: str | None = None
    clear_existing: bool = False


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: str
    files_processed: int
    chunks_indexed: int
    collection_size: int
    skipped_files: list[str] = Field(default_factory=list)
    note: str | None = None
