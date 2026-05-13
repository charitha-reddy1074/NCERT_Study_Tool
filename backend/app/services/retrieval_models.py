from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document


IntentType = Literal["flashcards", "mcq", "hots", "summary", "conceptual_qa"]


@dataclass(slots=True)
class RetrievalIntent:
    intent_type: IntentType
    include_content_types: list[str] = field(default_factory=list)
    exclude_content_types: list[str] = field(default_factory=list)
    blooms_levels: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(slots=True)
class QueryPlan:
    original_query: str
    transformed_query: str
    bm25_query: str
    intent: RetrievalIntent
    metadata_filter: dict[str, object]
    expand_terms: list[str] = field(default_factory=list)
    suppress_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedCandidate:
    document: Document
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    mmr_score: float = 0.0

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self.document.metadata or {})

    @property
    def text(self) -> str:
        return self.document.page_content or ""


@dataclass(slots=True)
class StructuredEducationalContext:
    concepts: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    formulae: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)

