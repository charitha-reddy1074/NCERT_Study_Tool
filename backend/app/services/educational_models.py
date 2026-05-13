from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ContentType = Literal[
    "definition",
    "concept_explanation",
    "formula",
    "theorem",
    "example",
    "activity",
    "solved_problem",
    "exercise",
    "fact",
    "procedure",
    "experiment",
    "diagram_explanation",
    "summary",
    "important_note",
]

BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate"]
DifficultyLevel = Literal["easy", "medium", "hard"]


@dataclass(slots=True)
class ParsedSection:
    source_path: Path
    source_name: str
    page_no: int
    block_index: int
    heading_path: list[str]
    level: int
    text: str
    source_kind: Literal["pdf", "text"]
    font_size: float | None = None
    is_heading: bool = False

    @property
    def section_path(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class ClassificationResult:
    content_type: ContentType
    blooms_level: BloomLevel
    difficulty: DifficultyLevel
    question_potential: list[str] = field(default_factory=list)
    is_conceptual: bool = True
    is_example_only: bool = False
    example_type: str | None = None


@dataclass(slots=True)
class ConceptProfile:
    primary_concept: str
    secondary_concepts: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    prerequisite_concepts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    abstract_concepts: list[str] = field(default_factory=list)
    formulae: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EnrichmentProfile:
    classification: ClassificationResult
    concepts: ConceptProfile
    misconceptions: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalyzedSection:
    section: ParsedSection
    enrichment: EnrichmentProfile


@dataclass(slots=True)
class ChunkDraft:
    text: str
    sections: list[ParsedSection]
    enrichment: EnrichmentProfile
    heading: str
    subheading: str
    page_nos: list[int]


@dataclass(slots=True)
class PreparedDocument:
    text: str
    metadata: dict[str, object]
