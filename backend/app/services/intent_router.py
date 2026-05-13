from __future__ import annotations

from app.core.utils import normalize_key
from app.services.retrieval_models import RetrievalIntent


class IntentRouter:
    def detect(self, *, task_type: str, query: str, quiz_type: str | None = None) -> RetrievalIntent:
        task = normalize_key(task_type)
        lowered = query.strip().lower()

        if task in {"flashcards", "flashcard"}:
            return RetrievalIntent(
                intent_type="flashcards",
                include_content_types=["definition", "formula", "concept_explanation", "summary", "important_note", "theorem"],
                exclude_content_types=["example"],
                blooms_levels=["remember", "understand", "apply"],
                description="Retrieve compact concept-first material for flashcard generation.",
            )

        if task in {"summary", "chapter_summary"}:
            return RetrievalIntent(
                intent_type="summary",
                include_content_types=["summary", "definition", "concept_explanation", "formula", "important_note", "theorem"],
                exclude_content_types=["example"],
                blooms_levels=["remember", "understand", "analyze"],
                description="Retrieve chapter ideas suitable for a curriculum-aligned summary.",
            )

        if task in {"question", "questions", "conceptual_qa"}:
            return RetrievalIntent(
                intent_type="conceptual_qa",
                include_content_types=["definition", "concept_explanation", "formula", "theorem", "important_note", "solved_problem"],
                exclude_content_types=["example"],
                blooms_levels=["understand", "apply", "analyze"],
                description="Retrieve concepts and explanations for conceptual question answering.",
            )

        if task in {"quiz", "mcq"} or quiz_type:
            quiz = normalize_key(quiz_type or "mcq")
            if quiz in {"short-answer", "hots", "higher-order", "shortanswer"}:
                return RetrievalIntent(
                    intent_type="hots",
                    include_content_types=["concept_explanation", "exercise", "solved_problem", "theorem", "important_note", "procedure"],
                    exclude_content_types=["example"],
                    blooms_levels=["apply", "analyze", "evaluate"],
                    description="Retrieve deeper material for short-answer or HOTS-style questions.",
                )
            return RetrievalIntent(
                intent_type="mcq",
                include_content_types=["definition", "concept_explanation", "formula", "theorem", "solved_problem", "important_note"],
                exclude_content_types=["example"],
                blooms_levels=["remember", "understand", "apply"],
                description="Retrieve concept-rich chunks for MCQ generation.",
            )

        if any(token in lowered for token in ["why", "how", "justify", "analyze", "evaluate", "compare"]):
            return RetrievalIntent(
                intent_type="hots",
                include_content_types=["concept_explanation", "exercise", "solved_problem", "theorem", "important_note", "procedure", "experiment"],
                exclude_content_types=["example"],
                blooms_levels=["apply", "analyze", "evaluate"],
                description="Retrieve higher-order thinking material for analytical prompts.",
            )

        return RetrievalIntent(
            intent_type="conceptual_qa",
            include_content_types=["definition", "concept_explanation", "formula", "theorem", "important_note", "solved_problem"],
            exclude_content_types=["example"],
            blooms_levels=["understand", "apply"],
            description="Default to concept-centric educational retrieval.",
        )

