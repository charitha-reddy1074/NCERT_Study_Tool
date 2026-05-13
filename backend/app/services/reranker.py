from __future__ import annotations

from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.services.retrieval_models import QueryPlan, RetrievedCandidate


class EducationalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def rerank(self, plan: QueryPlan, candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
        if not candidates:
            return []

        if self._model is None:
            self._model = self._load_model()

        if self._model is not None:
            pairs = [(plan.transformed_query, self._candidate_text(candidate)) for candidate in candidates]
            scores = self._model.predict(pairs)
            reranked: list[RetrievedCandidate] = []
            for candidate, score in zip(candidates, scores):
                reranked.append(replace(candidate, rerank_score=float(score) + self._intent_boost(candidate, plan)))
            reranked.sort(key=lambda item: item.rerank_score, reverse=True)
            return reranked

        reranked = [replace(candidate, rerank_score=candidate.hybrid_score + self._intent_boost(candidate, plan)) for candidate in candidates]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked

    def _load_model(self) -> CrossEncoder | None:
        try:
            return CrossEncoder(self.model_name)
        except Exception:
            return None

    def _intent_boost(self, candidate: RetrievedCandidate, plan: QueryPlan) -> float:
        metadata = candidate.metadata
        content_type = str(metadata.get("content_type", "")).strip().lower()
        boost = 0.0

        if content_type in {"definition", "concept_explanation", "formula", "theorem"}:
            boost += 0.35
        if content_type in {"summary", "important_note"}:
            boost += 0.22
        if content_type == "misconception":
            boost += 0.30
        if content_type in {"example", "activity"}:
            boost -= 0.30
        if bool(metadata.get("is_example_only", False)):
            boost -= 0.40

        if plan.intent.intent_type == "flashcards" and content_type in {"definition", "formula", "summary", "important_note"}:
            boost += 0.20
        if plan.intent.intent_type == "mcq" and content_type in {"concept_explanation", "solved_problem", "important_note"}:
            boost += 0.18
        if plan.intent.intent_type == "hots" and content_type in {"exercise", "solved_problem", "theorem", "procedure", "experiment"}:
            boost += 0.24
        if plan.intent.intent_type == "summary" and content_type in {"summary", "definition", "concept_explanation"}:
            boost += 0.24
        if plan.intent.intent_type == "conceptual_qa" and content_type in {"definition", "concept_explanation", "formula", "theorem", "important_note"}:
            boost += 0.20

        if metadata.get("primary_concept"):
            boost += 0.05
        if metadata.get("misconceptions"):
            boost += 0.05
        return boost

    def _candidate_text(self, candidate: RetrievedCandidate) -> str:
        metadata = candidate.metadata
        parts = [
            str(metadata.get("primary_concept", "")),
            " ".join(metadata.get("definitions", []) or []),
            " ".join(metadata.get("formulae", []) or []),
            " ".join(metadata.get("learning_objectives", []) or []),
            candidate.text,
        ]
        return "\n".join(part for part in parts if part)

