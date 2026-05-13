from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from app.services.retrieval_models import QueryPlan, RetrievedCandidate


class MMRDiversitySelector:
    def __init__(self, lambda_param: float = 0.72, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.lambda_param = lambda_param
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def select(self, plan: QueryPlan, candidates: list[RetrievedCandidate], *, top_k: int) -> list[RetrievedCandidate]:
        if not candidates:
            return []

        candidate_texts = [self._selection_text(candidate) for candidate in candidates]
        query_embedding, candidate_embeddings = self._embed(plan.transformed_query, candidate_texts)

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        content_type_counts: dict[str, int] = {}

        while remaining and len(selected) < top_k:
            best_index = None
            best_score = float("-inf")
            for index in remaining:
                relevance = float(cosine_similarity(query_embedding, candidate_embeddings[index : index + 1])[0][0])
                diversity_penalty = 0.0
                if selected:
                    sims = cosine_similarity(candidate_embeddings[index : index + 1], candidate_embeddings[selected])[0]
                    diversity_penalty = float(np.max(sims))

                candidate = candidates[index]
                type_bonus = self._type_bonus(candidate, content_type_counts)
                mmr = (self.lambda_param * relevance) - ((1 - self.lambda_param) * diversity_penalty) + type_bonus + candidate.rerank_score
                if mmr > best_score:
                    best_score = mmr
                    best_index = index

            if best_index is None:
                break

            selected.append(best_index)
            remaining.remove(best_index)
            selected_candidate = candidates[best_index]
            content_type = str(selected_candidate.metadata.get("content_type", "")).strip().lower() or "other"
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1

        results: list[RetrievedCandidate] = []
        for rank, index in enumerate(selected[:top_k], start=1):
            candidate = candidates[index]
            mmr_score = candidate.rerank_score + (top_k - rank) * 0.001
            results.append(replace(candidate, mmr_score=mmr_score))
        return results

    def _type_bonus(self, candidate: RetrievedCandidate, content_type_counts: dict[str, int]) -> float:
        content_type = str(candidate.metadata.get("content_type", "")).strip().lower()
        if not content_type:
            return 0.0
        if content_type_counts.get(content_type, 0) == 0:
            return 0.10
        if content_type in {"definition", "formula", "misconception", "summary"}:
            return 0.04
        return -0.02 * content_type_counts.get(content_type, 0)

    def _embed(self, query: str, texts: list[str]):
        if self._model is None:
            self._model = self._load_model()

        if self._model is not None:
            query_embedding = np.asarray(self._model.encode([query], normalize_embeddings=True), dtype=float)
            candidate_embeddings = np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=float)
            return query_embedding, candidate_embeddings

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query, *texts]).toarray().astype(float)
        query_embedding = matrix[:1]
        candidate_embeddings = matrix[1:]
        return query_embedding, candidate_embeddings

    def _load_model(self) -> SentenceTransformer | None:
        try:
            return SentenceTransformer(self.model_name)
        except Exception:
            return None

    def _selection_text(self, candidate: RetrievedCandidate) -> str:
        metadata = candidate.metadata
        parts = [
            str(metadata.get("primary_concept", "")),
            str(metadata.get("content_type", "")),
            str(metadata.get("blooms_level", "")),
            " ".join(metadata.get("definitions", []) or []),
            " ".join(metadata.get("formulae", []) or []),
            " ".join(metadata.get("misconceptions", []) or []),
            " ".join(metadata.get("learning_objectives", []) or []),
            candidate.text,
        ]
        return "\n".join(part for part in parts if part)

