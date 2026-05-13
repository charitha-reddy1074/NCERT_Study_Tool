from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document

from app.services.retrieval_models import QueryPlan, RetrievedCandidate
from app.services.vectorstore import VectorStoreService


class BM25Retriever:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized_docs = [self._tokenize(self._document_text(document)) for document in documents]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_docs]
        self._avg_doc_length = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        self._doc_freqs: dict[str, int] = defaultdict(int)
        for tokens in self._tokenized_docs:
            for token in set(tokens):
                self._doc_freqs[token] += 1
        self._document_count = len(documents)

    def search(self, query: str, *, top_k: int) -> list[tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores: list[tuple[int, float]] = []
        for index, doc_tokens in enumerate(self._tokenized_docs):
            if not doc_tokens:
                continue
            freq = Counter(doc_tokens)
            score = 0.0
            doc_len = self._doc_lengths[index] or 1
            for token in query_tokens:
                if token not in freq:
                    continue
                df = self._doc_freqs.get(token, 0)
                idf = math.log(1 + (self._document_count - df + 0.5) / (df + 0.5)) if df else 0.0
                numerator = freq[token] * (self.k1 + 1)
                denominator = freq[token] + self.k1 * (1 - self.b + self.b * (doc_len / max(self._avg_doc_length, 1e-6)))
                score += idf * (numerator / denominator)
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.documents[index], score) for index, score in scores[:top_k]]

    def _document_text(self, document: Document) -> str:
        metadata = document.metadata or {}
        parts = [
            str(metadata.get("primary_concept", "")),
            " ".join(metadata.get("secondary_concepts", []) or []),
            " ".join(metadata.get("related_concepts", []) or []),
            " ".join(metadata.get("definitions", []) or []),
            " ".join(metadata.get("formulae", []) or []),
            " ".join(metadata.get("misconceptions", []) or []),
            " ".join(metadata.get("learning_objectives", []) or []),
            document.page_content or "",
        ]
        return " ".join(part for part in parts if part)

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


class EducationalHybridRetriever:
    def __init__(self, vectorstore: VectorStoreService) -> None:
        self.vectorstore = vectorstore

    def retrieve(self, plan: QueryPlan, *, top_k: int) -> list[RetrievedCandidate]:
        documents = self.vectorstore.list_documents(filter=plan.metadata_filter)
        if not documents:
            return []

        bm25 = BM25Retriever(documents)
        bm25_results = bm25.search(plan.bm25_query, top_k=max(top_k * 4, top_k))
        bm25_scores = self._normalize_scores(bm25_results)

        vector_results = self.vectorstore.similarity_search_with_scores(
            plan.transformed_query,
            k=max(top_k * 4, top_k),
            filter=plan.metadata_filter,
        )
        vector_scores = self._normalize_vector_scores(vector_results)

        candidate_map: dict[str, RetrievedCandidate] = {}
        for document, score in vector_results:
            key = self._candidate_key(document)
            candidate_map[key] = RetrievedCandidate(
                document=document,
                vector_score=vector_scores.get(key, float(score)),
                bm25_score=0.0,
                hybrid_score=0.0,
            )

        for document, score in bm25_results:
            key = self._candidate_key(document)
            candidate = candidate_map.get(key)
            if candidate is None:
                candidate = RetrievedCandidate(document=document)
                candidate_map[key] = candidate
            candidate.bm25_score = bm25_scores.get(key, float(score))

        candidates = list(candidate_map.values())
        for candidate in candidates:
            candidate.hybrid_score = (candidate.vector_score * 0.6) + (candidate.bm25_score * 0.4)
            candidate.hybrid_score += self._metadata_boost(candidate.metadata, plan)

        candidates.sort(key=lambda item: item.hybrid_score, reverse=True)
        return candidates[: max(top_k * 4, top_k)]

    def _candidate_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        return str(metadata.get("chunk_id") or metadata.get("source_path") or hash(document.page_content))

    def _metadata_boost(self, metadata: dict[str, object], plan: QueryPlan) -> float:
        content_type = str(metadata.get("content_type", "")).strip().lower()
        primary_concept = str(metadata.get("primary_concept", "")).strip().lower()
        is_example_only = bool(metadata.get("is_example_only", False))
        boost = 0.0

        if content_type in plan.intent.include_content_types:
            boost += 0.12
        if content_type in {"definition", "concept_explanation", "formula", "theorem"}:
            boost += 0.15
        if content_type in {"summary", "important_note", "misconception"}:
            boost += 0.10
        if content_type in {"example", "activity"}:
            boost -= 0.12
        if is_example_only:
            boost -= 0.18
        if primary_concept and any(term in primary_concept for term in self._normalize_terms(plan.expand_terms)):
            boost += 0.08
        return boost

    def _normalize_terms(self, terms: list[str]) -> list[str]:
        return [re.sub(r"\s+", " ", term.strip().lower()) for term in terms if term.strip()]

    def _normalize_scores(self, results: list[tuple[Document, float]]) -> dict[str, float]:
        if not results:
            return {}
        min_score = min(score for _, score in results)
        max_score = max(score for _, score in results)
        span = max(max_score - min_score, 1e-6)
        normalized: dict[str, float] = {}
        for document, score in results:
            normalized[self._candidate_key(document)] = (score - min_score) / span
        return normalized

    def _normalize_vector_scores(self, results: list[tuple[Document, float]]) -> dict[str, float]:
        if not results:
            return {}
        max_score = max(score for _, score in results)
        min_score = min(score for _, score in results)
        span = max(max_score - min_score, 1e-6)
        normalized: dict[str, float] = {}
        for document, score in results:
            normalized[self._candidate_key(document)] = (score - min_score) / span
        return normalized

