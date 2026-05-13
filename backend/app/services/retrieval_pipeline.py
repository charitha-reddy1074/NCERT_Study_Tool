from __future__ import annotations

from app.core.config import Settings
from app.schemas import SourceCitation
from app.services.context_compressor import ContextCompressor
from app.services.diversity_selector import MMRDiversitySelector
from app.services.hybrid_retriever import EducationalHybridRetriever
from app.services.intent_router import IntentRouter
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.query_transformer import QueryTransformer
from app.services.reranker import EducationalReranker
from app.services.retrieval_models import QueryPlan, RetrievedCandidate, StructuredEducationalContext
from app.services.vectorstore import VectorStoreService
from langchain_core.documents import Document


class EducationalRetrievalPipeline:
    def __init__(self, settings: Settings, vectorstore: VectorStoreService) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.intent_router = IntentRouter()
        self.query_transformer = QueryTransformer()
        self.metadata_filters = MetadataFilterBuilder()
        self.hybrid_retriever = EducationalHybridRetriever(vectorstore)
        self.reranker = EducationalReranker()
        self.diversity_selector = MMRDiversitySelector()
        self.context_compressor = ContextCompressor()

    def retrieve(self, *, task_type: str, query: str, class_num: int, subject: str | None, chapter: str | None, top_k: int, quiz_type: str | None = None) -> tuple[list[Document], list[SourceCitation], StructuredEducationalContext, str, bool]:
        intent = self.intent_router.detect(task_type=task_type, query=query, quiz_type=quiz_type)
        plan = self.query_transformer.transform(
            query=query,
            intent=intent,
            class_num=class_num,
            subject=subject,
            chapter=chapter,
        )
        plan.metadata_filter = self.metadata_filters.build(plan)

        candidates = self.hybrid_retriever.retrieve(plan, top_k=top_k)
        if not candidates:
            scope_bits = [f"class {class_num}"]
            if subject:
                scope_bits.append(subject)
            if chapter:
                scope_bits.append(chapter)
            return [], [], StructuredEducationalContext(), " / ".join(scope_bits), True

        reranked = self.reranker.rerank(plan, candidates)
        selected = self.diversity_selector.select(plan, reranked, top_k=top_k)
        structured_context = self.context_compressor.compress(selected)

        documents = [candidate.document for candidate in selected]
        citations = [self._citation_from_candidate(candidate) for candidate in selected]
        scope_bits = [f"class {class_num}"]
        if subject:
            scope_bits.append(subject)
        if chapter:
            scope_bits.append(chapter)

        not_in_textbook = self._out_of_scope(selected)
        return documents, citations, structured_context, " / ".join(scope_bits), not_in_textbook

    def _citation_from_candidate(self, candidate: RetrievedCandidate) -> SourceCitation:
        metadata = candidate.metadata
        return SourceCitation(
            source_path=str(metadata.get("source_path", "")),
            file_name=str(metadata.get("file_name", "")),
            class_num=self._safe_int(metadata.get("class_num")),
            subject=str(metadata.get("subject", "")) or None,
            chapter=str(metadata.get("chapter", "")) or None,
            page=self._safe_int(metadata.get("page_no") if metadata.get("page_no") is not None else metadata.get("page")),
            chunk_id=str(metadata.get("chunk_id", "")) or None,
            relevance_score=float(candidate.rerank_score or candidate.hybrid_score),
            excerpt=candidate.text[:350].strip(),
            topic=str(metadata.get("topic", "")) or None,
            exercise_section=str(metadata.get("exercise_section", "")) or None,
        )

    def _safe_int(self, value: object) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _out_of_scope(self, candidates: list[RetrievedCandidate]) -> bool:
        if not candidates:
            return True
        best = max((candidate.rerank_score or candidate.hybrid_score for candidate in candidates), default=0.0)
        return best < self.settings.answer_relevance_threshold

