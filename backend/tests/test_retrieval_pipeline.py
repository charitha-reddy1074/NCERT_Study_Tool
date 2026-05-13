from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.config import Settings
from app.services.context_compressor import ContextCompressor
from app.services.hybrid_retriever import EducationalHybridRetriever
from app.services.intent_router import IntentRouter
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.query_transformer import QueryTransformer
from app.services.retrieval_models import QueryPlan, RetrievedCandidate, RetrievalIntent
from app.services.retrieval_pipeline import EducationalRetrievalPipeline


@dataclass
class FakeVectorStore:
    docs: list[Document]

    def list_documents(self, *, filter: dict | None = None, batch_size: int = 200) -> list[Document]:
        if not filter:
            return self.docs[:]
        result: list[Document] = []
        for document in self.docs:
            metadata = document.metadata or {}
            if metadata.get("class_num") != filter.get("class_num"):
                continue
            if filter.get("subject") and metadata.get("subject") != filter.get("subject"):
                continue
            if filter.get("chapter") and metadata.get("chapter") != filter.get("chapter"):
                continue
            result.append(document)
        return result

    def similarity_search_with_scores(self, query: str, *, k: int, filter: dict | None = None):
        ranked = []
        query_lower = query.lower()
        for document in self.list_documents(filter=filter):
            text = document.page_content.lower()
            score = 0.15
            if "integer" in query_lower and "integer" in text:
                score += 0.5
            if document.metadata.get("content_type") == "definition":
                score += 0.25
            if document.metadata.get("is_example_only"):
                score -= 0.2
            ranked.append((document, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]


def make_document(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_intent_router_detects_educational_intents() -> None:
    router = IntentRouter()

    flashcards = router.detect(task_type="flashcards", query="make flashcards for integers")
    mcq = router.detect(task_type="quiz", query="Generate MCQ on integers", quiz_type="mcq")
    hots = router.detect(task_type="quiz", query="Why does the sign change matter?", quiz_type="short_answer")

    assert flashcards.intent_type == "flashcards"
    assert mcq.intent_type == "mcq"
    assert hots.intent_type == "hots"
    assert "example" not in flashcards.include_content_types


def test_query_transformer_expands_concept_queries() -> None:
    router = IntentRouter()
    intent = router.detect(task_type="quiz", query="Generate MCQ on integers", quiz_type="mcq")
    plan = QueryTransformer().transform(query="Generate MCQ on integers", intent=intent, class_num=10, subject="Mathematics", chapter="Integers")

    assert "integers" in plan.transformed_query.lower()
    assert "definitions" in plan.transformed_query.lower()
    assert "misconceptions" in plan.transformed_query.lower()
    assert plan.metadata_filter["class_num"] == 10
    assert plan.metadata_filter["subject"] == "mathematics"


def test_metadata_filter_builder_excludes_examples_for_flashcards() -> None:
    intent = RetrievalIntent(intent_type="flashcards", include_content_types=["definition"], exclude_content_types=["example"], blooms_levels=["remember"])
    plan = QueryPlan(
        original_query="flashcards on integers",
        transformed_query="integers definitions formulas",
        bm25_query="integers definitions formulas",
        intent=intent,
        metadata_filter={"class_num": 10, "subject": "mathematics", "chapter": "integers"},
    )

    metadata_filter = MetadataFilterBuilder().build(plan)

    assert metadata_filter["is_example_only"] is False
    assert metadata_filter["content_type"]["$in"]
    assert "definition" in metadata_filter["content_type"]["$in"]


def test_hybrid_retriever_prefers_concept_chunks_over_examples() -> None:
    docs = [
        make_document(
            "Subject: Mathematics\nChapter: Integers\nConcept: Integers\nDefinitions: Integers are whole numbers that can be positive, negative, or zero.\nContent: Integers are whole numbers that can be positive, negative, or zero.",
            chunk_id="def-1",
            class_num=10,
            subject="mathematics",
            chapter="integers",
            content_type="definition",
            primary_concept="Integers",
            is_example_only=False,
        ),
        make_document(
            "Subject: Mathematics\nChapter: Integers\nConcept: Integers\nContent: Example: shopping mall floors below ground represent negative numbers.",
            chunk_id="ex-1",
            class_num=10,
            subject="mathematics",
            chapter="integers",
            content_type="example",
            primary_concept="Integers",
            is_example_only=True,
        ),
    ]
    vectorstore = FakeVectorStore(docs)
    intent = RetrievalIntent(intent_type="mcq", include_content_types=["definition", "concept_explanation"], exclude_content_types=["example"], blooms_levels=["understand"])
    plan = QueryPlan(
        original_query="Generate MCQ on integers",
        transformed_query="integers definitions properties operations applications misconceptions",
        bm25_query="integers definition property operation application misconception",
        intent=intent,
        metadata_filter={"class_num": 10, "subject": "mathematics", "chapter": "integers", "content_type": {"$in": ["definition", "concept_explanation"]}},
    )

    candidates = EducationalHybridRetriever(vectorstore).retrieve(plan, top_k=2)

    assert len(candidates) == 2
    assert candidates[0].metadata["content_type"] == "definition"
    assert candidates[0].hybrid_score >= candidates[1].hybrid_score


def test_context_compressor_builds_structured_context() -> None:
    candidate = RetrievedCandidate(
        document=make_document(
            "Subject: Mathematics\nChapter: Integers\nConcept: Integers\nContent: Integers are whole numbers that can be positive, negative, or zero.",
            primary_concept="Integers",
            definitions=["Integers are whole numbers that can be positive, negative, or zero."],
            formulae=["a + b = c"],
            examples=["shopping mall floors"],
            misconceptions=["students may confuse negative values with subtraction"],
            learning_objectives=["understand integer addition"],
            content_type="definition",
        ),
        rerank_score=0.9,
    )
    context = ContextCompressor().compress([candidate])

    assert "Integers" in context.concepts[0]
    assert context.definitions
    assert context.formulae
    assert context.applications
    assert context.misconceptions
    assert context.learning_objectives


def test_pipeline_retrieves_structured_context() -> None:
    docs = [
        make_document(
            "Subject: Mathematics\nChapter: Integers\nConcept: Integers\nDefinitions: Integers are whole numbers that can be positive, negative, or zero.\nLearning Objective: understand integer addition\nContent: Integers are whole numbers that can be positive, negative, or zero.",
            chunk_id="def-1",
            class_num=10,
            subject="mathematics",
            chapter="integers",
            content_type="definition",
            primary_concept="Integers",
            is_example_only=False,
            definitions=["Integers are whole numbers that can be positive, negative, or zero."],
            learning_objectives=["understand integer addition"],
        )
    ]
    pipeline = EducationalRetrievalPipeline(Settings(), FakeVectorStore(docs))
    pipeline.reranker._load_model = lambda: None  # type: ignore[method-assign]
    pipeline.diversity_selector._load_model = lambda: None  # type: ignore[method-assign]

    documents, citations, structured_context, scope, not_in_textbook = pipeline.retrieve(
        task_type="flashcards",
        query="make flashcards for integers",
        class_num=10,
        subject="Mathematics",
        chapter="Integers",
        top_k=1,
    )

    assert documents
    assert citations
    assert structured_context.definitions
    assert scope
    assert not not_in_textbook
