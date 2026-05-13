from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.services.concept_extractor import ConceptExtractor
from app.services.educational_classifier import EducationalClassifier
from app.services.embedding_pipeline import EmbeddingPipeline
from app.services.educational_models import AnalyzedSection, ClassificationResult, ConceptProfile, EnrichmentProfile, ParsedSection
from app.services.hybrid_chunker import ConceptAwareChunker
from app.services.learning_objective_extractor import LearningObjectiveExtractor
from app.services.metadata_builder import MetadataBuilder
from app.services.misconception_detector import MisconceptionDetector


def make_section(text: str, *, heading_path: list[str] | None = None, page_no: int = 1, block_index: int = 0) -> ParsedSection:
    return ParsedSection(
        source_path=Path("chapter_1.pdf"),
        source_name="chapter_1.pdf",
        page_no=page_no,
        block_index=block_index,
        heading_path=heading_path or ["Mathematics", "Integers", "Addition of Integers"],
        level=2,
        text=text,
        source_kind="pdf",
    )


def test_classifier_distinguishes_core_content_types() -> None:
    classifier = EducationalClassifier()

    definition = classifier.classify(make_section("Integers are whole numbers that can be positive, negative, or zero."), chapter_name="Integers", subject="Mathematics")
    formula = classifier.classify(make_section("a + b = c is the rule for addition."), chapter_name="Integers", subject="Mathematics")
    example = classifier.classify(make_section("Example: shopping mall floors below ground are represented by negative numbers."), chapter_name="Integers", subject="Mathematics")

    assert definition.content_type == "definition"
    assert formula.content_type == "formula"
    assert example.content_type == "example"
    assert example.example_type == "real_world"


def test_concept_extractor_and_misconception_detection_are_concept_aware() -> None:
    section = make_section("Integers are whole numbers that can be positive, negative, or zero. For example, shopping mall floors below ground can be represented using negative numbers.")
    extractor = ConceptExtractor()
    detector = MisconceptionDetector()

    concepts = extractor.extract(section, content_type="definition", chapter_name="Integers", subject="Mathematics")
    misconceptions = detector.detect(section, primary_concept=concepts.primary_concept, content_type="definition", subject="Mathematics")

    assert concepts.primary_concept == "Integers"
    assert "integer arithmetic" in concepts.abstract_concepts or concepts.abstract_concepts == []
    assert concepts.definitions
    assert any("negative" in item.lower() for item in misconceptions)


def test_learning_objectives_and_metadata_capture_educational_tags() -> None:
    section = make_section("Integers are whole numbers that can be positive, negative, or zero.")
    classification = ClassificationResult(
        content_type="definition",
        blooms_level="remember",
        difficulty="easy",
        question_potential=["flashcard", "mcq"],
        is_conceptual=True,
        is_example_only=False,
    )
    concepts = ConceptProfile(primary_concept="Integers", definitions=["whole numbers that can be positive, negative, or zero"])
    learning_objectives = LearningObjectiveExtractor().extract(section, primary_concept="Integers", content_type="definition", blooms_level="remember")
    enrichment = EnrichmentProfile(classification=classification, concepts=concepts, misconceptions=["sign confusion"], learning_objectives=learning_objectives)
    analyzed = AnalyzedSection(section=section, enrichment=enrichment)
    chunk = ConceptAwareChunker(min_tokens=10, max_tokens=50, overlap_tokens=5).chunk([analyzed])[0]

    metadata = MetadataBuilder().build(
        chunk,
        class_num=10,
        subject="Mathematics",
        chapter_no="1",
        chapter_name="Integers",
        source_pdf="chapter_1.pdf",
        source_path=str(section.source_path),
    )
    prepared = EmbeddingPipeline().prepare_document(metadata, chunk.text)

    assert metadata["content_type"] == "definition"
    assert metadata["primary_concept"] == "Integers"
    assert metadata["blooms_level"] == "remember"
    assert metadata["question_potential"] == ["flashcard", "mcq"]
    assert metadata["is_conceptual"] is True
    assert UUID(str(metadata["chunk_id"]))
    assert "Subject: Mathematics" in prepared.text
    assert "Concept: Integers" in prepared.text
    assert "Learning Objective:" in prepared.text
    assert "Content:" in prepared.text


def test_chunker_keeps_definition_and_example_separate() -> None:
    definition = AnalyzedSection(
        section=make_section("Integers are whole numbers that can be positive, negative, or zero.", block_index=0),
        enrichment=EnrichmentProfile(
            classification=ClassificationResult(
                content_type="definition",
                blooms_level="remember",
                difficulty="easy",
                question_potential=["flashcard"],
            ),
            concepts=ConceptProfile(primary_concept="Integers"),
        ),
    )
    example = AnalyzedSection(
        section=make_section("Example: shopping mall floors below ground are represented by negative numbers.", block_index=1),
        enrichment=EnrichmentProfile(
            classification=ClassificationResult(
                content_type="example",
                blooms_level="apply",
                difficulty="medium",
                question_potential=["application_question"],
                is_conceptual=False,
                is_example_only=True,
                example_type="real_world",
            ),
            concepts=ConceptProfile(primary_concept="Integers", abstract_concepts=["integer arithmetic"]),
        ),
    )

    chunks = ConceptAwareChunker(min_tokens=10, max_tokens=50, overlap_tokens=5).chunk([definition, example])

    assert len(chunks) == 2
    assert chunks[0].enrichment.classification.content_type == "definition"
    assert chunks[1].enrichment.classification.content_type == "example"
