from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.utils import SUPPORTED_TEXT_EXTENSIONS, humanize_key, normalize_key
from app.schemas import IngestRequest, IngestResponse
from app.services.concept_extractor import ConceptExtractor
from app.services.educational_classifier import EducationalClassifier
from app.services.embedding_pipeline import EmbeddingPipeline
from app.services.educational_models import AnalyzedSection, EnrichmentProfile
from app.services.hybrid_chunker import ConceptAwareChunker
from app.services.learning_objective_extractor import LearningObjectiveExtractor
from app.services.metadata_builder import MetadataBuilder
from app.services.misconception_detector import MisconceptionDetector
from app.services.parser import TextbookParser
from app.services.vectorstore import VectorStoreService


class IngestService:
    def __init__(self, settings: Settings, vectorstore: VectorStoreService) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.parser = TextbookParser()
        self.classifier = EducationalClassifier()
        self.concept_extractor = ConceptExtractor()
        self.misconception_detector = MisconceptionDetector()
        self.learning_objective_extractor = LearningObjectiveExtractor()
        self.chunker = ConceptAwareChunker(
            min_tokens=700,
            max_tokens=1000,
            overlap_tokens=120,
        )
        self.metadata_builder = MetadataBuilder()
        self.embedding_pipeline = EmbeddingPipeline()

    def ingest_directory(self, payload: IngestRequest) -> IngestResponse:
        source_dir = Path(payload.source_dir) if payload.source_dir else self.settings.ncert_data_dir / f"class-{payload.class_num}"
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        requested_subject = normalize_key(payload.subject) if payload.subject else None
        requested_chapter = normalize_key(payload.chapter) if payload.chapter else None

        if payload.clear_existing:
            self.vectorstore.reset()

        if not payload.clear_existing and requested_subject and requested_chapter:
            if self.vectorstore.has_documents(
                filter={
                    "class_num": payload.class_num,
                    "subject": requested_subject,
                    "chapter": requested_chapter,
                }
            ):
                return IngestResponse(
                    source_dir=str(source_dir),
                    files_processed=0,
                    chunks_indexed=0,
                    collection_size=self.vectorstore.count_documents(),
                    skipped_files=[],
                    note="Chapter already indexed in cache.",
                )

        supported_files = sorted(
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
        )
        skipped_files: list[str] = []
        files_processed = 0
        prepared_documents: list[Document] = []

        for file_path in supported_files:
            try:
                inferred_subject, inferred_chapter = self._infer_subject_and_chapter(file_path, source_dir)
                normalized_subject = normalize_key(inferred_subject)
                normalized_chapter = normalize_key(inferred_chapter)

                if requested_subject and normalized_subject != requested_subject:
                    continue
                if requested_chapter and normalized_chapter != requested_chapter:
                    continue

                display_subject = payload.subject or inferred_subject
                display_chapter = payload.chapter or inferred_chapter
                chapter_no = self._infer_chapter_number(file_path, display_chapter)

                sections = self.parser.parse(file_path, chapter_name=display_chapter)
                if not sections:
                    continue

                analyzed_sections = [self._analyze_section(section, display_subject, display_chapter) for section in sections]
                chunks = self.chunker.chunk(analyzed_sections)

                for chunk in chunks:
                    metadata = self.metadata_builder.build(
                        chunk,
                        class_num=payload.class_num,
                        subject=display_subject,
                        chapter_no=chapter_no,
                        chapter_name=display_chapter,
                        source_pdf=file_path.name,
                        source_path=str(file_path),
                    )
                    prepared = self.embedding_pipeline.prepare_document(metadata, chunk.text)
                    prepared_documents.append(Document(page_content=prepared.text, metadata=prepared.metadata))

                files_processed += 1
            except Exception:
                skipped_files.append(str(file_path))

        if not prepared_documents:
            return IngestResponse(
                source_dir=str(source_dir),
                files_processed=0,
                chunks_indexed=0,
                collection_size=self.vectorstore.count_documents(),
                skipped_files=skipped_files,
                note="No supported documents were indexed.",
            )

        chunk_ids = [str(document.metadata.get("chunk_id")) for document in prepared_documents]
        self.vectorstore.add_documents(prepared_documents, ids=chunk_ids)

        return IngestResponse(
            source_dir=str(source_dir),
            files_processed=files_processed,
            chunks_indexed=len(prepared_documents),
            collection_size=self.vectorstore.count_documents(),
            skipped_files=skipped_files,
            note=None,
        )

    def _analyze_section(self, section, subject: str, chapter_name: str) -> AnalyzedSection:
        classification = self.classifier.classify(section, chapter_name=chapter_name, subject=subject)
        concepts = self.concept_extractor.extract(section, content_type=classification.content_type, chapter_name=chapter_name, subject=subject)
        misconceptions = self.misconception_detector.detect(
            section,
            primary_concept=concepts.primary_concept,
            content_type=classification.content_type,
            subject=subject,
        )
        learning_objectives = self.learning_objective_extractor.extract(
            section,
            primary_concept=concepts.primary_concept,
            content_type=classification.content_type,
            blooms_level=classification.blooms_level,
        )
        enrichment = EnrichmentProfile(
            classification=classification,
            concepts=concepts,
            misconceptions=misconceptions,
            learning_objectives=learning_objectives,
        )
        return AnalyzedSection(section=section, enrichment=enrichment)

    def _infer_subject_and_chapter(self, file_path: Path, source_root: Path) -> tuple[str, str]:
        relative = file_path.relative_to(source_root)
        parts = relative.parts

        inferred_subject = parts[0] if len(parts) > 1 else file_path.parent.name

        if len(parts) >= 3:
            inferred_chapter = parts[1]
        else:
            inferred_chapter = file_path.stem

        return inferred_subject, inferred_chapter

    def _infer_chapter_number(self, file_path: Path, chapter_name: str) -> str:
        match = re.search(r"(?:chapter|ch\.?|unit)\s*(\d+)", file_path.stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"(\d+)", chapter_name)
        if match:
            return match.group(1)
        return normalize_key(chapter_name) or "1"

