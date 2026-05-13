from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from app.core.utils import humanize_key, normalize_key
from app.services.educational_models import ChunkDraft, PreparedDocument


class MetadataBuilder:
    def build(
        self,
        chunk: ChunkDraft,
        *,
        class_num: int,
        subject: str,
        chapter_no: str,
        chapter_name: str,
        source_pdf: str,
        source_path: str,
    ) -> dict[str, object]:
        normalized_subject = normalize_key(subject)
        normalized_chapter = normalize_key(chapter_name)
        section_path = " > ".join(chunk.sections[0].heading_path) if chunk.sections else chunk.heading
        primary_concept = chunk.enrichment.concepts.primary_concept
        chunk_key = "|".join(
            [
                str(class_num),
                normalized_subject,
                chapter_no,
                normalized_chapter,
                section_path,
                primary_concept,
                chunk.text[:500],
            ]
        )
        chunk_id = str(uuid5(NAMESPACE_URL, chunk_key))

        metadata: dict[str, object] = {
            "chunk_id": chunk_id,
            "class_num": class_num,
            "class": str(class_num),
            "subject": normalized_subject,
            "subject_label": humanize_key(subject),
            "chapter_no": str(chapter_no),
            "chapter": normalized_chapter,
            "chapter_name": chapter_name,
            "heading": chunk.heading,
            "subheading": chunk.subheading,
            "section_path": section_path,
            "content_type": chunk.enrichment.classification.content_type,
            "primary_concept": primary_concept,
            "secondary_concepts": chunk.enrichment.concepts.secondary_concepts,
            "related_concepts": chunk.enrichment.concepts.related_concepts,
            "prerequisite_concepts": chunk.enrichment.concepts.prerequisite_concepts,
            "formulae": chunk.enrichment.concepts.formulae,
            "definitions": chunk.enrichment.concepts.definitions,
            "examples": chunk.enrichment.concepts.examples,
            "abstract_concepts": chunk.enrichment.concepts.abstract_concepts,
            "misconceptions": chunk.enrichment.misconceptions,
            "learning_objectives": chunk.enrichment.learning_objectives,
            "blooms_level": chunk.enrichment.classification.blooms_level,
            "difficulty": chunk.enrichment.classification.difficulty,
            "question_potential": chunk.enrichment.classification.question_potential,
            "is_conceptual": chunk.enrichment.classification.is_conceptual,
            "is_example_only": chunk.enrichment.classification.is_example_only,
            "source_pdf": source_pdf,
            "source_path": source_path,
            "page_no": chunk.page_nos[0] if chunk.page_nos else 1,
            "page_nos": chunk.page_nos,
            "topic": chunk.heading,
            "subtopic": chunk.subheading or None,
            "chunk_strategy": "concept-aware-educational",
        }

        if any("exercise" in heading.lower() for heading in chunk.sections[0].heading_path if chunk.sections):
            metadata["exercise_section"] = chunk.heading if chunk.sections else None

        return metadata

