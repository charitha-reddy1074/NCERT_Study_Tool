from __future__ import annotations

from app.core.utils import normalize_key, to_json
from app.services.educational_models import PreparedDocument


class EmbeddingPipeline:
    def build_embedding_text(self, metadata: dict[str, object], chunk_text: str) -> str:
        subject = self._string(metadata.get("subject_label") or metadata.get("subject"))
        chapter = self._string(metadata.get("chapter_name") or metadata.get("chapter"))
        heading = self._string(metadata.get("heading"))
        subheading = self._string(metadata.get("subheading"))
        concept = self._string(metadata.get("primary_concept"))
        objectives = self._join_list(metadata.get("learning_objectives"))
        definitions = self._join_list(metadata.get("definitions"))
        formulae = self._join_list(metadata.get("formulae"))
        related = self._join_list(metadata.get("related_concepts"))
        misconceptions = self._join_list(metadata.get("misconceptions"))

        segments = [
            f"Subject: {subject}" if subject else None,
            f"Chapter: {chapter}" if chapter else None,
            f"Heading: {heading}" if heading else None,
            f"Subheading: {subheading}" if subheading else None,
            f"Concept: {concept}" if concept else None,
            f"Learning Objective: {objectives}" if objectives else None,
            f"Definitions: {definitions}" if definitions else None,
            f"Formulae: {formulae}" if formulae else None,
            f"Related Concepts: {related}" if related else None,
            f"Misconceptions: {misconceptions}" if misconceptions else None,
            f"Content Type: {self._string(metadata.get('content_type'))}" if metadata.get("content_type") else None,
            f"Content: {chunk_text.strip()}",
        ]
        return "\n".join(segment for segment in segments if segment)

    def prepare_document(self, metadata: dict[str, object], chunk_text: str) -> PreparedDocument:
        embedding_text = self.build_embedding_text(metadata, chunk_text)
        prepared_metadata = dict(metadata)
        prepared_metadata["embedding_text"] = embedding_text
        prepared_metadata["chunk_text"] = chunk_text
        prepared_metadata["embedding_profile"] = {
            "subject": self._string(metadata.get("subject_label") or metadata.get("subject")),
            "chapter": self._string(metadata.get("chapter_name") or metadata.get("chapter")),
            "concept": self._string(metadata.get("primary_concept")),
            "content_type": self._string(metadata.get("content_type")),
        }
        return PreparedDocument(text=embedding_text, metadata=prepared_metadata)

    def _string(self, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _join_list(self, value: object) -> str:
        if not value:
            return ""
        if isinstance(value, list):
            return "; ".join(str(item).strip() for item in value if str(item).strip())
        return str(value)

