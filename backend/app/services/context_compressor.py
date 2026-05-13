from __future__ import annotations

from collections import OrderedDict

from app.core.utils import normalize_key, to_json
from app.services.retrieval_models import RetrievedCandidate, StructuredEducationalContext


class ContextCompressor:
    def compress(self, candidates: list[RetrievedCandidate]) -> StructuredEducationalContext:
        concepts: list[str] = []
        definitions: list[str] = []
        formulae: list[str] = []
        applications: list[str] = []
        misconceptions: list[str] = []
        learning_objectives: list[str] = []

        for candidate in candidates:
            metadata = candidate.metadata
            concept_bits = [
                str(metadata.get("primary_concept", "")),
                *self._ensure_list(metadata.get("secondary_concepts")),
                *self._ensure_list(metadata.get("related_concepts")),
                *self._ensure_list(metadata.get("abstract_concepts")),
            ]
            concepts.extend([bit for bit in concept_bits if bit])
            definitions.extend(self._ensure_list(metadata.get("definitions")))
            formulae.extend(self._ensure_list(metadata.get("formulae")))
            applications.extend(self._applications_from_candidate(candidate))
            misconceptions.extend(self._ensure_list(metadata.get("misconceptions")))
            learning_objectives.extend(self._ensure_list(metadata.get("learning_objectives")))

        return StructuredEducationalContext(
            concepts=self._dedupe(concepts)[:8],
            definitions=self._dedupe(definitions)[:8],
            formulae=self._dedupe(formulae)[:8],
            applications=self._dedupe(applications)[:8],
            misconceptions=self._dedupe(misconceptions)[:8],
            learning_objectives=self._dedupe(learning_objectives)[:8],
        )

    def serialize(self, context: StructuredEducationalContext) -> str:
        return to_json(
            {
                "concepts": context.concepts,
                "definitions": context.definitions,
                "formulae": context.formulae,
                "applications": context.applications,
                "misconceptions": context.misconceptions,
                "learning_objectives": context.learning_objectives,
            }
        )

    def _applications_from_candidate(self, candidate: RetrievedCandidate) -> list[str]:
        metadata = candidate.metadata
        applications = []
        applications.extend(self._ensure_list(metadata.get("examples")))
        applications.extend(self._ensure_list(metadata.get("abstract_concepts")))
        content_type = str(metadata.get("content_type", "")).strip().lower()
        if content_type in {"solved_problem", "exercise", "procedure", "experiment", "activity"}:
            applications.append(candidate.text[:240])
        return applications

    def _ensure_list(self, value: object) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _dedupe(self, values: list[str]) -> list[str]:
        ordered: OrderedDict[str, str] = OrderedDict()
        for value in values:
            cleaned = " ".join(str(value).split()).strip()
            if not cleaned:
                continue
            key = normalize_key(cleaned)
            if key not in ordered:
                ordered[key] = cleaned
        return list(ordered.values())

