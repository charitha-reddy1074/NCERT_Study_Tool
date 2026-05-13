from __future__ import annotations

from app.core.utils import humanize_key, normalize_key
from app.services.retrieval_models import QueryPlan, RetrievalIntent


class QueryTransformer:
    def transform(
        self,
        *,
        query: str,
        intent: RetrievalIntent,
        class_num: int,
        subject: str | None,
        chapter: str | None,
    ) -> QueryPlan:
        base_query = self._clean(query)
        subject_label = humanize_key(subject or "") if subject else ""
        chapter_label = humanize_key(chapter or "") if chapter else ""

        concept_terms: list[str] = []
        if subject:
            concept_terms.append(subject_label)
        if chapter:
            concept_terms.append(chapter_label)
        concept_terms.extend(self._content_type_terms(intent.include_content_types))
        concept_terms.extend(self._intent_terms(intent.intent_type))

        transformed_query = self._dedupe_terms([base_query, subject_label, chapter_label, *concept_terms])
        bm25_query = self._dedupe_terms([base_query, subject_label, chapter_label, *self._bm25_terms(intent.intent_type), *self._content_type_terms(intent.include_content_types)])

        metadata_filter: dict[str, object] = {"class_num": class_num}
        if subject:
            metadata_filter["subject"] = normalize_key(subject)
        if chapter:
            metadata_filter["chapter"] = normalize_key(chapter)

        return QueryPlan(
            original_query=query,
            transformed_query=transformed_query,
            bm25_query=bm25_query,
            intent=intent,
            metadata_filter=metadata_filter,
            expand_terms=concept_terms,
            suppress_terms=intent.exclude_content_types[:],
        )

    def _intent_terms(self, intent_type: str) -> list[str]:
        mapping = {
            "flashcards": ["definitions", "formulae", "key concepts", "memory cues", "concept summary"],
            "mcq": ["properties", "operations", "applications", "misconceptions", "conceptual understanding"],
            "hots": ["why", "how", "analyze", "compare", "evaluate", "reasoning"],
            "summary": ["main ideas", "learning objectives", "relationships", "important points"],
            "conceptual_qa": ["concept explanation", "applications", "definitions", "misconceptions"],
        }
        return mapping.get(intent_type, ["concept explanation", "definitions", "applications"])

    def _content_type_terms(self, content_types: list[str]) -> list[str]:
        mapping = {
            "definition": ["definition", "definitions"],
            "formula": ["formula", "formulae"],
            "concept_explanation": ["concept explanation", "concept explanations"],
            "summary": ["summary", "summaries"],
            "important_note": ["important note", "important notes"],
            "theorem": ["theorem", "theorems"],
            "solved_problem": ["solved problem", "solved problems"],
            "exercise": ["exercise", "exercises"],
            "procedure": ["procedure", "procedures"],
            "experiment": ["experiment", "experiments"],
        }
        terms: list[str] = []
        for content_type in content_types:
            terms.extend(mapping.get(content_type, [content_type.replace("_", " ")]))
        return terms

    def _bm25_terms(self, intent_type: str) -> list[str]:
        mapping = {
            "flashcards": ["definition", "formula", "concept", "key point"],
            "mcq": ["definition", "property", "operation", "application", "misconception"],
            "hots": ["why", "how", "analyze", "justify", "compare"],
            "summary": ["summary", "learning objective", "important", "relationship"],
            "conceptual_qa": ["concept", "definition", "application", "misconception"],
        }
        return mapping.get(intent_type, ["concept", "definition", "application"])

    def _dedupe_terms(self, terms: list[str]) -> str:
        seen: set[str] = set()
        ordered: list[str] = []
        for term in terms:
            cleaned = self._clean(term)
            if not cleaned:
                continue
            key = normalize_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cleaned)
        return " ".join(ordered)

    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").replace("\r", " ").split())

