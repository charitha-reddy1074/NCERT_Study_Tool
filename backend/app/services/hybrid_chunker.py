from __future__ import annotations

import re

from app.services.educational_models import AnalyzedSection, ChunkDraft, ConceptProfile, EnrichmentProfile, ParsedSection

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


class ConceptAwareChunker:
    def __init__(self, *, min_tokens: int = 700, max_tokens: int = 1000, overlap_tokens: int = 120) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

    def chunk(self, analyzed_sections: list[AnalyzedSection]) -> list[ChunkDraft]:
        ordered = sorted(analyzed_sections, key=lambda item: (item.section.page_no, item.section.block_index))
        chunks: list[ChunkDraft] = []
        current_sections: list[AnalyzedSection] = []
        current_tokens = 0
        current_anchor: str | None = None

        for item in ordered:
            if not item.section.text.strip():
                continue

            if self._is_isolating_type(item.enrichment.classification.content_type):
                if current_sections:
                    chunks.extend(self._flush_group(current_sections))
                    current_sections = []
                    current_tokens = 0
                    current_anchor = None
                chunks.extend(self._emit_isolated(item))
                continue

            anchor = self._normalize_anchor(item.enrichment.concepts.primary_concept)
            item_tokens = self._token_count(item.section.text)

            if current_sections and current_anchor and anchor != current_anchor and current_tokens >= self.min_tokens:
                chunks.extend(self._flush_group(current_sections))
                current_sections = []
                current_tokens = 0

            if current_sections and current_tokens + item_tokens > self.max_tokens:
                chunks.extend(self._flush_group(current_sections))
                current_sections = []
                current_tokens = 0

            current_sections.append(item)
            current_tokens += item_tokens
            current_anchor = anchor

        if current_sections:
            chunks.extend(self._flush_group(current_sections))

        return [chunk for chunk in chunks if chunk.text.strip()]

    def _flush_group(self, analyzed_sections: list[AnalyzedSection]) -> list[ChunkDraft]:
        if not analyzed_sections:
            return []

        merged = self._merge_group(analyzed_sections)
        if self._token_count(merged.text) <= self.max_tokens:
            return [merged]

        return self._split_large_chunk(merged)

    def _merge_group(self, analyzed_sections: list[AnalyzedSection]) -> ChunkDraft:
        texts = [item.section.text.strip() for item in analyzed_sections if item.section.text.strip()]
        first = analyzed_sections[0]
        section = first.section
        enrichment = self._merge_enrichment(analyzed_sections)
        page_nos = sorted({item.section.page_no for item in analyzed_sections})
        heading = section.heading_path[1] if len(section.heading_path) > 1 else section.heading_path[0]
        subheading = section.heading_path[2] if len(section.heading_path) > 2 else ""
        return ChunkDraft(
            text="\n\n".join(texts).strip(),
            sections=[item.section for item in analyzed_sections],
            enrichment=enrichment,
            heading=heading,
            subheading=subheading,
            page_nos=page_nos,
        )

    def _merge_enrichment(self, analyzed_sections: list[AnalyzedSection]) -> EnrichmentProfile:
        first = analyzed_sections[0].enrichment
        content_priority = {
            "definition": 0,
            "formula": 1,
            "solved_problem": 2,
            "exercise": 3,
            "example": 4,
            "procedure": 5,
            "experiment": 6,
            "activity": 7,
            "concept_explanation": 8,
            "diagram_explanation": 9,
            "important_note": 10,
            "fact": 11,
            "summary": 12,
            "theorem": 13,
        }
        classification = min((item.enrichment.classification for item in analyzed_sections), key=lambda item: content_priority.get(item.content_type, 99))
        concepts = self._merge_concepts([item.enrichment.concepts for item in analyzed_sections])
        misconceptions = self._dedupe([misconception for item in analyzed_sections for misconception in item.enrichment.misconceptions])
        learning_objectives = self._dedupe([objective for item in analyzed_sections for objective in item.enrichment.learning_objectives])
        return EnrichmentProfile(
            classification=classification,
            concepts=concepts,
            misconceptions=misconceptions,
            learning_objectives=learning_objectives,
        )

    def _merge_concepts(self, concept_profiles: list[ConceptProfile]) -> ConceptProfile:
        primary = concept_profiles[0].primary_concept if concept_profiles else ""
        return ConceptProfile(
            primary_concept=primary,
            secondary_concepts=self._dedupe([concept for profile in concept_profiles for concept in profile.secondary_concepts]),
            related_concepts=self._dedupe([concept for profile in concept_profiles for concept in profile.related_concepts]),
            prerequisite_concepts=self._dedupe([concept for profile in concept_profiles for concept in profile.prerequisite_concepts]),
            examples=self._dedupe([example for profile in concept_profiles for example in profile.examples]),
            abstract_concepts=self._dedupe([concept for profile in concept_profiles for concept in profile.abstract_concepts]),
            formulae=self._dedupe([formula for profile in concept_profiles for formula in profile.formulae]),
            definitions=self._dedupe([definition for profile in concept_profiles for definition in profile.definitions]),
        )

    def _emit_isolated(self, analyzed_section: AnalyzedSection) -> list[ChunkDraft]:
        draft = self._merge_group([analyzed_section])
        if self._token_count(draft.text) <= self.max_tokens:
            return [draft]
        return self._split_large_chunk(draft)

    def _split_large_chunk(self, draft: ChunkDraft) -> list[ChunkDraft]:
        parts = self._paragraph_windows(draft.text)
        if not parts:
            return [draft]

        chunks: list[ChunkDraft] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = self._token_count(part)
            if current_parts and current_tokens + part_tokens > self.max_tokens:
                chunks.append(self._clone_with_text(draft, "\n\n".join(current_parts)))
                current_parts = self._carry_overlap_text(current_parts)
                current_tokens = self._token_count("\n\n".join(current_parts)) if current_parts else 0

            current_parts.append(part)
            current_tokens += part_tokens

        if current_parts:
            chunks.append(self._clone_with_text(draft, "\n\n".join(current_parts)))

        return chunks

    def _clone_with_text(self, draft: ChunkDraft, text: str) -> ChunkDraft:
        return ChunkDraft(
            text=text.strip(),
            sections=draft.sections,
            enrichment=draft.enrichment,
            heading=draft.heading,
            subheading=draft.subheading,
            page_nos=draft.page_nos,
        )

    def _carry_overlap_text(self, parts: list[str]) -> list[str]:
        joined = "\n\n".join(parts)
        if self._encoding is None:
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", joined) if sentence.strip()]
            return sentences[-2:]

        try:
            tokens = self._encoding.encode(joined)
            overlap = tokens[-self.overlap_tokens :] if len(tokens) > self.overlap_tokens else tokens
            text = self._encoding.decode(overlap)
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
            return sentences[-2:] if sentences else [text.strip()]
        except Exception:
            return parts[-2:]

    def _paragraph_windows(self, text: str) -> list[str]:
        return [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]

    def _is_isolating_type(self, content_type: str) -> bool:
        return content_type in {"definition", "formula", "example", "exercise", "solved_problem", "activity", "experiment", "diagram_explanation"}

    def _token_count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return max(1, int(len(text) / 4))
        try:
            return len(self._encoding.encode(text))
        except Exception:
            return max(1, int(len(text) / 4))

    def _normalize_anchor(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def _dedupe(self, items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = self._normalize_anchor(item)
            if key and key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered

