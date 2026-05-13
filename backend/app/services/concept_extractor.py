from __future__ import annotations

import re

from app.core.utils import humanize_key, normalize_key
from app.services.educational_models import ConceptProfile, ParsedSection


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "while",
    "have",
    "has",
    "been",
    "are",
    "was",
    "were",
    "can",
    "will",
    "may",
    "should",
    "must",
    "your",
    "their",
    "our",
    "its",
    "they",
    "you",
    "his",
    "her",
    "because",
    "there",
    "here",
    "also",
    "such",
    "more",
    "most",
    "less",
    "very",
    "chapter",
    "exercise",
    "example",
    "note",
    "summary",
    "activity",
    "figure",
    "table",
    "page",
}


class ConceptExtractor:
    def extract(self, section: ParsedSection, *, content_type: str, chapter_name: str, subject: str | None) -> ConceptProfile:
        text = self._clean(section.text)
        primary = self._primary_concept(section, text, content_type, chapter_name)
        secondary = self._keyword_concepts(text, primary)
        related = self._related_concepts(primary, secondary, subject)
        prerequisites = self._prerequisite_concepts(primary, subject)
        examples = self._extract_examples(text, content_type)
        abstract_concepts = [self._abstract_example(example, primary, section) for example in examples]
        formulae = self._extract_formulae(text)
        definitions = self._extract_definitions(text, primary)

        return ConceptProfile(
            primary_concept=primary,
            secondary_concepts=secondary,
            related_concepts=related,
            prerequisite_concepts=prerequisites,
            examples=examples,
            abstract_concepts=abstract_concepts,
            formulae=formulae,
            definitions=definitions,
        )

    def _primary_concept(self, section: ParsedSection, text: str, content_type: str, chapter_name: str) -> str:
        if content_type == "definition":
            definition_subject = re.match(r"^([A-Z][A-Za-z0-9\- ]{1,60})\s+(?:is|are|was|were)\b", text)
            if definition_subject:
                return self._normalize_concept(definition_subject.group(1))

        heading_candidates = [heading for heading in section.heading_path[1:] if self._is_meaningful_heading(heading)]
        if heading_candidates:
            if content_type in {"example", "exercise", "solved_problem"} or len(heading_candidates) == 1:
                return self._normalize_concept(heading_candidates[-1])
            return self._normalize_concept(heading_candidates[0])

        definition_match = re.search(r"(?:definition of|define|defined as|is called)\s+([A-Za-z0-9\- ]{3,80})", text, flags=re.IGNORECASE)
        if definition_match:
            return self._normalize_concept(definition_match.group(1))

        if content_type == "formula":
            formula_match = re.search(r"([A-Za-z][A-Za-z0-9_ ]{2,60})\s*[=:]", text)
            if formula_match:
                return self._normalize_concept(formula_match.group(1))

        chapter_concept = self._normalize_concept(chapter_name)
        if chapter_concept:
            return chapter_concept

        keywords = self._keyword_concepts(text, "")
        if keywords:
            return keywords[0]

        return self._normalize_concept(section.source_path.stem)

    def _keyword_concepts(self, text: str, primary: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
        concepts: list[str] = []
        seen = {normalize_key(primary)} if primary else set()
        for word in words:
            normalized = normalize_key(word)
            if len(normalized) < 3 or normalized in STOPWORDS or normalized in seen:
                continue
            if word[0].isupper() or len(concepts) < 6:
                concepts.append(self._normalize_concept(word))
                seen.add(normalized)
            if len(concepts) >= 5:
                break
        return concepts

    def _related_concepts(self, primary: str, secondary: list[str], subject: str | None) -> list[str]:
        base = [concept for concept in secondary[:2] if concept.lower() != primary.lower()]
        subject_key = normalize_key(subject or "")
        concept_key = normalize_key(primary)

        if subject_key == "mathematics" and any(token in concept_key for token in ["integer", "number", "fraction", "ratio", "decimal"]):
            base.extend(["number line", "arithmetic", "place value"])
        elif subject_key == "science" and any(token in concept_key for token in ["force", "motion", "light", "sound", "matter", "cell", "plant"]):
            base.extend(["observation", "cause and effect", "application"])
        elif subject_key in {"social-science", "social-science"}:
            base.extend(["cause and effect", "timeline", "location"])

        ordered: list[str] = []
        seen: set[str] = set()
        for item in base:
            normalized = normalize_key(item)
            if normalized and normalized not in seen and normalized != normalize_key(primary):
                seen.add(normalized)
                ordered.append(item)
        return ordered[:4]

    def _prerequisite_concepts(self, primary: str, subject: str | None) -> list[str]:
        primary_key = normalize_key(primary)
        prerequisites: list[str] = []
        if any(token in primary_key for token in ["integer", "fraction", "decimal"]):
            prerequisites.extend(["whole numbers", "number sense"])
        if any(token in primary_key for token in ["geometry", "angle", "triangle", "quadrilateral"]):
            prerequisites.extend(["basic shapes", "measurement"])
        if any(token in primary_key for token in ["matter", "force", "motion", "cell", "plant"]):
            prerequisites.extend(["observation", "basic science vocabulary"])
        if normalize_key(subject or "") == "mathematics":
            prerequisites.append("arithmetic")
        return self._dedupe(prerequisites)

    def _extract_examples(self, text: str, content_type: str) -> list[str]:
        if content_type == "example":
            return [self._trim_excerpt(text)]

        matches = re.findall(r"(?:for example|example|illustration|consider)[:\- ]+(.+?)(?:\.|;|$)", text, flags=re.IGNORECASE)
        return [self._trim_excerpt(match) for match in matches if match.strip()]

    def _abstract_example(self, example: str, primary: str, section: ParsedSection) -> str:
        lowered = example.lower()
        subject_key = normalize_key(section.heading_path[0] if section.heading_path else section.source_path.stem)

        abstraction_rules = [
            (["shopping", "floor", "lift", "elevator", "up", "down", "direction"], "positive-negative directional movement"),
            (["money", "price", "cost", "rupee", "coins", "change"], "numerical comparison and arithmetic"),
            (["temperature", "cold", "hot", "degree"], "measurement and sign interpretation"),
            (["distance", "road", "journey", "bus", "train", "travel"], "real-world quantity change"),
            (["table", "graph", "chart"], "data interpretation"),
            (["force", "push", "pull", "motion"], "force and motion concepts"),
        ]

        for keywords, abstraction in abstraction_rules:
            if any(keyword in lowered for keyword in keywords):
                return abstraction

        if subject_key == "mathematics" and any(token in normalize_key(primary) for token in ["integer", "number"]):
            return "integer arithmetic"
        if subject_key == "science" and any(token in normalize_key(primary) for token in ["matter", "force", "motion"]):
            return self._normalize_concept(primary)

        return self._normalize_concept(primary)

    def _extract_formulae(self, text: str) -> list[str]:
        matches = re.findall(r"(?:formula|equation|rule)[:\- ]+(.+?)(?:\.|;|$)", text, flags=re.IGNORECASE)
        if matches:
            return [self._trim_excerpt(match) for match in matches if match.strip()]
        if re.search(r"[A-Za-z]\s*[=±+\-×÷*/^]\s*[A-Za-z0-9]", text):
            return [self._trim_excerpt(text)]
        return []

    def _extract_definitions(self, text: str, primary: str) -> list[str]:
        matches = re.findall(r"(?:is defined as|definition of|defined as|means|is called)[:\- ]+(.+?)(?:\.|;|$)", text, flags=re.IGNORECASE)
        if matches:
            return [self._trim_excerpt(match) for match in matches if match.strip()]
        if re.search(r"\bis\s+called\b|\bmeans\b", text, flags=re.IGNORECASE):
            return [self._trim_excerpt(text)]
        return [self._normalize_concept(primary)] if primary else []

    def _is_meaningful_heading(self, heading: str) -> bool:
        cleaned = normalize_key(heading)
        return bool(cleaned) and cleaned not in {"chapter", "exercise", "summary", "activity", "examples"}

    def _normalize_concept(self, value: str) -> str:
        return humanize_key(normalize_key(value).replace("-", " "))

    def _trim_excerpt(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        return cleaned[:240]

    def _dedupe(self, items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = normalize_key(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(item)
        return ordered

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
