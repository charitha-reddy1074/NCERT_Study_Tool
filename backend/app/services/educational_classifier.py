from __future__ import annotations

import re

from app.core.utils import humanize_key, normalize_key
from app.services.educational_models import ClassificationResult, ParsedSection


class EducationalClassifier:
    def classify(self, section: ParsedSection, *, chapter_name: str, subject: str | None) -> ClassificationResult:
        text = self._clean(section.text)
        lowered = text.lower()
        heading = self._heading_text(section)

        content_type = self._detect_content_type(text, lowered, heading, subject)
        blooms_level = self._infer_blooms_level(content_type, text, lowered)
        difficulty = self._infer_difficulty(content_type, text, lowered, subject)
        question_potential = self._question_potential(content_type, blooms_level, lowered)
        is_example_only = content_type == "example"
        is_conceptual = content_type not in {"example", "exercise"} or bool(re.search(r"how|why|what happens|explain", lowered))

        return ClassificationResult(
            content_type=content_type,
            blooms_level=blooms_level,
            difficulty=difficulty,
            question_potential=question_potential,
            is_conceptual=is_conceptual,
            is_example_only=is_example_only,
            example_type=self._example_type(lowered) if content_type == "example" else None,
        )

    def _detect_content_type(self, text: str, lowered: str, heading: str, subject: str | None) -> str:
        normalized_heading = normalize_key(heading)
        combined = f"{normalized_heading} {lowered}"

        if self._matches_any(combined, ["definition", "defined as", "is called", "means"]):
            return "definition"
        if self._matches_any(combined, ["formula", "equation", "=", "therefore", "hence"]):
            if any(symbol in text for symbol in ["=", "+", "-", "×", "÷", "^", "/"]):
                return "formula"
        if self._matches_any(combined, ["theorem", "law", "principle", "corollary", "postulate"]):
            return "theorem"
        if self._matches_any(combined, ["example", "for example", "let us see", "illustration", "consider"]):
            return "example"
        if self._matches_any(combined, ["activity", "do this", "try this", "let us do", "activity time"]):
            return "activity"
        if re.match(r"^[A-Z][A-Za-z0-9\- ]{2,60}\s+(is|are|was|were)\b", text) and len(text.split()) <= 45:
            return "definition"
        if self._matches_any(combined, ["exercise", "review questions", "practice", "answer the following", "solve"]):
            if self._matches_any(combined, ["solution", "answer", "steps", "work out"]):
                return "solved_problem"
            return "exercise"
        if self._matches_any(combined, ["experiment", "observe", "observation", "materials", "procedure"]):
            if self._matches_any(combined, ["procedure", "step 1", "step 2", "step 3"]):
                return "procedure"
            return "experiment"
        if self._matches_any(combined, ["diagram", "figure", "label the", "observe the figure", "illustrated"]):
            return "diagram_explanation"
        if self._matches_any(combined, ["summary", "in short", "let us revise", "recap", "chapter summary"]):
            return "summary"
        if self._matches_any(combined, ["important", "remember", "note", "key point", "caution"]):
            return "important_note"
        if self._looks_like_procedure(text, lowered):
            return "procedure"
        if self._looks_like_fact(text, lowered, subject):
            return "fact"
        return "concept_explanation"

    def _infer_blooms_level(self, content_type: str, text: str, lowered: str) -> str:
        if content_type in {"definition", "fact", "summary", "important_note"}:
            return "remember"
        if content_type in {"formula", "procedure", "exercise", "solved_problem", "activity"}:
            return "apply"
        if content_type in {"experiment", "diagram_explanation"}:
            return "analyze"
        if content_type == "theorem":
            return "understand"
        if re.search(r"compare|contrast|differentiate|analyze|why does|how does", lowered):
            return "analyze"
        if re.search(r"justify|evaluate|prove|argue", lowered):
            return "evaluate"
        if len(text.split()) > 180:
            return "analyze"
        return "understand"

    def _infer_difficulty(self, content_type: str, text: str, lowered: str, subject: str | None) -> str:
        if content_type in {"definition", "fact", "summary"}:
            return "easy"
        if content_type in {"formula", "procedure", "activity"}:
            return "medium"
        if content_type in {"exercise", "solved_problem", "experiment", "theorem"}:
            return "hard" if len(text.split()) > 120 or re.search(r"prove|derive|reason|analyze|apply", lowered) else "medium"
        if subject and normalize_key(subject) in {"mathematics", "science"} and re.search(r"why|how|explain", lowered):
            return "medium"
        return "medium"

    def _question_potential(self, content_type: str, blooms_level: str, lowered: str) -> list[str]:
        options: list[str] = []
        if content_type in {"definition", "fact"}:
            options.extend(["flashcard", "mcq"])
        if content_type in {"concept_explanation", "summary", "important_note", "diagram_explanation"}:
            options.extend(["flashcard", "mcq", "concept_qa"])
        if content_type in {"formula", "procedure", "solved_problem"} or blooms_level == "apply":
            options.extend(["mcq", "application_question", "assertion_reason"])
        if blooms_level in {"analyze", "evaluate"} or re.search(r"why|compare|predict|justify", lowered):
            options.extend(["hots", "application_question", "assertion_reason"])
        if content_type == "example":
            options.extend(["application_question", "concept_qa"])
        if content_type == "exercise":
            options.extend(["mcq", "application_question", "hots"])
        seen: set[str] = set()
        ordered: list[str] = []
        for item in options:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered[:4] or ["mcq", "flashcard"]

    def _example_type(self, lowered: str) -> str:
        if re.search(r"shopping|money|price|market|floor|bus|train|road|direction|distance|temperature", lowered):
            return "real_world"
        if re.search(r"table|list|diagram|figure", lowered):
            return "textbook"
        return "worked_example"

    def _looks_like_procedure(self, text: str, lowered: str) -> bool:
        step_markers = len(re.findall(r"\bstep\s*\d+|^\d+[.)]", lowered, flags=re.MULTILINE))
        imperative_markers = len(re.findall(r"\b(add|mix|observe|measure|calculate|draw|write|complete|use|take)\b", lowered))
        return step_markers >= 2 or (imperative_markers >= 3 and len(text.split()) > 40)

    def _looks_like_fact(self, text: str, lowered: str, subject: str | None) -> bool:
        if len(text.split()) <= 35 and re.search(r"\b(is|are|was|were|has|have|contain|includes)\b", lowered):
            return True
        subject_key = normalize_key(subject or "")
        return subject_key in {"social-science", "science"} and len(text.split()) <= 50

    def _matches_any(self, text: str, candidates: list[str]) -> bool:
        return any(candidate in text for candidate in candidates)

    def _heading_text(self, section: ParsedSection) -> str:
        if section.heading_path:
            return " ".join(section.heading_path)
        return humanize_key(section.source_path.stem)

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
