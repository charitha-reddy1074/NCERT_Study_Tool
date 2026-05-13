from __future__ import annotations

import re

from app.core.utils import normalize_key
from app.services.educational_models import ParsedSection


class MisconceptionDetector:
    def detect(self, section: ParsedSection, *, primary_concept: str, content_type: str, subject: str | None) -> list[str]:
        text = self._clean(section.text)
        lowered = text.lower()
        concept_key = normalize_key(primary_concept)
        subject_key = normalize_key(subject or "")
        misconceptions: list[str] = []

        if subject_key == "mathematics" and any(token in concept_key for token in ["integer", "number", "fraction", "decimal"]):
            if re.search(r"negative|minus|sign", lowered):
                misconceptions.append("students may confuse negative values with subtraction or sign changes")
            if re.search(r"add|addition|subtract|subtraction", lowered) and re.search(r"rule|formula", lowered):
                misconceptions.append("students may misuse the sign rule while adding or subtracting integers")
        if subject_key == "science" and any(token in concept_key for token in ["matter", "force", "motion", "cell", "plant"]):
            if re.search(r"heat|temperature|change", lowered):
                misconceptions.append("students may confuse cause and effect when tracking process changes")
            if content_type in {"formula", "procedure"}:
                misconceptions.append("students may apply the procedure in the wrong order")
        if re.search(r"unit|km|m|cm|kg|g|l|ml", lowered):
            misconceptions.append("students may confuse the unit with the quantity being measured")
        if content_type == "formula":
            misconceptions.append("students may misapply the formula without understanding what each term represents")
        if subject_key in {"english", "hindi"} and re.search(r"tense|verb|sentence", lowered):
            misconceptions.append("students may confuse grammar form with meaning in context")
        if re.search(r"because|therefore|so that|as a result", lowered) and subject_key == "science":
            misconceptions.append("students may confuse correlation with cause-effect relationships")

        return self._dedupe(misconceptions)

    def _dedupe(self, items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = normalize_key(item)
            if key and key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
