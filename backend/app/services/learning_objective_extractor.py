from __future__ import annotations

import re

from app.core.utils import normalize_key
from app.services.educational_models import ParsedSection


class LearningObjectiveExtractor:
    def extract(self, section: ParsedSection, *, primary_concept: str, content_type: str, blooms_level: str) -> list[str]:
        concept = primary_concept.strip() or "the topic"
        objectives: list[str] = []

        if content_type == "definition":
            objectives.append(f"define {concept} accurately")
            objectives.append(f"identify {concept} in textbook and real-world examples")
        elif content_type == "formula":
            objectives.append(f"apply the formula for {concept}")
            objectives.append(f"explain what each term in the formula represents")
        elif content_type == "example":
            objectives.append(f"map a textbook example to the underlying concept of {concept}")
            objectives.append(f"solve similar problems using the same method")
        elif content_type in {"exercise", "solved_problem"}:
            objectives.append(f"solve practice problems on {concept}")
            objectives.append(f"choose the correct strategy for {concept}-based questions")
        elif content_type in {"experiment", "procedure", "activity"}:
            objectives.append(f"follow the steps needed to investigate {concept}")
            objectives.append(f"record observations and explain the result")
        elif content_type == "theorem":
            objectives.append(f"state and use the rule or theorem related to {concept}")
            objectives.append(f"justify why the rule works in the given context")
        else:
            objectives.append(f"understand the core idea of {concept}")
            objectives.append(f"explain how {concept} connects to related concepts")

        if blooms_level == "analyze":
            objectives.append(f"compare {concept} with related ideas and extract the differences")
        elif blooms_level == "evaluate":
            objectives.append(f"judge when {concept} can or cannot be applied")
        elif blooms_level == "apply":
            objectives.append(f"use {concept} to solve textbook and application questions")

        if re.search(r"why|how|compare|different", section.text.lower()):
            objectives.append(f"answer reasoning questions about {concept}")

        return self._dedupe(objectives)

    def _dedupe(self, items: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = normalize_key(item)
            if key and key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered[:4]
