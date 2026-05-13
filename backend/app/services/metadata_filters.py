from __future__ import annotations

from app.services.retrieval_models import QueryPlan


class MetadataFilterBuilder:
    def build(self, plan: QueryPlan) -> dict[str, object]:
        raw: dict[str, object] = dict(plan.metadata_filter)

        if plan.intent.intent_type == "flashcards":
            raw["content_type"] = {"$in": ["definition", "formula", "concept_explanation", "summary", "important_note", "theorem"]}
            raw["is_example_only"] = False
        elif plan.intent.intent_type == "mcq":
            raw["content_type"] = {"$in": ["definition", "concept_explanation", "formula", "theorem", "solved_problem", "important_note"]}
            raw["is_example_only"] = False
        elif plan.intent.intent_type == "hots":
            raw["content_type"] = {"$in": ["concept_explanation", "exercise", "solved_problem", "theorem", "important_note", "procedure", "experiment"]}
            raw["blooms_level"] = {"$in": ["apply", "analyze", "evaluate"]}
        elif plan.intent.intent_type == "summary":
            raw["content_type"] = {"$in": ["summary", "definition", "concept_explanation", "formula", "important_note", "theorem"]}
            raw["is_example_only"] = False
        else:
            raw["content_type"] = {"$in": ["definition", "concept_explanation", "formula", "theorem", "important_note", "solved_problem"]}
            raw["is_example_only"] = False

        if plan.intent.blooms_levels:
            raw.setdefault("blooms_level", {"$in": plan.intent.blooms_levels})

        return raw

