from __future__ import annotations

import json
import re
from pathlib import Path


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def humanize_key(value: str) -> str:
    cleaned = value.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in cleaned.split())


def extract_json_text(raw_text: str) -> str:
    trimmed = raw_text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```(?:json)?", "", trimmed).strip()
        if trimmed.endswith("```"):
            trimmed = trimmed[:-3].strip()

    first_object = trimmed.find("{")
    first_array = trimmed.find("[")
    candidates = [index for index in [first_object, first_array] if index != -1]
    if not candidates:
        return trimmed

    start = min(candidates)
    end_object = trimmed.rfind("}")
    end_array = trimmed.rfind("]")
    end = max(end_object, end_array)
    return trimmed[start : end + 1]


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)
