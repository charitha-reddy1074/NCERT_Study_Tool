from __future__ import annotations

from pathlib import Path
import re

from app.core.config import Settings
from app.core.utils import SUPPORTED_TEXT_EXTENSIONS, humanize_key, normalize_key
from app.schemas import CatalogChapter, CatalogResponse, CatalogSubject


DEFAULT_SUBJECTS = [
    ("mathematics", "Mathematics"),
    ("science", "Science"),
    ("social-science", "Social Science"),
    ("english", "English"),
    ("hindi", "Hindi"),
]


class CatalogService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_catalog(self, class_num: int) -> CatalogResponse:
        class_dir = self.settings.ncert_data_dir / f"class-{class_num}"
        if not class_dir.exists():
            return CatalogResponse(
                class_num=class_num,
                subjects=[CatalogSubject(key=key, label=label, chapter_count=0, chapters=[]) for key, label in DEFAULT_SUBJECTS],
                source_dir=str(class_dir),
                note="No textbook files were found yet. Place official NCERT files under the class directory and ingest them.",
            )

        subjects: list[CatalogSubject] = []
        for subject_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            chapters = self._discover_chapters(subject_dir)
            subjects.append(
                CatalogSubject(
                    key=normalize_key(subject_dir.name),
                    label=humanize_key(subject_dir.name),
                    chapter_count=len(chapters),
                    chapters=chapters,
                )
            )

        root_chapters = self._discover_root_chapters(class_dir)
        if root_chapters:
            subjects.append(
                CatalogSubject(
                    key="uploaded-textbooks",
                    label="Uploaded Textbooks",
                    chapter_count=len(root_chapters),
                    chapters=root_chapters,
                )
            )

        note = None
        if not subjects:
            subjects = [CatalogSubject(key=key, label=label, chapter_count=0, chapters=[]) for key, label in DEFAULT_SUBJECTS]
            note = "No textbook files were found yet. Add official NCERT PDFs under backend/data/ncert/class-6/ and ingest them."

        return CatalogResponse(class_num=class_num, subjects=subjects, source_dir=str(class_dir), note=note)

    def _discover_chapters(self, subject_dir: Path) -> list[CatalogChapter]:
        chapters: dict[str, CatalogChapter] = {}
        for file_path in subject_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
                continue
            if not self._is_chapter_pdf(file_path):
                continue

            relative = file_path.relative_to(subject_dir)
            if len(relative.parts) > 1:
                chapter_key = normalize_key(relative.parts[0])
                chapter_label = humanize_key(relative.parts[0])
            else:
                chapter_key = normalize_key(file_path.stem)
                chapter_label = humanize_key(file_path.stem)

            chapters.setdefault(
                chapter_key,
                CatalogChapter(key=chapter_key, label=chapter_label, file_path=str(file_path)),
            )

        return sorted(chapters.values(), key=lambda chapter: chapter.label)

    def _discover_root_chapters(self, class_dir: Path) -> list[CatalogChapter]:
        chapters: dict[str, CatalogChapter] = {}
        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
                continue
            if not self._is_chapter_pdf(file_path):
                continue

            chapter_key = normalize_key(file_path.stem)
            chapters.setdefault(
                chapter_key,
                CatalogChapter(key=chapter_key, label=humanize_key(file_path.stem), file_path=str(file_path)),
            )

        return sorted(chapters.values(), key=lambda chapter: chapter.label)

    def _is_chapter_pdf(self, file_path: Path) -> bool:
        if file_path.suffix.lower() != ".pdf":
            return False
        return re.match(r"^chapter\s*\d+", file_path.stem.strip().lower()) is not None
