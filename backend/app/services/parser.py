from __future__ import annotations

import re
from pathlib import Path

from app.core.utils import humanize_key, normalize_key
from app.services.educational_models import ParsedSection

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None


class TextbookParser:
    def parse(self, file_path: Path, *, chapter_name: str | None = None) -> list[ParsedSection]:
        if file_path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path, chapter_name=chapter_name)
        return self._parse_text(file_path, chapter_name=chapter_name)

    def _parse_pdf(self, file_path: Path, *, chapter_name: str | None = None) -> list[ParsedSection]:
        chapter_title = chapter_name or humanize_key(file_path.stem)
        if fitz is None:
            return self._parse_pdf_fallback(file_path, chapter_title)

        document = fitz.open(str(file_path))
        heading_stack: list[tuple[int, str]] = [(1, chapter_title)]
        block_index = 0
        sections: list[ParsedSection] = []

        for page_number, page in enumerate(document, start=1):
            page_dict = page.get_text("dict")
            blocks = sorted(page_dict.get("blocks", []), key=lambda block: (block.get("bbox", [0, 0, 0, 0])[1], block.get("bbox", [0, 0, 0, 0])[0]))
            page_max_font = self._page_max_font_size(blocks)

            for raw_block in blocks:
                lines = raw_block.get("lines") or []
                text = self._block_text(lines)
                text = self._clean_text(text)
                if not text:
                    continue

                font_size = self._block_font_size(lines)
                if self._is_heading_candidate(text, font_size, page_max_font, chapter_title):
                    heading_stack = self._update_heading_stack(heading_stack, text, chapter_title)
                    continue

                for paragraph in self._split_paragraphs(text):
                    sections.append(
                        ParsedSection(
                            source_path=file_path,
                            source_name=file_path.name,
                            page_no=page_number,
                            block_index=block_index,
                            heading_path=[heading for _, heading in heading_stack],
                            level=heading_stack[-1][0],
                            text=paragraph,
                            source_kind="pdf",
                            font_size=font_size,
                            is_heading=False,
                        )
                    )
                    block_index += 1

        return sections

    def _parse_pdf_fallback(self, file_path: Path, chapter_title: str) -> list[ParsedSection]:
        try:
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("PyMuPDF is unavailable and pypdf fallback could not be loaded") from exc

        reader = PdfReader(str(file_path))
        heading_stack: list[tuple[int, str]] = [(1, chapter_title)]
        sections: list[ParsedSection] = []
        block_index = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean_text(page.extract_text() or "")
            if not text:
                continue
            for paragraph in self._split_paragraphs(text):
                if self._is_heading_candidate(paragraph, None, None, chapter_title):
                    heading_stack = self._update_heading_stack(heading_stack, paragraph, chapter_title)
                    continue
                sections.append(
                    ParsedSection(
                        source_path=file_path,
                        source_name=file_path.name,
                        page_no=page_number,
                        block_index=block_index,
                        heading_path=[heading for _, heading in heading_stack],
                        level=heading_stack[-1][0],
                        text=paragraph,
                        source_kind="pdf",
                        is_heading=False,
                    )
                )
                block_index += 1
        return sections

    def _parse_text(self, file_path: Path, *, chapter_name: str | None = None) -> list[ParsedSection]:
        chapter_title = chapter_name or humanize_key(file_path.stem)
        heading_stack: list[tuple[int, str]] = [(1, chapter_title)]
        sections: list[ParsedSection] = []
        text = self._read_text(file_path)
        block_index = 0

        for paragraph in self._split_paragraphs(text):
            if self._is_heading_candidate(paragraph, None, None, chapter_title):
                heading_stack = self._update_heading_stack(heading_stack, paragraph, chapter_title)
                continue

            sections.append(
                ParsedSection(
                    source_path=file_path,
                    source_name=file_path.name,
                    page_no=1,
                    block_index=block_index,
                    heading_path=[heading for _, heading in heading_stack],
                    level=heading_stack[-1][0],
                    text=paragraph,
                    source_kind="text",
                    is_heading=False,
                )
            )
            block_index += 1

        return sections

    def _read_text(self, file_path: Path) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except Exception:
                continue
        raise ValueError(f"Could not load text file: {file_path}")

    def _split_paragraphs(self, text: str) -> list[str]:
        normalized = self._clean_text(text)
        if not normalized:
            return []
        return [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", normalized) if paragraph.strip()]

    def _update_heading_stack(self, heading_stack: list[tuple[int, str]], heading_text: str, chapter_title: str) -> list[tuple[int, str]]:
        heading_level = self._detect_heading_level(heading_text, chapter_title)
        normalized_heading = normalize_key(heading_text)
        normalized_chapter = normalize_key(chapter_title)

        if heading_level <= 1:
            new_stack: list[tuple[int, str]] = [(1, chapter_title)]
            if normalized_heading and normalized_heading != normalized_chapter:
                new_stack.append((2, heading_text))
            return new_stack

        while len(heading_stack) > 1 and heading_stack[-1][0] >= heading_level:
            heading_stack.pop()

        heading_stack.append((heading_level, heading_text))
        return heading_stack

    def _detect_heading_level(self, paragraph: str, chapter_title: str | None = None) -> int:
        cleaned = self._clean_text(paragraph)
        if not cleaned or len(cleaned) > 140:
            return 0

        normalized = normalize_key(cleaned)
        if chapter_title and normalized == normalize_key(chapter_title):
            return 1

        lowered = cleaned.lower()
        if re.match(r"^(chapter|unit|lesson)\s+\d+\b", lowered):
            return 1

        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+.+", cleaned):
            depth = cleaned.split()[0].count(".")
            return min(4, depth + 2)

        if cleaned.isupper() and len(cleaned.split()) <= 12:
            return 2

        if any(lowered.startswith(prefix) for prefix in ["exercise", "summary", "key words", "key points", "activity", "activities", "let us", "what we have learned", "review questions"]):
            return 2

        words = cleaned.split()
        if len(words) <= 10 and not cleaned.endswith((".", "?", "!")):
            title_like = sum(1 for word in words if word[:1].isupper()) / max(len(words), 1)
            if title_like >= 0.7:
                return 2 if len(words) > 2 else 1

        return 0

    def _is_heading_candidate(self, text: str, font_size: float | None, page_max_font: float | None, chapter_title: str) -> bool:
        if self._detect_heading_level(text, chapter_title) > 0:
            return True
        if font_size is None or page_max_font is None:
            return False
        if len(text.split()) > 14:
            return False
        if font_size >= max(page_max_font - 0.5, page_max_font * 0.85):
            return True
        return False

    def _page_max_font_size(self, blocks: list[dict]) -> float | None:
        sizes: list[float] = []
        for block in blocks:
            sizes.extend(self._block_font_sizes(block.get("lines") or []))
        return max(sizes) if sizes else None

    def _block_font_size(self, lines: list[dict]) -> float | None:
        sizes = self._block_font_sizes(lines)
        return max(sizes) if sizes else None

    def _block_font_sizes(self, lines: list[dict]) -> list[float]:
        sizes: list[float] = []
        for line in lines:
            for span in line.get("spans") or []:
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))
        return sizes

    def _block_text(self, lines: list[dict]) -> str:
        parts: list[str] = []
        for line in lines:
            line_text = "".join(span.get("text", "") for span in line.get("spans") or [])
            line_text = self._clean_text(line_text)
            if line_text:
                parts.append(line_text)
        return "\n".join(parts)

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

