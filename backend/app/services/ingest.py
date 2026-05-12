from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.utils import SUPPORTED_TEXT_EXTENSIONS, humanize_key, normalize_key
from app.schemas import IngestRequest, IngestResponse
from app.services.vectorstore import VectorStoreService


@dataclass(slots=True)
class ChunkBlock:
    heading_path: list[str]
    level: int
    text: str


class IngestService:
    def __init__(self, settings: Settings, vectorstore: VectorStoreService) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Try to load token encoder (tiktoken). If unavailable, use character fallback.
        try:
            import tiktoken  # type: ignore

            # prefer cl100k_base for modern models; it's a reasonable default
            self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tiktoken_encoding = None

    def ingest_directory(self, payload: IngestRequest) -> IngestResponse:
        source_dir = Path(payload.source_dir) if payload.source_dir else self.settings.ncert_data_dir / f"class-{payload.class_num}"
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        requested_subject = normalize_key(payload.subject) if payload.subject else None
        requested_chapter = normalize_key(payload.chapter) if payload.chapter else None

        if payload.clear_existing:
            self.vectorstore.reset()

        supported_files = sorted(
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
        )
        skipped_files: list[str] = []
        loaded_files = 0
        chunks: list[Document] = []

        for file_path in supported_files:
            try:
                inferred_subject, inferred_chapter = self._infer_subject_and_chapter(file_path, source_dir)
                normalized_inferred_subject = normalize_key(inferred_subject)
                normalized_inferred_chapter = normalize_key(inferred_chapter)

                if requested_subject and normalized_inferred_subject != requested_subject:
                    continue
                if requested_chapter and normalized_inferred_chapter != requested_chapter:
                    continue

                metadata_subject = payload.subject or inferred_subject
                metadata_chapter = payload.chapter or inferred_chapter

                file_documents = self._load_file(
                    file_path,
                    payload.class_num,
                    metadata_subject,
                    metadata_chapter,
                )
                chunks.extend(
                    self._chunk_file_documents(
                        file_documents=file_documents,
                        file_path=file_path,
                        class_num=payload.class_num,
                        subject=metadata_subject,
                        chapter=metadata_chapter,
                    )
                )
                loaded_files += 1
            except Exception:
                skipped_files.append(str(file_path))

        if not chunks:
            return IngestResponse(
                source_dir=str(source_dir),
                files_processed=0,
                chunks_indexed=0,
                collection_size=self.vectorstore.count_documents(),
                skipped_files=skipped_files,
                note="No supported documents were indexed.",
            )

        chunk_ids = [self._make_chunk_id(document) for document in chunks]
        self.vectorstore.add_documents(chunks, ids=chunk_ids)

        return IngestResponse(
            source_dir=str(source_dir),
            files_processed=loaded_files,
            chunks_indexed=len(chunks),
            collection_size=self.vectorstore.count_documents(),
            skipped_files=skipped_files,
            note=None,
        )

    def _load_file(
        self,
        file_path: Path,
        class_num: int,
        subject: str,
        chapter: str,
    ) -> list[Document]:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        else:
            documents = self._load_text_file(file_path)

        normalized_subject = normalize_key(subject)
        normalized_chapter = normalize_key(chapter)

        enriched_documents: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "class_num": class_num,
                    "subject": normalized_subject,
                    "chapter": normalized_chapter,
                    "source_path": str(file_path),
                    "file_name": file_path.name,
                    "document_type": file_path.suffix.lower().lstrip("."),
                }
            )
            enriched_documents.append(Document(page_content=document.page_content, metadata=metadata))

        return enriched_documents

    def _chunk_file_documents(
        self,
        *,
        file_documents: list[Document],
        file_path: Path,
        class_num: int,
        subject: str,
        chapter: str,
    ) -> list[Document]:
        chapter_title = self._chapter_title_for_file(file_path, chapter)
        normalized_subject = normalize_key(subject)
        normalized_chapter = normalize_key(chapter)
        chunk_documents: list[Document] = []

        for page_document in file_documents:
            page_text = self._normalize_text(page_document.page_content)
            if not page_text:
                continue

            page_blocks = self._extract_hierarchical_blocks(page_text, chapter_title)
            if not page_blocks:
                page_blocks = [ChunkBlock(heading_path=[chapter_title], level=1, text=page_text)]

            for block in page_blocks:
                semantic_chunks = self._semantic_chunks(block.text)
                for chunk_index, chunk_text in enumerate(semantic_chunks):
                    enriched_text = self._attach_heading_context(block.heading_path, chunk_text)
                    metadata = dict(page_document.metadata)
                    metadata.update(
                        {
                            "class_num": class_num,
                            "subject": normalized_subject,
                            "chapter": normalized_chapter,
                            "source_path": str(file_path),
                            "file_name": file_path.name,
                            "document_type": file_path.suffix.lower().lstrip("."),
                            "chunk_index": chunk_index,
                            "chunk_level": block.level,
                            "section_path": " > ".join(block.heading_path),
                            "chunk_strategy": "hierarchical-semantic",
                        }
                    )
                    if len(block.heading_path) > 1:
                        metadata["topic"] = block.heading_path[1]
                    if len(block.heading_path) > 2:
                        metadata["subtopic"] = block.heading_path[2]
                    chunk_documents.append(Document(page_content=enriched_text, metadata=metadata))

        return chunk_documents

    def _chapter_title_for_file(self, file_path: Path, chapter: str) -> str:
        if chapter:
            return humanize_key(chapter)
        return humanize_key(file_path.stem)

    def _extract_hierarchical_blocks(self, text: str, chapter_title: str) -> list[ChunkBlock]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
        if not paragraphs:
            return []

        heading_stack: list[tuple[int, str]] = [(1, chapter_title)]
        current_level = 1
        current_body: list[str] = []
        blocks: list[ChunkBlock] = []

        for paragraph in paragraphs:
            heading_level = self._detect_heading_level(paragraph, chapter_title)
            if heading_level:
                if current_body:
                    blocks.append(
                        ChunkBlock(
                            heading_path=[heading for _, heading in heading_stack],
                            level=current_level,
                            text="\n\n".join(current_body).strip(),
                        )
                    )
                    current_body = []

                heading_text = self._clean_heading_text(paragraph)
                heading_stack = self._update_heading_stack(heading_stack, heading_text, heading_level, chapter_title)
                current_level = heading_level
                continue

            current_body.append(paragraph)

        if current_body:
            blocks.append(
                ChunkBlock(
                    heading_path=[heading for _, heading in heading_stack],
                    level=current_level,
                    text="\n\n".join(current_body).strip(),
                )
            )

        return [block for block in blocks if block.text]

    def _update_heading_stack(
        self,
        heading_stack: list[tuple[int, str]],
        heading_text: str,
        heading_level: int,
        chapter_title: str,
    ) -> list[tuple[int, str]]:
        normalized_heading = normalize_key(heading_text)
        normalized_chapter = normalize_key(chapter_title)

        if heading_level <= 1:
            new_stack: list[tuple[int, str]] = [(1, chapter_title)]
            if normalized_heading and normalized_heading != normalized_chapter:
                new_stack.append((1, heading_text))
            return new_stack

        while len(heading_stack) > 1 and heading_stack[-1][0] >= heading_level:
            heading_stack.pop()

        heading_stack.append((heading_level, heading_text))
        return heading_stack

    def _detect_heading_level(self, paragraph: str, chapter_title: str) -> int:
        cleaned = self._normalize_text(paragraph)
        if not cleaned or len(cleaned) > 140:
            return 0

        normalized = normalize_key(cleaned)
        chapter_normalized = normalize_key(chapter_title)
        if normalized and normalized == chapter_normalized:
            return 1

        lowered = cleaned.lower()
        if re.match(r"^(chapter|unit|lesson)\s+\d+\b", lowered):
            return 1

        number_match = re.match(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+.+", cleaned)
        if number_match:
            number_depth = number_match.group(1).count(".")
            return min(3, number_depth + 2)

        if cleaned.isupper() and len(cleaned.split()) <= 12:
            return 2

        major_heading_prefixes = (
            "exercise",
            "summary",
            "key words",
            "key points",
            "activities",
            "activity",
            "let us",
            "what we have learned",
            "recall",
            "review questions",
        )
        if any(lowered.startswith(prefix) for prefix in major_heading_prefixes):
            return 2

        words = cleaned.split()
        if len(words) <= 10 and not cleaned.endswith((".", "?", "!")):
            title_like = sum(1 for word in words if word[:1].isupper()) / max(len(words), 1)
            if title_like >= 0.7:
                return 2 if len(words) > 2 else 1

        return 0

    def _semantic_chunks(self, text: str) -> list[str]:
        cleaned = self._normalize_text(text)
        if not cleaned:
            return []
        # Prefer token-based chunking when encoder is available
        if self._tiktoken_encoding is not None:
            target_tokens = getattr(self.settings, "chunk_size_tokens", None) or int(self.settings.chunk_size / 4)
            overlap_tokens = getattr(self.settings, "chunk_overlap_tokens", None) or int(self.settings.chunk_overlap / 4)

            # quick path if small
            if self._num_tokens(cleaned) <= target_tokens:
                return [cleaned]

            units = self._split_into_semantic_units(cleaned)
            if not units:
                # fallback to token chunking entire cleaned text
                return [chunk for chunk in self._split_text_by_tokens(cleaned, target_tokens, overlap_tokens)]

            chunks: list[str] = []
            current_units: list[str] = []
            current_tokens = 0

            for unit in units:
                unit = unit.strip()
                if not unit:
                    continue

                unit_tokens = self._num_tokens(unit)
                if unit_tokens > target_tokens:
                    # flush current
                    if current_units:
                        chunks.append(self._join_units(current_units))
                        current_units = self._carry_overlap_tokens(current_units, overlap_tokens)
                        current_tokens = self._num_tokens(self._join_units(current_units)) if current_units else 0

                    # split the large unit directly by tokens
                    chunks.extend([c for c in self._split_text_by_tokens(unit, target_tokens, overlap_tokens)])
                    continue

                # if adding this unit would exceed token budget, flush current
                if current_units and (current_tokens + unit_tokens) > target_tokens:
                    chunks.append(self._join_units(current_units))
                    current_units = self._carry_overlap_tokens(current_units, overlap_tokens)
                    current_tokens = self._num_tokens(self._join_units(current_units)) if current_units else 0

                current_units.append(unit)
                current_tokens += unit_tokens

            if current_units:
                chunks.append(self._join_units(current_units))

            return [chunk.strip() for chunk in chunks if chunk.strip()]

        # Fallback: character-based behavior
        target_size = self.settings.chunk_size
        if len(cleaned) <= target_size:
            return [cleaned]

        units = self._split_into_semantic_units(cleaned)
        if not units:
            return [chunk.strip() for chunk in self._fallback_splitter.split_text(cleaned) if chunk.strip()]

        chunks: list[str] = []
        current_units: list[str] = []

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            if len(unit) > target_size:
                if current_units:
                    chunks.append(self._join_units(current_units))
                    current_units = self._carry_overlap(current_units)

                chunks.extend(chunk.strip() for chunk in self._fallback_splitter.split_text(unit) if chunk.strip())
                continue

            candidate_units = current_units + [unit]
            if current_units and len(self._join_units(candidate_units)) > target_size:
                chunks.append(self._join_units(current_units))
                current_units = self._carry_overlap(current_units)

            current_units.append(unit)

        if current_units:
            chunks.append(self._join_units(current_units))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_into_semantic_units(self, text: str) -> list[str]:
        paragraph_candidates = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
        units: list[str] = []

        for paragraph in paragraph_candidates:
            line_candidates = [line.strip() for line in paragraph.split("\n") if line.strip()]
            if len(line_candidates) > 1:
                units.extend(line_candidates)
                continue

            sentence_candidates = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
            if len(sentence_candidates) > 1:
                units.extend(sentence_candidates)
                continue

            clause_candidates = [clause.strip() for clause in re.split(r"(?<=[;:])\s+|,\s+", paragraph) if clause.strip()]
            if len(clause_candidates) > 1:
                units.extend(clause_candidates)
                continue

            units.append(paragraph)

        return units

    def _carry_overlap(self, units: list[str]) -> list[str]:
        # If token encoder available, use token-based overlap. Otherwise keep char-based.
        overlap_limit = getattr(self.settings, "chunk_overlap_tokens", None) if self._tiktoken_encoding is not None else self.settings.chunk_overlap
        if overlap_limit is None:
            overlap_limit = self.settings.chunk_overlap

        if overlap_limit <= 0:
            return []

        if self._tiktoken_encoding is not None:
            # token-based carry; reuse helper
            return self._carry_overlap_tokens(units, overlap_limit)

        carried: list[str] = []
        total_length = 0
        for unit in reversed(units):
            unit_length = len(unit)
            if carried and total_length + unit_length + 2 > overlap_limit:
                break
            carried.append(unit)
            total_length += unit_length + 2
            if total_length >= overlap_limit:
                break

        return list(reversed(carried))

    def _carry_overlap_tokens(self, units: list[str], overlap_limit_tokens: int) -> list[str]:
        # choose units from the end until we reach overlap token budget
        carried: list[str] = []
        total_tokens = 0
        for unit in reversed(units):
            unit_tokens = self._num_tokens(unit)
            if carried and total_tokens + unit_tokens > overlap_limit_tokens:
                break
            carried.append(unit)
            total_tokens += unit_tokens
            if total_tokens >= overlap_limit_tokens:
                break

        return list(reversed(carried))

    def _num_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tiktoken_encoding is None:
            # approximate 4 chars per token
            return max(1, int(len(text) / 4))
        try:
            return len(self._tiktoken_encoding.encode(text))
        except Exception:
            return max(1, int(len(text) / 4))

    def _split_text_by_tokens(self, text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
        """Split text into decoded token chunks using tiktoken when available.
        Falls back to character splitter if encoding is not available."""
        if self._tiktoken_encoding is None:
            return [chunk.strip() for chunk in self._fallback_splitter.split_text(text) if chunk.strip()]

        try:
            tokens = self._tiktoken_encoding.encode(text)
            chunks: list[str] = []
            step = max(1, target_tokens - overlap_tokens)
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + target_tokens]
                try:
                    chunk_text = self._tiktoken_encoding.decode(chunk_tokens)
                except Exception:
                    # decoding might fail on partial tokens; fallback to joining approx
                    chunk_text = text
                chunks.append(chunk_text)
            return chunks
        except Exception:
            return [chunk.strip() for chunk in self._fallback_splitter.split_text(text) if chunk.strip()]

    def _join_units(self, units: list[str]) -> str:
        return "\n\n".join(unit.strip() for unit in units if unit.strip())

    def _attach_heading_context(self, heading_path: list[str], text: str) -> str:
        header = " > ".join(heading_path)
        if not header:
            return text.strip()
        return f"{header}\n\n{text.strip()}"

    def _clean_heading_text(self, paragraph: str) -> str:
        cleaned = self._normalize_text(paragraph)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-\t")
        return cleaned

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\r", "\n").strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _load_text_file(self, file_path: Path) -> list[Document]:
        for encoding in ("utf-8", "latin-1"):
            try:
                loader = TextLoader(str(file_path), encoding=encoding)
                return loader.load()
            except Exception:
                continue
        raise ValueError(f"Could not load text file: {file_path}")

    def _infer_subject_and_chapter(self, file_path: Path, source_root: Path) -> tuple[str, str]:
        relative = file_path.relative_to(source_root)
        parts = relative.parts

        # Common layout: class-X/subject/chapter N.pdf or class-X/subject/chapter-folder/file.pdf
        inferred_subject = parts[0] if len(parts) > 1 else file_path.parent.name

        if len(parts) >= 3:
            inferred_chapter = parts[1]
        elif len(parts) == 2:
            inferred_chapter = file_path.stem
        else:
            inferred_chapter = file_path.stem

        return inferred_subject, inferred_chapter

    def _make_chunk_id(self, document: Document) -> str:
        raw_value = "|".join(
            [
                str(document.metadata.get("source_path", "")),
                str(document.metadata.get("page", "")),
                str(document.metadata.get("section_path", "")),
                str(document.metadata.get("chunk_index", "")),
                document.page_content[:512],
            ]
        )
        return hashlib.sha1(raw_value.encode("utf-8")).hexdigest()
