from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.utils import extract_json_text, normalize_key
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FlashcardItem,
    FlashcardRequest,
    FlashcardResponse,
    QuestionItem,
    QuestionRequest,
    QuestionResponse,
    QuizItem,
    QuizOption,
    QuizRequest,
    QuizResponse,
    SourceCitation,
)
from app.services.llm import build_chat_model
from app.services.vectorstore import VectorStoreService


SYSTEM_PROMPT = """You are the NCERT Class 6 Intelligent Study Assistant.
You must answer using only the official NCERT textbook context supplied to you.
If the answer is not available in the provided context, say that clearly and do not invent facts.
Keep the language clear, student-friendly, and accurate.
Always cite only the provided sources. Never mention unsupported external knowledge.
"""


@dataclass(slots=True)
class RetrievedBundle:
    documents: list[Document]
    citations: list[SourceCitation]
    scope: str
    not_in_textbook: bool


class RagService:
    def __init__(self, settings: Settings, vectorstore: VectorStoreService) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.chat_model = build_chat_model(settings, num_predict=384)
        self.quiz_chat_model = build_chat_model(settings, model_name=settings.quiz_ollama_model, num_predict=512)

    def answer_question(self, payload: ChatRequest) -> ChatResponse:
        bundle = self._retrieve(payload.class_num, payload.subject, payload.chapter, payload.question, payload.top_k)
        if not bundle.documents:
            return ChatResponse(
                answer="I could not find this in the official NCERT Class 6 textbook corpus.",
                citations=[],
                not_in_textbook=True,
                retrieved_documents=0,
                source_scope=bundle.scope,
            )

        context = self._format_context(bundle.documents)
        history = self._format_history(payload.chat_history)
        prompt = f"""{SYSTEM_PROMPT}

Conversation history:
{history}

Official NCERT context:
{context}

Student question:
{payload.question}

Answer in 3 to 6 short paragraphs or bullet points. If the context is insufficient, say so explicitly.
Return plain text only.
"""

        try:
            answer = self.chat_model.invoke(prompt).content.strip()
        except Exception as exc:
            answer = f"I could not generate an answer because the local Ollama model returned an error: {exc}"

        not_in_textbook = self._is_out_of_scope(answer, bundle.not_in_textbook)
        if not_in_textbook and "could not find" not in answer.lower():
            answer = "I could not find this in the official NCERT Class 6 textbook corpus."

        return ChatResponse(
            answer=answer,
            citations=bundle.citations,
            not_in_textbook=not_in_textbook,
            retrieved_documents=len(bundle.documents),
            source_scope=bundle.scope,
        )

    def generate_flashcards(self, payload: FlashcardRequest) -> FlashcardResponse:
        bundle = self._retrieve(payload.class_num, payload.subject, payload.chapter, payload.focus_area or "Generate flashcards from the chapter.", payload.top_k)
        if not bundle.documents:
            return FlashcardResponse(
                flashcards=[],
                citations=[],
                notes=["No official NCERT content was found for this selection."],
                not_in_textbook=True,
            )

        context = self._format_context(bundle.documents)
        prompt = f"""{SYSTEM_PROMPT}

Create exactly {payload.count} flashcards from the official NCERT context below.
Use only the provided context.
If the context is limited, produce the best possible set without inventing facts.
Return valid JSON with this shape:
{{
  "flashcards": [
    {{"front": "...", "back": "...", "explanation": "..."}}
  ],
  "notes": ["..."]
}}

Official NCERT context:
{context}

Topic focus:
{payload.focus_area or "Whole chapter"}
"""

        try:
            parsed = self._invoke_json(prompt, timeout_seconds=20)
        except Exception:
            parsed = {}
        flashcards = [FlashcardItem.model_validate(item) for item in parsed.get("flashcards", [])]
        notes = [str(note) for note in parsed.get("notes", [])]
        return FlashcardResponse(
            flashcards=flashcards,
            citations=bundle.citations,
            notes=notes,
            not_in_textbook=bundle.not_in_textbook and not flashcards,
        )

    def generate_questions(self, payload: QuestionRequest) -> QuestionResponse:
        is_math = self._is_math_subject(payload.subject)
        bundle = self._retrieve(
            payload.class_num,
            payload.subject,
            payload.chapter,
            "End-of-chapter exercise questions and answers from this NCERT chapter.",
            payload.top_k,
        )
        if not bundle.documents:
            return QuestionResponse(
                questions=[],
                citations=[],
                notes=["No official NCERT content was found for this selection."],
                not_in_textbook=True,
            )

        context = self._format_context(bundle.documents)
        math_instruction = (
            "For Mathematics, generate mostly problem-solving questions from chapter exercises (about 70%) and keep the rest concept-based (about 30%)."
            if is_math
            else ""
        )
        prompt = f"""{SYSTEM_PROMPT}

Create exactly {payload.count} important questions with answers from the official NCERT context.
Priority rule: pick questions from the end-of-chapter exercise section of the selected chapter.
If the exercise section is partially available, use only whatever is available there before using any other chapter text.
Every question and answer must be grounded in the provided context. Do not invent values, facts, or formulas.
{math_instruction}
Return valid JSON with this shape:
{{
  "questions": [
    {{"question": "...", "answer": "...", "explanation": "...", "difficulty": "easy|medium|hard"}}
  ],
  "notes": ["..."]
}}

Official NCERT context:
{context}
"""

        try:
            parsed = self._invoke_json(prompt, timeout_seconds=20)
        except Exception:
            parsed = {}
        questions = [self._coerce_question_item(item) for item in parsed.get("questions", [])]
        if not questions:
            questions = self._fallback_questions(bundle.documents, payload.count, is_math)
        notes = [str(note) for note in parsed.get("notes", [])]
        return QuestionResponse(
            questions=questions,
            citations=bundle.citations,
            notes=notes,
            not_in_textbook=bundle.not_in_textbook and not questions,
        )

    def generate_quiz(self, payload: QuizRequest) -> QuizResponse:
        is_math = self._is_math_subject(payload.subject)
        bundle = self._retrieve(
            payload.class_num,
            payload.subject,
            payload.chapter,
            "End-of-chapter exercise questions for quiz generation from this NCERT chapter.",
            payload.top_k,
        )
        if not bundle.documents:
            return QuizResponse(
                quiz_title="NCERT Class 6 Quiz",
                questions=[],
                citations=[],
                notes=["No official NCERT content was found for this selection."],
                not_in_textbook=True,
            )

        context = self._format_context(bundle.documents)
        math_instruction = (
            "For Mathematics, include mostly numerical/problem-solving items from exercises (about 70%) and the rest concept-based questions (about 30%)."
            if is_math
            else ""
        )
        # Force MCQ-only generation for all subjects
        prompt = f"""{SYSTEM_PROMPT}

Create exactly {payload.count} multiple choice quiz questions (MCQs only) from the official NCERT context.
Difficulty: {payload.difficulty}
Priority rule: source questions from the end-of-chapter exercise section of the selected chapter.
Every question must be answerable from the provided context only.
{math_instruction}
Return valid JSON with this shape:
{{
  "quiz_title": "...",
  "questions": [
    {{
      "question": "...",
      "question_type": "mcq",
      "options": [{{"label": "A", "text": "..."}}, {{"label": "B", "text": "..."}}, {{"label": "C", "text": "..."}}, {{"label": "D", "text": "..."}}],
      "correct_answer": "...",
      "explanation": "...",
      "difficulty": "easy|medium|hard"
    }}
  ],
  "notes": ["..."]
}}

Ensure all questions have exactly four options labeled A, B, C, and D with one correct answer.

Official NCERT context:
{context}
"""

        try:
            parsed = self._invoke_json(prompt, model=self.quiz_chat_model, timeout_seconds=20)
        except Exception:
            parsed = {}
        raw_items = parsed.get("questions", [])
        questions: list[QuizItem] = []
        for item in raw_items:
            try:
                # Force MCQ type for all subjects
                questions.append(self._coerce_quiz_item(item, "mcq", payload.difficulty))
            except Exception:
                continue

        # Ensure uniqueness and prefer the requested difficulty
        questions = self._dedupe_questions(questions)
        if questions:
            questions = self._select_by_difficulty(questions, payload.count, payload.difficulty)

        if not questions:
            # Force MCQ type in fallback as well
            questions = self._fallback_quiz(bundle.documents, payload.count, "mcq", payload.difficulty, is_math)

        # Final enforcement: ensure all questions are MCQ
        questions = self._enforce_mcq_only(questions)

        notes = [str(note) for note in parsed.get("notes", [])]
        quiz_title = str(parsed.get("quiz_title") or "NCERT Class 6 Quiz")
        return QuizResponse(
            quiz_title=quiz_title,
            questions=questions,
            citations=bundle.citations,
            notes=notes,
            not_in_textbook=bundle.not_in_textbook and not questions,
        )

    def _retrieve(self, class_num: int, subject: str | None, chapter: str | None, query: str, top_k: int) -> RetrievedBundle:
        metadata_filter: dict[str, object] = {"class_num": class_num}
        if subject:
            metadata_filter["subject"] = normalize_key(subject)
        if chapter:
            metadata_filter["chapter"] = normalize_key(chapter)

        results = self.vectorstore.similarity_search_with_scores(query, k=top_k, filter=metadata_filter)
        documents: list[Document] = []
        citations: list[SourceCitation] = []
        for document, score in results:
            documents.append(document)
            citations.append(self._citation_from_document(document, score))

        best_score = max((score for _, score in results), default=0.0)
        not_in_textbook = best_score < self.settings.answer_relevance_threshold
        scope_bits = [f"class {class_num}"]
        if subject:
            scope_bits.append(normalize_key(subject))
        if chapter:
            scope_bits.append(normalize_key(chapter))

        return RetrievedBundle(documents=documents, citations=citations, scope=" / ".join(scope_bits), not_in_textbook=not_in_textbook)

    def _format_context(self, documents: list[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            title_bits = [str(metadata.get("file_name", "unknown source"))]
            if metadata.get("subject"):
                title_bits.append(f"subject={metadata['subject']}")
            if metadata.get("chapter"):
                title_bits.append(f"chapter={metadata['chapter']}")
            if metadata.get("page") is not None:
                title_bits.append(f"page={metadata['page']}")
            header = "; ".join(title_bits)
            blocks.append(f"[{index}] {header}\n{document.page_content.strip()}")
        return "\n\n".join(blocks)

    def _format_history(self, messages: list[ChatMessage]) -> str:
        if not messages:
            return "(no prior messages)"
        return "\n".join(f"{message.role}: {message.content}" for message in messages[-8:])

    def _invoke_json(self, prompt: str, model=None, timeout_seconds: float | None = None) -> dict:
        active_model = model or self.chat_model
        try:
            if timeout_seconds is None:
                response = active_model.invoke(prompt).content
            else:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(lambda: active_model.invoke(prompt).content)
                try:
                    response = future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return {}
                except Exception:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return {}
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            return {}
        payload = extract_json_text(response)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def _citation_from_document(self, document: Document, score: float) -> SourceCitation:
        metadata = document.metadata
        return SourceCitation(
            source_path=str(metadata.get("source_path", "")),
            file_name=str(metadata.get("file_name", "")),
            class_num=self._safe_int(metadata.get("class_num")),
            subject=str(metadata.get("subject", "")) or None,
            chapter=str(metadata.get("chapter", "")) or None,
            page=self._safe_int(metadata.get("page")),
            chunk_id=self._safe_int(metadata.get("chunk_id")),
            relevance_score=float(score),
            excerpt=document.page_content[:350].strip(),
        )

    def _coerce_quiz_item(self, item: dict, default_type: str, default_difficulty: str) -> QuizItem:
        question_type = self._normalize_question_type(item.get("question_type"), default_type)
        if default_type == "mcq":
            question_type = "mcq"
        raw_options = item.get("options", [])
        options = self._normalize_quiz_options(raw_options, question_type)
        difficulty = self._normalize_difficulty(item.get("difficulty"), default_difficulty)

        correct_answer = str(item.get("correct_answer", "")).strip()
        if not correct_answer and question_type == "true_false":
            correct_answer = "True"
        # If MCQ but options are missing or placeholders, build sensible fallback options
        if question_type == "mcq":
            invalid = len(options) < 4 or any(self._is_placeholder_option(opt.text) for opt in options)
            if invalid:
                # Prefer using provided correct_answer; if missing, use explanation or question as source
                answer_source = correct_answer or str(item.get("explanation", "")).strip() or str(item.get("question", "")).strip()
                options = self._build_fallback_options(answer_source, str(item.get("question", "")))

        # Normalize correct_answer to option label when possible
        if question_type == "mcq" and options:
            ca = correct_answer.strip()
            # If already a label like 'A', 'B', etc., normalize
            if ca.upper() in {"A", "B", "C", "D"}:
                correct_answer = ca.upper()
            else:
                # Try to match by text (exact or substring)
                matched_label = ""
                for opt in options:
                    if not opt.text:
                        continue
                    if opt.text.strip().lower() == ca.lower():
                        matched_label = opt.label
                        break
                if not matched_label:
                    for opt in options:
                        if ca.lower() in opt.text.strip().lower() or opt.text.strip().lower() in ca.lower():
                            matched_label = opt.label
                            break
                if matched_label:
                    correct_answer = matched_label

        return QuizItem(
            question=str(item.get("question", "")).strip(),
            question_type=question_type,  # type: ignore[arg-type]
            options=options,
            correct_answer=correct_answer,
            explanation=str(item.get("explanation", "")).strip(),
            difficulty=difficulty,
        )

    def _coerce_question_item(self, item: dict) -> QuestionItem:
        difficulty = self._normalize_difficulty(item.get("difficulty"), "medium")
        return QuestionItem(
            question=str(item.get("question", "")).strip(),
            answer=str(item.get("answer", "")).strip(),
            explanation=str(item.get("explanation", "")).strip(),
            difficulty=difficulty,
        )

    def _normalize_difficulty(self, value: object, fallback: str) -> str:
        candidate = str(value or fallback).strip().lower()
        if candidate not in {"easy", "medium", "hard"}:
            return "medium"
        return candidate

    def _normalize_question_type(self, value: object, fallback: str) -> str:
        candidate = str(value or fallback).strip().lower()
        if candidate == "mixed":
            candidate = fallback if fallback in {"mcq", "true_false", "short_answer"} else "mcq"
        if candidate not in {"mcq", "true_false", "short_answer"}:
            return "mcq"
        return candidate

    def _fallback_questions(self, documents: list[Document], count: int, is_math: bool) -> list[QuestionItem]:
        pairs = self._extract_exercise_pairs(documents)
        questions: list[QuestionItem] = []
        for index, pair in enumerate(pairs[:count], start=1):
            question_text = pair[0]
            answer_text = pair[1]
            explanation = "Based on the end-of-chapter exercise section in the selected NCERT chapter."
            if is_math:
                explanation = "Solve the problem using the chapter exercise method shown in the NCERT text."
            questions.append(
                QuestionItem(
                    question=question_text,
                    answer=answer_text,
                    explanation=explanation,
                    difficulty="medium",
                )
            )

        if questions:
            return questions

        context_lines = self._first_context_sentences(documents, count)
        for line in context_lines:
            questions.append(
                QuestionItem(
                    question=f"What does the chapter say about: {line}?",
                    answer=line,
                    explanation="Grounded in the selected chapter context.",
                    difficulty="easy",
                )
            )
        return questions[:count]

    def _fallback_quiz(self, documents: list[Document], count: int, quiz_type: str, difficulty: str, is_math: bool) -> list[QuizItem]:
        pairs = self._extract_exercise_pairs(documents)
        questions: list[QuizItem] = []
        for question_text, answer_text in pairs[:count]:
            question_type = quiz_type if quiz_type in {"mcq", "true_false", "short_answer"} else "mcq"
            if is_math:
                question_type = "short_answer" if quiz_type == "short_answer" else question_type

            options = self._build_fallback_options(answer_text, question_text) if question_type == "mcq" else []
            correct_answer = answer_text.strip() if question_type != "true_false" else self._normalize_true_false_answer(answer_text)
            if question_type == "true_false" and correct_answer not in {"True", "False"}:
                correct_answer = "True"

            questions.append(
                QuizItem(
                    question=question_text,
                    question_type=question_type,  # type: ignore[arg-type]
                    options=options,
                    correct_answer=correct_answer,
                    explanation="Based on the end-of-chapter exercise section in the selected NCERT chapter.",
                    difficulty=self._normalize_difficulty(difficulty, "medium"),
                )
            )

        if questions:
            return questions

        # If no exercise pairs found, generate MCQ fallback questions
        fallback_lines = self._first_context_sentences(documents, count)
        for line in fallback_lines:
            # Force MCQ generation even in fallback
            options = self._build_fallback_options(line, "")
            questions.append(
                QuizItem(
                    question=f"According to the chapter, what is the important concept: {line}?",
                    question_type="mcq",
                    options=options,
                    correct_answer="A",
                    explanation="Grounded in the selected chapter context.",
                    difficulty=self._normalize_difficulty(difficulty, "medium"),
                )
            )
        return questions[:count]

    def _extract_exercise_pairs(self, documents: list[Document]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for document in documents:
            text = document.page_content.replace("\r", "\n")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            current_question: str | None = None
            current_answer: str | None = None

            for line in lines:
                question_match = re.match(r"^(?:Q\.?|Question\s*\d+\.?|Exercise\s*\d+\.?)(.*)$", line, flags=re.IGNORECASE)
                answer_match = re.match(r"^(?:Ans\.?|Answer\.?)(.*)$", line, flags=re.IGNORECASE)

                if question_match:
                    if current_question and current_answer:
                        pairs.append((current_question.strip(), current_answer.strip()))
                    current_question = question_match.group(1).strip(" :-\t") or line.strip()
                    current_answer = None
                    continue

                if answer_match and current_question:
                    current_answer = answer_match.group(1).strip(" :-\t") or line.strip()
                    continue

                if current_question and current_answer is None and len(line) < 220:
                    current_question = f"{current_question} {line}".strip()
                elif current_question and current_answer is not None and len(line) < 220:
                    current_answer = f"{current_answer} {line}".strip()

            if current_question and current_answer:
                pairs.append((current_question.strip(), current_answer.strip()))

        # Prefer actual exercise-style question/answer pairs and deduplicate them.
        seen: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str]] = []
        for question, answer in pairs:
            normalized = (question.strip(), answer.strip())
            if not normalized[0] or not normalized[1] or normalized in seen:
                continue
            seen.add(normalized)
            unique_pairs.append(normalized)
        return unique_pairs

    def _first_context_sentences(self, documents: list[Document], count: int) -> list[str]:
        snippets: list[str] = []
        for document in documents:
            text = re.sub(r"\s+", " ", document.page_content).strip()
            if not text:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                if len(sentence) < 12:
                    continue
                snippets.append(sentence[:120])
                if len(snippets) >= count:
                    return snippets
        return snippets

    def _build_fallback_options(self, answer_text: str, question_text: str) -> list[QuizOption]:
        correct = self._clean_option_text(answer_text)[:120] or "Correct answer"

        numeric_options = self._build_numeric_distractors(correct)
        if numeric_options:
            return numeric_options

        similar_distractors = self._build_similar_distractors(correct, question_text)
        options = [
            QuizOption(label="A", text=correct),
            QuizOption(label="B", text=similar_distractors[0]),
            QuizOption(label="C", text=similar_distractors[1]),
            QuizOption(label="D", text=similar_distractors[2]),
        ]
        return self._dedupe_and_relabel_options(options)

    def _is_placeholder_option(self, text: str) -> bool:
        if not text:
            return True
        lowered = text.strip().lower()
        placeholders = [
            "enter",
            "your answer",
            "type",
            "option",
            "write",
            "fill",
        ]
        # treat very short generic labels as placeholders too
        if len(lowered) < 3:
            return True
        return any(p in lowered for p in placeholders)

    def _mutate_text(self, text: str) -> str:
        t = text.strip()
        if not t:
            return "An incorrect statement"
        parts = t.split()
        if len(parts) > 2:
            # swap two middle words to create a plausible distractor
            i = max(1, len(parts) // 3)
            j = min(len(parts) - 2, i + 1)
            parts[i], parts[j] = parts[j], parts[i]
            return " ".join(parts)[:80]
        # if single/short phrase, append a small negation
        if len(t) < 30:
            return f"{t} (not correct)"
        return t[:80]

    def _normalize_true_false_answer(self, answer_text: str) -> str:
        lowered = answer_text.strip().lower()
        if lowered.startswith(("true", "yes", "correct")):
            return "True"
        if lowered.startswith(("false", "no", "incorrect")):
            return "False"
        return "True"

    def _normalize_quiz_options(self, raw_options: object, question_type: str) -> list[QuizOption]:
        if question_type != "mcq":
            return []
        options: list[QuizOption] = []
        # Accept several incoming shapes: list[dict], list[str], comma-separated string, single dict
        if isinstance(raw_options, list):
            for option in raw_options:
                if isinstance(option, dict):
                    label = str(option.get("label", "")).strip().upper()[:1]
                    text = self._clean_option_text(str(option.get("text", "")))
                else:
                    # treat non-dict as plain text option
                    label = ""
                    text = self._clean_option_text(str(option))

                if not text:
                    continue
                if label in {"A", "B", "C", "D"}:
                    options.append(QuizOption(label=label, text=text))
                else:
                    options.append(QuizOption(label="", text=text))

        elif isinstance(raw_options, dict):
            # try to pull A-D keys or convert dict values
            for key in ["A", "B", "C", "D", "a", "b", "c", "d"]:
                if key in raw_options:
                    text = self._clean_option_text(str(raw_options[key]))
                    if text:
                        options.append(QuizOption(label=key.upper(), text=text))
            if not options:
                # fallback: treat dict values as possible options
                for val in raw_options.values():
                    text = self._clean_option_text(str(val))
                    if text:
                        options.append(QuizOption(label="", text=text))

        elif isinstance(raw_options, str):
            # comma or semicolon separated
            parts = re.split(r"[,;\n]", raw_options)
            for part in parts:
                part = part.strip()
                if part:
                    options.append(QuizOption(label="", text=part))

        # Clean up placeholders and assign labels A-D
        clean_texts: list[str] = []
        for opt in options:
            if not opt.text or self._is_placeholder_option(opt.text):
                continue
            clean_texts.append(opt.text)

        # If we have >=4 cleaned options, label and return first 4
        if len(clean_texts) >= 4:
            labeled = [QuizOption(label=lab, text=txt) for lab, txt in zip(["A", "B", "C", "D"], clean_texts[:4])]
            return self._dedupe_and_relabel_options(labeled)

        # Otherwise build additional distractors from available data
        labeled: list[QuizOption] = []
        for i, txt in enumerate(clean_texts[:4]):
            labeled.append(QuizOption(label=["A", "B", "C", "D"][i], text=txt))

        # Pad using similar but distinct distractors.
        seed = clean_texts[0] if clean_texts else self._clean_option_text(question_text) or "Correct answer"
        candidates = [seed]
        candidates.extend(self._build_similar_distractors(seed, question_text))
        candidates.extend([
            f"Related but not exact: {seed}",
            f"A closely related idea to {seed}",
            f"A partial version of {seed}",
        ])

        seen = {seed.lower()}
        while len(labeled) < 4 and candidates:
            cand = self._clean_option_text(candidates.pop(0))
            if not cand:
                continue
            key = cand.lower()
            if key in seen or self._is_placeholder_option(cand):
                continue
            seen.add(key)
            labeled.append(QuizOption(label=["A", "B", "C", "D"][len(labeled)], text=cand))

        while len(labeled) < 4:
            labeled.append(QuizOption(label=["A", "B", "C", "D"][len(labeled)], text=f"A similar but incorrect option {len(labeled) + 1}"))

        return self._dedupe_and_relabel_options(labeled)

    def _clean_option_text(self, text: str) -> str:
        cleaned = str(text).strip()
        cleaned = re.sub(r"^\s*[A-Da-d]\s*[\).:-]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*(?:option|answer)\s*[A-Da-d]?\s*[\).:-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _build_numeric_distractors(self, correct: str) -> list[QuizOption] | None:
        number_match = re.search(r"-?\d+(?:\.\d+)?", correct)
        if not number_match:
            return None

        value_text = number_match.group(0)
        try:
            numeric_value = float(value_text)
        except ValueError:
            return None

        if numeric_value.is_integer():
            base = int(numeric_value)
            choices = [str(base), str(base + 1), str(max(base - 1, 0)), str(base + 2)]
        else:
            choices = [f"{numeric_value:g}", f"{numeric_value + 0.5:g}", f"{max(numeric_value - 0.5, 0):g}", f"{numeric_value + 1:g}"]

        unique_choices: list[str] = []
        for choice in choices:
            if choice not in unique_choices:
                unique_choices.append(choice)
        while len(unique_choices) < 4:
            unique_choices.append(f"{numeric_value + len(unique_choices) + 1:g}")

        return [
            QuizOption(label="A", text=unique_choices[0]),
            QuizOption(label="B", text=unique_choices[1]),
            QuizOption(label="C", text=unique_choices[2]),
            QuizOption(label="D", text=unique_choices[3]),
        ]

    def _build_similar_distractors(self, correct: str, question_text: str) -> list[str]:
        base = self._clean_option_text(correct)
        lower = base.lower()

        synonym_map: list[tuple[str, list[str]]] = [
            ("step-by-step", ["gradual", "systematic", "ordered"]),
            ("process", ["procedure", "method", "approach"]),
            ("experiment", ["observation", "guess", "theory"]),
            ("curiosity", ["interest", "wonder", "questioning"]),
            ("exploration", ["inspection", "study", "review"]),
            ("science", ["observation", "discovery", "learning"]),
            ("method", ["process", "procedure", "way"]),
            ("answer", ["response", "result", "solution"]),
        ]

        candidates: list[str] = []
        for source, replacements in synonym_map:
            if source in lower:
                for replacement in replacements:
                    candidate = re.sub(re.escape(source), replacement, lower, flags=re.IGNORECASE)
                    candidate = candidate[:1].upper() + candidate[1:] if candidate else candidate
                    if candidate and candidate != base:
                        candidates.append(candidate)

        if not candidates:
            tokens = base.split()
            if len(tokens) >= 3:
                variants = [tokens[:], tokens[:], tokens[:]]
                variants[0][0] = "A"
                variants[1][-1] = "approach"
                variants[2][max(0, len(tokens) // 2)] = "idea"
                candidates.extend([" ".join(v) for v in variants])

        question_fragment = self._clean_option_text(question_text)
        if question_fragment and len(candidates) < 3:
            short_fragment = question_fragment[:80]
            candidates.append(f"Similar idea from the question: {short_fragment}")

        candidates.extend([
            f"Related but not exact: {base}",
            f"A closely related idea to {base}",
            f"A partial version of {base}",
        ])

        unique: list[str] = []
        for candidate in candidates:
            cleaned = self._clean_option_text(candidate)
            if not cleaned:
                continue
            if cleaned.lower() == base.lower():
                continue
            if cleaned.lower() in {item.lower() for item in unique}:
                continue
            unique.append(cleaned)
            if len(unique) == 3:
                break

        while len(unique) < 3:
            unique.append(f"A similar but incorrect option {len(unique) + 1}")

        return unique[:3]

    def _dedupe_and_relabel_options(self, options: list[QuizOption]) -> list[QuizOption]:
        unique: list[QuizOption] = []
        seen: set[str] = set()
        for option in options:
            text = self._clean_option_text(option.text)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(QuizOption(label=option.label, text=text))

        while len(unique) < 4:
            unique.append(QuizOption(label="", text=f"A similar but incorrect option {len(unique) + 1}"))

        for index, option in enumerate(unique[:4]):
            option.label = ["A", "B", "C", "D"][index]
        return unique[:4]

    def _dedupe_questions(self, questions: list[QuizItem]) -> list[QuizItem]:
        """Remove duplicate or near-duplicate questions, preserving order."""
        seen: set[str] = set()
        unique: list[QuizItem] = []
        for q in questions:
            key = re.sub(r"\s+", " ", q.question or "").strip().lower()
            # Normalize trivial punctuation
            key = re.sub(r"[^a-z0-9 ]+", "", key)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(q)
        return unique

    def _select_by_difficulty(self, questions: list[QuizItem], count: int, difficulty: str) -> list[QuizItem]:
        """Prefer questions matching the requested difficulty, fill with nearby difficulties if needed."""
        difficulty = self._normalize_difficulty(difficulty, "medium")
        by_diff: dict[str, list[QuizItem]] = {"easy": [], "medium": [], "hard": []}
        for q in questions:
            lvl = self._normalize_difficulty(q.difficulty, "medium")
            by_diff[lvl].append(q)

        selected: list[QuizItem] = []

        # Order of preference depending on requested difficulty
        if difficulty == "easy":
            prefs = ["easy", "medium", "hard"]
        elif difficulty == "hard":
            prefs = ["hard", "medium", "easy"]
        else:
            prefs = ["medium", "easy", "hard"]

        for pref in prefs:
            for q in by_diff.get(pref, []):
                if len(selected) >= count:
                    break
                selected.append(q)
            if len(selected) >= count:
                break

        # If still short, append more from original list preserving uniqueness
        if len(selected) < count:
            for q in questions:
                if len(selected) >= count:
                    break
                if q in selected:
                    continue
                selected.append(q)

        # Trim to requested count
        return selected[:count]

    def _enforce_mcq_only(self, questions: list[QuizItem]) -> list[QuizItem]:
        """Ensure all questions are MCQ format, converting non-MCQ questions if needed."""
        enforced: list[QuizItem] = []
        for q in questions:
            try:
                if q.question_type == "mcq" and q.options and len(q.options) >= 4:
                    # Already proper MCQ, keep as is
                    enforced.append(q)
                else:
                    # Convert to MCQ by generating options from correct answer
                    if not q.correct_answer or not q.question:
                        continue
                    options = self._build_fallback_options(q.correct_answer, q.question)
                    if options and len(options) >= 4:
                        enforced_q = QuizItem(
                            question=q.question,
                            question_type="mcq",
                            options=options,
                            correct_answer="A",
                            explanation=q.explanation or "",
                            difficulty=q.difficulty or "medium",
                        )
                        enforced.append(enforced_q)
                    else:
                        # If we can't build proper options, keep original and convert type
                        enforced.append(QuizItem(
                            question=q.question,
                            question_type="mcq",
                            options=options if options else [],
                            correct_answer=q.correct_answer,
                            explanation=q.explanation or "",
                            difficulty=q.difficulty or "medium",
                        ))
            except Exception:
                # If anything fails, try to keep the question as MCQ at least
                try:
                    enforced.append(QuizItem(
                        question=q.question,
                        question_type="mcq",
                        options=q.options if q.options else [],
                        correct_answer=q.correct_answer,
                        explanation=q.explanation or "",
                        difficulty=q.difficulty or "medium",
                    ))
                except:
                    pass
        return enforced

    def _is_math_subject(self, subject: str | None) -> bool:
        normalized = normalize_key(subject or "")
        return normalized in {"maths", "mathematics"}

    def _safe_int(self, value: object) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _is_out_of_scope(self, answer: str, fallback_flag: bool) -> bool:
        if fallback_flag:
            return True
        lowered = answer.lower()
        return "could not find" in lowered or "not enough information" in lowered or "not in the provided context" in lowered
