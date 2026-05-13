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
from app.services.retrieval_pipeline import EducationalRetrievalPipeline
from app.services.vectorstore import VectorStoreService


SYSTEM_PROMPT = """You are the NCERT Intelligent Study Assistant for Indian school students.

Your role is to generate accurate educational content strictly from the provided NCERT textbook context.

CORE RULES:
1. Use ONLY the provided textbook context.
2. Never use outside knowledge, assumptions, or fabricated facts.
3. If information is missing or incomplete, explicitly say:
    "The provided NCERT context does not contain enough information."
4. Keep explanations student-friendly, concise, and educational.
5. Preserve scientific, mathematical, and factual correctness.
6. Never generate unsupported formulas, definitions, examples, or values.
7. Prioritize clarity over complexity.
8. When multiple retrieved chunks overlap, combine them carefully without repetition.
9. Maintain chapter consistency:
    - Do not mix concepts from unrelated chapters.
    - Keep terminology aligned with NCERT wording.
10. Output must strictly follow the requested format.
11. Never include markdown code fences unless explicitly requested.
12. Never explain your reasoning process.
13. For Mathematics:
    - Prefer step-by-step solutions.
    - Preserve equations exactly.
    - Avoid arithmetic mistakes.
14. For Science:
    - Preserve definitions and processes accurately.
15. For Social Science:
    - Keep dates, events, and terminology exact.
16. For Language subjects:
    - Keep interpretations grounded in the provided text only.
"""

CONCEPTUAL_QA_PROMPT = """You are an NCERT expert teacher creating concept-focused questions and answers.

Generate educationally useful conceptual questions and answers.

IMPORTANT:

* Focus on understanding and application.
* Do NOT ask vague questions like "What do you understand?" or "Write about the lesson".
* Avoid story-only questions.
* Questions should test concepts, reasoning, and applications.
* Use Bloom taxonomy.
* Questions must help students learn concepts deeply.
* Prefer educational clarity over textbook wording.
* If textbook uses examples, create new conceptual scenarios instead.

QUESTION TYPES:

* conceptual
* application-based
* reasoning
* compare and contrast
* problem-solving
* HOTS

OUTPUT FORMAT:
[
{{
"question": "",
"answer": "",
"type": "conceptual/application/HOTS",
"difficulty": "easy/medium/hard",
"blooms_level": "understand/apply/analyze"
}}
]

CONTEXT:
{context}
"""

ASSERTION_REASON_PROMPT = """You are an NCERT exam question creator.

Generate assertion-reason questions that test conceptual understanding.

RULES:

* Assertions must be meaningful.
* Reasons must test reasoning ability.
* Avoid trivial factual recall.
* Avoid copied textbook lines.
* Questions must test conceptual relationships.
* Ensure educational correctness.

OUTPUT FORMAT:
[
{{
"assertion": "",
"reason": "",
"correct_option": "",
"explanation": ""
}}
]

CONTEXT:
{context}
"""

CHAPTER_SUMMARY_PROMPT = """You are an expert NCERT revision teacher.

Generate a concept-focused educational summary.

RULES:

* Focus on concepts and learning outcomes.
* Avoid storytelling.
* Avoid unnecessary textbook examples.
* Organize concepts clearly.
* Highlight formulas, rules, and important ideas.
* Use concise educational language.
* Make summary useful for revision and exams.
* Preserve NCERT correctness.

OUTPUT FORMAT:
{{
"chapter_summary": "",
"key_concepts": [],
"important_formulas": [],
"common_mistakes": [],
"exam_points": []
}}

CONTEXT:
{context}
"""


@dataclass(slots=True)
class RetrievedBundle:
    documents: list[Document]
    citations: list[SourceCitation]
    structured_context: dict[str, list[str]]
    scope: str
    not_in_textbook: bool


class RagService:
    def __init__(self, settings: Settings, vectorstore: VectorStoreService) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.chat_model = build_chat_model(settings, num_predict=384)
        self.quiz_chat_model = build_chat_model(settings, model_name=settings.quiz_ollama_model, num_predict=512)
        self.retrieval_pipeline = EducationalRetrievalPipeline(settings, vectorstore)

    def answer_question(self, payload: ChatRequest) -> ChatResponse:
        bundle = self._retrieve("conceptual_qa", payload.class_num, payload.subject, payload.chapter, payload.question, payload.top_k)
        if not bundle.documents:
            return ChatResponse(
                answer="I could not find this in the official NCERT Class 6 textbook corpus.",
                citations=[],
                not_in_textbook=True,
                retrieved_documents=0,
                source_scope=bundle.scope,
            )

        context = self._format_structured_context(bundle.structured_context, payload.question)
        history = self._format_history(payload.chat_history)
        prompt = f"""{SYSTEM_PROMPT}

    Conversation history:
    {history}

    Structured NCERT educational context:
    {context}

    Student question:
    {payload.question}

    Instructions:
    1. Answer ONLY using the provided NCERT context.
    2. If the answer is partially available, answer only the supported parts.
    3. If the answer is unavailable, clearly say:
       "The provided NCERT context does not contain enough information."
    4. Keep the response concise, clear, and student-friendly.
    5. Prefer bullet points when explaining multiple ideas.
    6. Avoid repeating textbook sentences verbatim unless necessary.
    7. For maths/science:
       - explain steps clearly
       - preserve formulas exactly
    8. Keep the answer between 80 and 220 words.

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
        bundle = self._retrieve("flashcards", payload.class_num, payload.subject, payload.chapter, payload.focus_area or "Generate flashcards from the chapter.", payload.top_k)
        if not bundle.documents:
            return FlashcardResponse(
                flashcards=[],
                citations=[],
                notes=["No official NCERT content was found for this selection."],
                not_in_textbook=True,
            )

        context = self._format_structured_context(bundle.structured_context, payload.focus_area or "Entire chapter")
        prompt = f"""{SYSTEM_PROMPT}

Create exactly {payload.count} high-quality flashcards from the provided NCERT context.

Structured NCERT educational context:
{context}

Topic focus:
{payload.focus_area or "Entire chapter"}

Instructions:
1. Use ONLY the provided context.
2. Focus on concepts, definitions, formulas, rules, misconceptions, and applications.
3. Do NOT copy textbook sentences directly.
4. Do NOT generate story-based flashcards.
5. Do NOT ask vague prompts like "What did you understand?" or "Explain the lesson".
6. Prefer conceptual understanding over memorization.
7. If examples exist in context, abstract the underlying concept instead of repeating the story.
8. Use simple school-level English and keep every card concise.
9. Ensure every card helps with revision and exam preparation.
10. Prioritize these flashcard types:
     - definition
     - formula
     - concept
     - application
     - misconception
11. Front should test recall or understanding.
12. Back should contain the precise answer.
13. Explanation should clarify the idea in 1-2 short sentences.
14. Set type to one of: definition, concept, formula, application, misconception.
15. Set difficulty to one of: easy, medium, hard.
16. Set blooms_level to one of: remember, understand, apply.
17. If the context is limited, prefer fewer but higher-quality flashcards instead of inventing information.

Return STRICT valid JSON only.

Schema:
{{
    "flashcards": [
        {{
            "front": "Concise question or cue",
            "back": "Short answer",
            "explanation": "Simple explanation"
            "type": "definition/concept/formula/application/misconception",
            "difficulty": "easy/medium/hard",
            "blooms_level": "remember/understand/apply"
        }}
    ],
    "notes": [
        "Important revision note"
    ]
}}
"""

        try:
            parsed = self._invoke_json(prompt, timeout_seconds=35)
        except Exception:
            parsed = {}
        flashcards = []
        for item in parsed.get("flashcards", []):
            try:
                flashcard = FlashcardItem.model_validate(item)
            except Exception:
                continue
            if flashcard.front.strip() and flashcard.back.strip():
                flashcards.append(flashcard)
        if not flashcards:
            flashcards = self._fallback_flashcards(bundle.documents, payload.count)
        notes = [str(note) for note in parsed.get("notes", [])]
        if flashcards and not parsed.get("flashcards"):
            notes.append("Generated from retrieved NCERT context because the local model did not return valid flashcard JSON.")
        return FlashcardResponse(
            flashcards=flashcards,
            citations=bundle.citations,
            notes=notes,
            not_in_textbook=bundle.not_in_textbook and not flashcards,
        )

    def generate_questions(self, payload: QuestionRequest) -> QuestionResponse:
        is_math = self._is_math_subject(payload.subject)
        bundle = self._retrieve(
            "conceptual_qa",
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

        context = self._format_structured_context(bundle.structured_context, "Conceptual study questions")
        math_instruction = (
            "For Mathematics, generate mostly problem-solving questions from chapter exercises (about 70%) and keep the rest concept-based (about 30%)."
            if is_math
            else ""
        )
        prompt = f"""{SYSTEM_PROMPT}

{CONCEPTUAL_QA_PROMPT.format(context=context)}

Additional instructions:
1. Create exactly {payload.count} questions.
2. Prioritize:
     a) End-of-chapter exercises
     b) Important definitions
     c) Concept explanations
     d) Examples abstracted into new conceptual scenarios
3. Do not invent questions unsupported by context.
4. Answers must be concise but complete.
5. Explanations should improve conceptual understanding.
6. Difficulty distribution:
     - 40% easy
     - 40% medium
     - 20% hard
7. For Mathematics:
     - 70% problem-solving
     - 30% conceptual
     - include step-by-step reasoning when needed
8. Preserve formulas and units exactly.
9. Set type to conceptual, application, reasoning, compare and contrast, problem-solving, or HOTS as appropriate.
10. Keep blooms_level aligned with the cognitive demand.

{math_instruction}

Return STRICT valid JSON only.

If you need a schema reminder, use:
{{
    "questions": [
        {{
            "question": "Question text",
            "answer": "Correct answer",
            "explanation": "Concept explanation",
            "difficulty": "easy|medium|hard",
            "type": "conceptual/application/HOTS",
            "blooms_level": "understand/apply/analyze"
        }}
    ],
    "notes": [
        "Important study note"
    ]
}}
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
            "mcq",
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

        context = self._format_structured_context(bundle.structured_context, "MCQ generation")
        math_instruction = (
            "For Mathematics, include mostly numerical/problem-solving items from exercises (about 70%) and the rest concept-based questions (about 30%)."
            if is_math
            else ""
        )
        # Force MCQ-only generation for all subjects
        prompt = f"""{SYSTEM_PROMPT}

You are an NCERT educational assessment expert.

Generate HIGH-QUALITY concept-based MCQs for school students.

STRICT RULES:

1. Questions must test conceptual understanding.
2. Do NOT copy textbook lines directly.
3. Do NOT generate story-based questions.
4. Avoid questions based only on textbook examples.
5. Convert examples into generalized concepts.
6. Questions must be educationally meaningful.
7. Avoid vague questions.
8. Avoid repetitive questions.
9. Distractors must be realistic and educationally valid.
10. Wrong options should reflect common student mistakes.
11. Questions should align with Bloom taxonomy.
12. Use age-appropriate language.

FOR MATH:
Distractors should include:
- sign mistakes
- operation mistakes
- formula misuse
- common misconceptions

FOR SCIENCE:
Distractors should include:
- conceptual confusion
- incorrect processes
- wrong scientific reasoning

FOR SOCIAL:
Focus on:
- cause-effect
- understanding
- analysis
- chronology

FOR ENGLISH:
Focus on:
- grammar concepts
- comprehension
- vocabulary usage
- writing logic

AVOID:
- random options
- meaningless numbers
- trivial factual recall
- textbook sentence copying

Create exactly {payload.count} multiple-choice questions from the provided NCERT context.

Structured NCERT educational context:
{context}

Difficulty level:
{payload.difficulty}

Instructions:
1. Use ONLY the provided NCERT context.
2. Prioritize key concepts, definitions, formulas, rules, and conceptual applications.
3. Do NOT copy textbook questions verbatim.
4. Convert textbook ideas into assessment-style MCQs.
5. If examples are present, abstract the underlying concept instead of testing the example itself.
6. Every question must:
     - test understanding
     - have exactly 4 options
     - contain only 1 correct answer
7. Distractors should be believable, realistic, and reflect common student mistakes.
8. Avoid ambiguous wording and trivial recall-only questions.
9. Explanations must justify why the correct answer is right.
10. Maintain NCERT terminology and simple school-level English.
11. For Mathematics:
     - prioritize concept-based and application-based MCQs
     - include distractors based on sign mistakes, operation mistakes, formula misuse, and misconceptions
12. For Science:
     - include distractors based on conceptual confusion, wrong process order, and incorrect reasoning
13. For Social Science:
     - emphasize cause-effect, chronology, understanding, and analysis
14. For English:
     - emphasize grammar concepts, comprehension, vocabulary usage, and writing logic
15. Bloom levels should be one of: remember, understand, apply, analyze.
16. Choose the difficulty label carefully so it matches the cognitive demand.

{math_instruction}

Return STRICT valid JSON only.

Schema:
{{
    "quiz_title": "Chapter quiz title",
    "questions": [
        {{
            "question": "Question text",
            "question_type": "mcq",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A",
            "explanation": "Why the answer is correct",
            "difficulty": "easy|medium|hard",
            "blooms_level": "remember|understand|apply|analyze"
        }}
    ],
    "notes": [
        "Important revision note"
    ]
}}
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

    def summarize_chapter(self, class_num: int, subject: str | None, chapter: str | None, top_k: int) -> ChatResponse:
        bundle = self._retrieve(
            "summary",
            class_num,
            subject,
            chapter,
            "Chapter summary, key ideas, important concepts, and exercise overview from this NCERT chapter.",
            top_k,
        )
        if not bundle.documents:
            return ChatResponse(
                answer="The provided NCERT context does not contain enough information.",
                citations=[],
                not_in_textbook=True,
                retrieved_documents=0,
                source_scope=bundle.scope,
            )

        context = self._format_structured_context(bundle.structured_context, "Chapter summary")
        prompt = CHAPTER_SUMMARY_PROMPT.format(context=context)
        try:
            summary_payload = self._invoke_json(prompt, timeout_seconds=20)
            summary = self._format_summary_response(summary_payload)
            if not summary.strip():
                summary = self.chat_model.invoke(prompt).content.strip()
        except Exception as exc:
            summary = f"The provided NCERT context does not contain enough information. {exc}"

        if not summary:
            summary = "The provided NCERT context does not contain enough information."

        return ChatResponse(
            answer=summary,
            citations=bundle.citations,
            not_in_textbook=bundle.not_in_textbook,
            retrieved_documents=len(bundle.documents),
            source_scope=bundle.scope,
        )

    def _retrieve(self, task_type: str, class_num: int, subject: str | None, chapter: str | None, query: str, top_k: int) -> RetrievedBundle:
        documents, citations, structured_context, scope, not_in_textbook = self.retrieval_pipeline.retrieve(
            task_type=task_type,
            query=query,
            class_num=class_num,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )
        return RetrievedBundle(
            documents=documents,
            citations=citations,
            structured_context={
                "concepts": structured_context.concepts,
                "definitions": structured_context.definitions,
                "formulae": structured_context.formulae,
                "applications": structured_context.applications,
                "misconceptions": structured_context.misconceptions,
                "learning_objectives": structured_context.learning_objectives,
            },
            scope=scope,
            not_in_textbook=not_in_textbook,
        )

    def _format_structured_context(self, structured_context: dict[str, list[str]], focus: str) -> str:
        ordered_context = {
            "focus": focus,
            "concepts": structured_context.get("concepts", []),
            "definitions": structured_context.get("definitions", []),
            "formulae": structured_context.get("formulae", []),
            "applications": structured_context.get("applications", []),
            "misconceptions": structured_context.get("misconceptions", []),
            "learning_objectives": structured_context.get("learning_objectives", []),
        }
        return json.dumps(ordered_context, ensure_ascii=True, indent=2)

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

    def _coerce_quiz_item(self, item: dict, default_type: str, default_difficulty: str) -> QuizItem:
        question_type = self._normalize_question_type(item.get("question_type"), default_type)
        if default_type == "mcq":
            question_type = "mcq"
        raw_options = item.get("options", [])
        question_text = str(item.get("question", "")).strip()
        options = self._normalize_quiz_options(raw_options, question_type, question_text)
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
            question=question_text,
            question_type=question_type,  # type: ignore[arg-type]
            options=options,
            correct_answer=correct_answer,
            explanation=str(item.get("explanation", "")).strip(),
            difficulty=difficulty,
        )

    def _coerce_question_item(self, item: dict) -> QuestionItem:
        difficulty = self._normalize_difficulty(item.get("difficulty"), "medium")
        question_type = str(item.get("type") or item.get("question_type") or "conceptual").strip()
        blooms_level = self._normalize_blooms_level(item.get("blooms_level"), "understand")
        return QuestionItem(
            question=str(item.get("question", "")).strip(),
            answer=str(item.get("answer", "")).strip(),
            explanation=str(item.get("explanation", "")).strip(),
            difficulty=difficulty,
            type=question_type,
            blooms_level=blooms_level,
        )

    def _format_summary_response(self, summary_payload: dict) -> str:
        if not summary_payload:
            return ""
        chapter_summary = str(summary_payload.get("chapter_summary", "")).strip()
        key_concepts = self._ensure_list(summary_payload.get("key_concepts"))
        important_formulas = self._ensure_list(summary_payload.get("important_formulas"))
        common_mistakes = self._ensure_list(summary_payload.get("common_mistakes"))
        exam_points = self._ensure_list(summary_payload.get("exam_points"))

        blocks: list[str] = []
        if chapter_summary:
            blocks.append(chapter_summary)
        if key_concepts:
            blocks.append("Key concepts:\n- " + "\n- ".join(key_concepts))
        if important_formulas:
            blocks.append("Important formulas:\n- " + "\n- ".join(important_formulas))
        if common_mistakes:
            blocks.append("Common mistakes:\n- " + "\n- ".join(common_mistakes))
        if exam_points:
            blocks.append("Exam points:\n- " + "\n- ".join(exam_points))
        return "\n\n".join(blocks).strip()

    def _normalize_difficulty(self, value: object, fallback: str) -> str:
        candidate = str(value or fallback).strip().lower()
        if candidate not in {"easy", "medium", "hard"}:
            return "medium"
        return candidate

    def _normalize_blooms_level(self, value: object, fallback: str) -> str:
        candidate = str(value or fallback).strip().lower()
        if candidate not in {"understand", "apply", "analyze"}:
            return "understand"
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
                    type="application" if is_math else "conceptual",
                    blooms_level="apply" if is_math else "understand",
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
                    type="conceptual",
                    blooms_level="understand",
                )
            )
        return questions[:count]

    def _fallback_flashcards(self, documents: list[Document], count: int) -> list[FlashcardItem]:
        flashcards: list[FlashcardItem] = []

        for question_text, answer_text in self._extract_exercise_pairs(documents):
            if len(flashcards) >= count:
                return flashcards
            flashcards.append(
                FlashcardItem(
                    front=self._trim_text(question_text, 140),
                    back=self._trim_text(answer_text, 180),
                    explanation="Based on the selected NCERT chapter context.",
                    type="application",
                    difficulty="medium",
                    blooms_level="apply",
                )
            )

        for line in self._first_context_sentences(documents, count * 2):
            if len(flashcards) >= count:
                break
            if self._is_duplicate_text(line, [card.front for card in flashcards]):
                continue
            flashcards.append(
                FlashcardItem(
                    front=f"What is the key idea in this NCERT line: {self._trim_text(line, 100)}?",
                    back=self._trim_text(line, 180),
                    explanation="This card was generated directly from the retrieved textbook context.",
                    type="concept",
                    difficulty="easy",
                    blooms_level="understand",
                )
            )

        return flashcards[:count]

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
                    question=self._quiz_stem_from_pair(question_text, answer_text),
                    question_type=question_type,  # type: ignore[arg-type]
                    options=options,
                    correct_answer=correct_answer,
                    explanation="Based on the end-of-chapter exercise section in the selected NCERT chapter.",
                    difficulty=self._normalize_difficulty(difficulty, "medium"),
                    blooms_level="apply" if question_type == "mcq" else "understand",
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
                    blooms_level="understand",
                )
            )
        return questions[:count]

    def _quiz_stem_from_pair(self, question_text: str, answer_text: str) -> str:
        question = self._trim_text(question_text, 120).rstrip("?.")
        answer = self._trim_text(answer_text, 80).rstrip(".")
        if answer:
            return f"Which option correctly answers this NCERT prompt: {question}?"
        return f"Which option best matches this NCERT concept: {question}?"

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
        if len(lowered) < 3:
            return True
        placeholder_patterns = [
            r"^option\s*[a-d1-4]?$",
            r"^enter(?:\s+your)?\s+answer$",
            r"^type(?:\s+your)?\s+answer$",
            r"^write(?:\s+your)?\s+answer$",
            r"^fill(?:\s+in)?(?:\s+the)?(?:\s+blank)?$",
        ]
        return any(re.match(pattern, lowered) for pattern in placeholder_patterns)

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

    def _normalize_quiz_options(self, raw_options: object, question_type: str, question_text: str = "") -> list[QuizOption]:
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

    def _trim_text(self, text: str, max_length: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[: max_length - 1].rstrip() + "..."

    def _is_duplicate_text(self, text: str, existing: list[str]) -> bool:
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key:
            return True
        return any(re.sub(r"[^a-z0-9]+", " ", item.lower()).strip() == key for item in existing)

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
                            blooms_level=getattr(q, "blooms_level", "understand") or "understand",
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
                            blooms_level=getattr(q, "blooms_level", "understand") or "understand",
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
                        blooms_level=getattr(q, "blooms_level", "understand") or "understand",
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
