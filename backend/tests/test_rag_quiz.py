"""Unit tests for RAG quiz generation, deduplication, and difficulty selection."""

import pytest
from unittest.mock import MagicMock, patch
from app.core.config import Settings
from app.schemas import QuizItem, QuizOption
from app.services.rag import RagService
from app.services.vectorstore import VectorStoreService


@pytest.fixture
def mock_settings():
    """Create a mock settings object."""
    settings = MagicMock(spec=Settings)
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "qwen2.5:14b"
    settings.quiz_ollama_model = "mistral:7b"
    settings.math_ollama_model = "deepseek-r1:8b"
    settings.embedding_model_name = "nomic-embed-text"
    return settings


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    return MagicMock(spec=VectorStoreService)


@pytest.fixture
def rag_service(mock_settings, mock_vectorstore):
    """Create a RagService instance with mocked dependencies."""
    with patch("app.services.rag.build_chat_model") as mock_build:
        mock_build.return_value = MagicMock()
        service = RagService(mock_settings, mock_vectorstore)
    return service


def create_quiz_item(question: str, difficulty: str = "medium") -> QuizItem:
    """Helper to create a QuizItem for testing."""
    return QuizItem(
        question=question,
        question_type="mcq",
        options=[
            QuizOption(label="A", text="Option A"),
            QuizOption(label="B", text="Option B"),
            QuizOption(label="C", text="Option C"),
            QuizOption(label="D", text="Option D"),
        ],
        correct_answer="A",
        explanation="Test explanation",
        difficulty=difficulty,
    )


class TestDeduplication:
    """Tests for question deduplication logic."""

    def test_dedupe_removes_duplicates(self, rag_service):
        """Test that duplicate questions are removed."""
        q1 = create_quiz_item("What is photosynthesis?", "easy")
        q2 = create_quiz_item("What is photosynthesis?", "easy")  # Exact duplicate
        q3 = create_quiz_item("How does photosynthesis work?", "medium")

        questions = [q1, q2, q3]
        result = rag_service._dedupe_questions(questions)

        assert len(result) == 2, "Should have 2 unique questions (1 duplicate removed)"
        assert result[0].question == "What is photosynthesis?"
        assert result[1].question == "How does photosynthesis work?"

    def test_dedupe_ignores_whitespace(self, rag_service):
        """Test that deduplication ignores extra whitespace."""
        q1 = create_quiz_item("What is  photosynthesis?", "easy")
        q2 = create_quiz_item("What is photosynthesis?", "easy")

        questions = [q1, q2]
        result = rag_service._dedupe_questions(questions)

        assert len(result) == 1, "Should treat whitespace variations as duplicates"

    def test_dedupe_case_insensitive(self, rag_service):
        """Test that deduplication is case-insensitive."""
        q1 = create_quiz_item("what is photosynthesis?", "easy")
        q2 = create_quiz_item("What Is Photosynthesis?", "easy")

        questions = [q1, q2]
        result = rag_service._dedupe_questions(questions)

        assert len(result) == 1, "Should treat case variations as duplicates"

    def test_dedupe_preserves_order(self, rag_service):
        """Test that deduplication preserves question order."""
        q1 = create_quiz_item("Question A", "easy")
        q2 = create_quiz_item("Question B", "medium")
        q3 = create_quiz_item("Question C", "hard")

        questions = [q1, q2, q3]
        result = rag_service._dedupe_questions(questions)

        assert len(result) == 3
        assert result[0].question == "Question A"
        assert result[1].question == "Question B"
        assert result[2].question == "Question C"


class TestDifficultySelection:
    """Tests for difficulty-based question selection."""

    def test_prefer_easy_difficulty(self, rag_service):
        """Test that easy difficulty questions are preferred when requested."""
        q1 = create_quiz_item("Easy question 1", "easy")
        q2 = create_quiz_item("Easy question 2", "easy")
        q3 = create_quiz_item("Medium question", "medium")
        q4 = create_quiz_item("Hard question", "hard")

        questions = [q1, q2, q3, q4]
        result = rag_service._select_by_difficulty(questions, count=2, difficulty="easy")

        assert len(result) == 2
        assert all(q.difficulty == "easy" for q in result)

    def test_prefer_hard_difficulty(self, rag_service):
        """Test that hard difficulty questions are preferred when requested."""
        q1 = create_quiz_item("Easy question", "easy")
        q2 = create_quiz_item("Medium question", "medium")
        q3 = create_quiz_item("Hard question 1", "hard")
        q4 = create_quiz_item("Hard question 2", "hard")

        questions = [q1, q2, q3, q4]
        result = rag_service._select_by_difficulty(questions, count=2, difficulty="hard")

        assert len(result) == 2
        assert all(q.difficulty == "hard" for q in result)

    def test_fallback_to_other_difficulties(self, rag_service):
        """Test that other difficulties are used as fallback when requested difficulty is insufficient."""
        q1 = create_quiz_item("Easy question", "easy")
        q2 = create_quiz_item("Medium question 1", "medium")
        q3 = create_quiz_item("Medium question 2", "medium")

        questions = [q1, q2, q3]
        result = rag_service._select_by_difficulty(questions, count=3, difficulty="hard")

        assert len(result) == 3
        # Should get med first (closest to hard), then easy or med
        assert any(q.difficulty == "medium" for q in result)

    def test_respects_count_limit(self, rag_service):
        """Test that selection respects the requested count."""
        questions = [create_quiz_item(f"Question {i}", "medium") for i in range(10)]
        result = rag_service._select_by_difficulty(questions, count=5, difficulty="medium")

        assert len(result) == 5

    def test_handles_empty_list(self, rag_service):
        """Test that selection handles empty question list gracefully."""
        result = rag_service._select_by_difficulty([], count=5, difficulty="medium")

        assert len(result) == 0

    def test_handles_insufficient_questions(self, rag_service):
        """Test that selection returns all available questions when count exceeds available."""
        questions = [create_quiz_item(f"Question {i}", "medium") for i in range(3)]
        result = rag_service._select_by_difficulty(questions, count=10, difficulty="medium")

        assert len(result) == 3


class TestCombinedDedupAndDifficulty:
    """Tests for combined deduplication and difficulty selection."""

    def test_dedupe_then_select_difficulty(self, rag_service):
        """Test full workflow: dedup then select by difficulty."""
        # Create mix of duplicates and various difficulties
        questions = [
            create_quiz_item("What is photosynthesis?", "easy"),
            create_quiz_item("What is photosynthesis?", "easy"),  # duplicate
            create_quiz_item("Explain photosynthesis in detail", "hard"),
            create_quiz_item("Why does photosynthesis matter?", "medium"),
        ]

        deduped = rag_service._dedupe_questions(questions)
        result = rag_service._select_by_difficulty(deduped, count=2, difficulty="medium")

        assert len(result) == 2
        # Should prefer medium, then medium (no duplicates)
        assert result[0].question == "Why does photosynthesis matter?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
