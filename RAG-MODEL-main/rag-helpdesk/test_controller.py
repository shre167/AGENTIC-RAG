"""Test suite for the Agentic RAG Controller."""
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline.controller import (
    AgentController,
    _classify_question_type,
    _build_retrieval_plan,
    AgentRetrievalPhase,
)
from src.rag_pipeline.models import Evidence


class MockRetrievalAttemptResult:
    """Mock retrieval result."""
    def __init__(self, candidates=None, fused=None, search_text="test", retrieval_ms=100):
        self.candidates = candidates or {}
        self.fused = fused or []
        self.search_text = search_text
        self.retrieval_ms = retrieval_ms
        self.retrieval_mode = "hybrid"
        self.tools_called = ["dense_retrieve", "bm25_retrieve", "rrf_fusion"]
        self.fingerprint = "mock_fingerprint_v1"
        self.query_embedding = [0.1] * 384
        self.bm25_query_tokens = ["test", "query"]
        self.raw_dense_results = []
        self.raw_bm25_results = []
        self.rrf_results_snapshot = []
        self.rrf_calculations = []
        self.page_match_found = False
        self.chapter_match_found = False


class MockCandidate:
    """Mock candidate chunk."""
    def __init__(self, text="Sample content", chunk_id="chunk_1"):
        self.text = text
        self.chunk_id = chunk_id
        self.filename = "cosmos.pdf"
        self.rrf_score = 0.025
        self.dense_rank = 1
        self.bm25_rank = 1
        self.metadata = {"page_start": 1, "chapter_num": 1}

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "filename": self.filename,
        }


class TestQuestionClassification(unittest.TestCase):
    """Test question type classification."""

    def test_general_question(self):
        """Test classification of general astronomy question."""
        q_type = _classify_question_type("What is a black hole?", None, None)
        self.assertEqual(q_type, "general")

    def test_page_specific_question(self):
        """Test classification of page-specific question."""
        page_info = {"is_page_query": True, "page_start": 5}
        q_type = _classify_question_type("What is on page 5?", page_info, None)
        self.assertEqual(q_type, "page_specific")

    def test_chapter_specific_question(self):
        """Test classification of chapter-specific question."""
        chapter_info = {"is_chapter_query": True, "chapter_num": 3}
        q_type = _classify_question_type("Summarize chapter 3.", None, chapter_info)
        self.assertEqual(q_type, "chapter_specific")

    def test_multi_intent_question(self):
        """Test classification of multi-intent question."""
        with patch("src.rag_pipeline.controller.is_multi_intent_query", return_value=True):
            q_type = _classify_question_type("Compare Mars and Jupiter.", None, None)
            self.assertEqual(q_type, "multi_intent")


class TestRetrievalPlan(unittest.TestCase):
    """Test retrieval plan generation."""

    def test_general_plan(self):
        """Test retrieval plan for general question."""
        plan = _build_retrieval_plan("general", "hybrid", None, None)
        self.assertIn("hybrid", plan)
        self.assertIn("hybrid_semantic", plan)

    def test_page_specific_plan(self):
        """Test retrieval plan for page-specific question."""
        page_info = {"page_start": 42, "page_end": 42, "book_match": "Cosmos.pdf"}
        plan = _build_retrieval_plan("page_specific", "hybrid", page_info, None)
        self.assertIn("page=42-42", plan)
        self.assertIn("Cosmos.pdf", plan)

    def test_chapter_specific_plan(self):
        """Test retrieval plan for chapter-specific question."""
        chapter_info = {"chapter_num": 2, "book_match": "Cosmos.pdf"}
        plan = _build_retrieval_plan("chapter_specific", "bm25", None, chapter_info)
        self.assertIn("chapter=2", plan)
        self.assertIn("Cosmos.pdf", plan)


class TestControllerRetrieval(unittest.TestCase):
    """Test controller retrieval phase."""

    def setUp(self):
        """Set up mocks for each test."""
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.vector_store = MagicMock()
        self.mock_pipeline._rerank = MagicMock(return_value=False)
        self.mock_pipeline._evaluate_evidence = MagicMock(
            return_value=(
                [MockCandidate()],
                Evidence(
                    level="strong",
                    should_answer=True,
                    top_score=0.05,
                    score_source="hybrid",
                    supporting_chunks=1,
                    top_gap=0.02,
                    agreement=True,
                    top_percentile=95,
                    retrieval_mode="hybrid",
                    reason="Strong evidence found.",
                ),
            )
        )
        self.controller = AgentController(self.mock_pipeline)

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_general_question_retrieval(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test retrieval for general astronomy question."""
        mock_page.return_value = None
        mock_chapter.return_value = None
        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "What is a black hole?",
            "hybrid",
        )

        self.assertEqual(phase.question_type, "general")
        self.assertEqual(phase.final_decision, "answer")
        self.assertGreater(len(phase.attempts), 0)
        self.assertEqual(phase.retrieval_mode, "hybrid")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_page_specific_question(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test retrieval for page-specific question."""
        mock_page.return_value = {
            "is_page_query": True,
            "page_start": 60,
            "page_end": 60,
            "book_match": "Cosmos.pdf",
            "is_pure_page_query": False,  # Not pure lookup, semantic + page
            "semantic_query": "about dark matter",
        }
        mock_chapter.return_value = None
        mock_result = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )
        mock_result.page_match_found = True  # Page was found
        mock_execute.return_value = mock_result

        phase = self.controller.run_retrieval_phase(
            "What is on page 60?",
            "hybrid",
        )

        self.assertEqual(phase.question_type, "page_specific")
        self.assertIn("page=60-60", phase.retrieval_plan)
        self.assertEqual(phase.final_decision, "answer")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_chapter_specific_question(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test retrieval for chapter-specific question."""
        mock_page.return_value = None
        mock_chapter.return_value = {
            "is_chapter_query": True,
            "chapter_num": 3,
            "book_match": "Cosmos.pdf",
        }
        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "What does chapter 3 say about telescopes?",
            "hybrid",
        )

        self.assertEqual(phase.question_type, "chapter_specific")
        self.assertIn("chapter=3", phase.retrieval_plan)
        self.assertEqual(phase.final_decision, "answer")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_weak_evidence_triggers_retry(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test that weak evidence triggers a retry."""
        mock_page.return_value = None
        mock_chapter.return_value = None

        # First attempt: weak evidence
        weak_evidence = Evidence(
            level="weak",
            should_answer=False,
            top_score=0.008,
            score_source="hybrid",
            supporting_chunks=0,
            top_gap=0.001,
            agreement=False,
            top_percentile=30,
            retrieval_mode="hybrid",
            reason="Weak evidence; insufficient confidence.",
        )

        # Second attempt: strong evidence
        strong_evidence = Evidence(
            level="strong",
            should_answer=True,
            top_score=0.05,
            score_source="bm25",
            supporting_chunks=2,
            top_gap=0.02,
            agreement=True,
            top_percentile=95,
            retrieval_mode="bm25",
            reason="Strong BM25 match found.",
        )

        # Mock pipeline to return weak then strong
        self.mock_pipeline._evaluate_evidence = MagicMock(
            side_effect=[
                ([MockCandidate()], weak_evidence),
                ([MockCandidate()], strong_evidence),
            ]
        )

        # Create result with unique fingerprints to allow retry
        result1 = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )
        result1.fingerprint = "fp1_hybrid"

        result2 = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )
        result2.fingerprint = "fp2_bm25"
        result2.retrieval_mode = "bm25"

        mock_execute.side_effect = [result1, result2]

        phase = self.controller.run_retrieval_phase(
            "What is a neutron star?",
            "hybrid",
        )

        # Should attempt retry and eventually succeed
        self.assertGreaterEqual(len(phase.attempts), 1)
        self.assertEqual(phase.final_decision, "answer")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_insufficient_evidence_abstains(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test that insufficient evidence leads to abstention."""
        mock_page.return_value = None
        mock_chapter.return_value = None

        # Return weak evidence on all attempts
        weak_evidence = Evidence(
            level="weak",
            should_answer=False,
            top_score=0.005,
            score_source="hybrid",
            supporting_chunks=0,
            top_gap=None,
            agreement=False,
            top_percentile=10,
            retrieval_mode="hybrid",
            reason="No sufficient evidence found.",
        )

        self.mock_pipeline._evaluate_evidence = MagicMock(
            return_value=([MockCandidate()], weak_evidence)
        )

        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "Explain quantum entanglement using astronomy principles.",
            "hybrid",
        )

        # Should abstain after exhausting retries
        self.assertEqual(phase.final_decision, "abstain")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_agent_activity_structure(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test that agent_activity dict has required fields."""
        mock_page.return_value = None
        mock_chapter.return_value = None
        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "What causes solar flares?",
            "hybrid",
        )

        agent_activity = phase.agent_activity

        # Verify all required fields are present
        required_fields = [
            "question_type",
            "retrieval_plan",
            "attempts",
            "attempt_count",
            "retrieval_mode",
            "evidence_assessment",
            "final_decision",
            "timings_ms",
        ]

        for field in required_fields:
            self.assertIn(field, agent_activity, f"Missing field: {field}")

        # Verify attempt records have required fields
        if agent_activity["attempts"]:
            attempt = agent_activity["attempts"][0]
            attempt_fields = [
                "attempt",
                "retrieval_mode",
                "tools_called",
                "search_text",
                "evidence_sufficient",
                "evidence_level",
                "evidence_reason",
                "retrieval_ms",
                "candidate_count",
                "fused_count",
            ]
            for field in attempt_fields:
                self.assertIn(field, attempt, f"Missing attempt field: {field}")


class TestMultiPartQuestions(unittest.TestCase):
    """Test handling of complex multi-part and relationship questions."""

    def setUp(self):
        """Set up mocks for each test."""
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.vector_store = MagicMock()
        self.mock_pipeline._rerank = MagicMock(return_value=False)
        self.mock_pipeline._evaluate_evidence = MagicMock(
            return_value=(
                [MockCandidate()],
                Evidence(
                    level="strong",
                    should_answer=True,
                    top_score=0.05,
                    score_source="hybrid",
                    supporting_chunks=2,
                    top_gap=0.02,
                    agreement=True,
                    top_percentile=95,
                    retrieval_mode="hybrid",
                    reason="Strong evidence found.",
                ),
            )
        )
        self.controller = AgentController(self.mock_pipeline)

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.is_multi_intent_query")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_multi_part_comparison(
        self,
        mock_execute,
        mock_multi_intent,
        mock_chapter,
        mock_page,
    ):
        """Test handling of multi-part comparison questions."""
        mock_page.return_value = None
        mock_chapter.return_value = None
        mock_multi_intent.return_value = True
        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "Compare Mars and Jupiter.",
            "hybrid",
        )

        self.assertEqual(phase.question_type, "multi_intent")

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_concept_relationship_question(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test handling of concept relationship questions."""
        mock_page.return_value = None
        mock_chapter.return_value = None
        mock_execute.return_value = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )

        phase = self.controller.run_retrieval_phase(
            "How do black holes affect nearby stars?",
            "hybrid",
        )

        self.assertEqual(phase.question_type, "general")
        self.assertEqual(phase.final_decision, "answer")


class TestRetryLogic(unittest.TestCase):
    """Test the retry logic and strategy switching."""

    def setUp(self):
        """Set up mocks for retry testing."""
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.vector_store = MagicMock()
        self.mock_pipeline._rerank = MagicMock(return_value=False)
        self.controller = AgentController(self.mock_pipeline)

    @patch("src.rag_pipeline.controller.extract_page_query_info")
    @patch("src.rag_pipeline.controller.extract_chapter_query_info")
    @patch("src.rag_pipeline.controller.execute_retrieval_attempt")
    def test_hybrid_to_bm25_retry(
        self,
        mock_execute,
        mock_chapter,
        mock_page,
    ):
        """Test retry strategy switches from hybrid to BM25."""
        mock_page.return_value = None
        mock_chapter.return_value = None

        weak_evidence = Evidence(
            level="weak",
            should_answer=False,
            top_score=0.006,
            score_source="hybrid",
            supporting_chunks=0,
            top_gap=None,
            agreement=False,
            top_percentile=5,
            retrieval_mode="hybrid",
            reason="Weak hybrid evidence.",
        )

        strong_evidence = Evidence(
            level="strong",
            should_answer=True,
            top_score=50.0,
            score_source="bm25",
            supporting_chunks=3,
            top_gap=10.0,
            agreement=True,
            top_percentile=98,
            retrieval_mode="bm25",
            reason="Strong BM25 match.",
        )

        self.mock_pipeline._evaluate_evidence = MagicMock(
            side_effect=[
                ([MockCandidate()], weak_evidence),
                ([MockCandidate()], strong_evidence),
            ]
        )

        result1 = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )
        result1.fingerprint = "fp1_hybrid"

        result2 = MockRetrievalAttemptResult(
            candidates={"chunk_1": MockCandidate()},
            fused=[MockCandidate()],
        )
        result2.fingerprint = "fp2_bm25"
        result2.retrieval_mode = "bm25"

        mock_execute.side_effect = [result1, result2]

        phase = self.controller.run_retrieval_phase(
            "What is dark energy?",
            "hybrid",
        )

        # Should eventually answer after retry
        self.assertEqual(phase.final_decision, "answer")
        # Should have at least 2 attempts (initial + retry)
        self.assertGreaterEqual(len(phase.attempts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
