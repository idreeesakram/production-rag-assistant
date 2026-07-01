"""Tests for generate_related_work."""
import pytest
from unittest.mock import patch, MagicMock

from src.generation.related_work import generate_related_work, RelatedWorkResponse


class TestRelatedWorkResponse:
    def test_valid_response(self):
        r = RelatedWorkResponse(
            related_work="A" * 100 + " [Some Paper Title] more text.",
            cited_papers=["Some Paper Title"],
        )
        assert r.valid is True

    def test_too_short_raises(self):
        with pytest.raises(Exception):
            RelatedWorkResponse(
                related_work="too short",
                cited_papers=["Paper"],
            )

    def test_no_brackets_raises(self):
        with pytest.raises(Exception):
            RelatedWorkResponse(
                related_work="A" * 100 + " no brackets here at all.",
                cited_papers=["Paper"],
            )

    def test_empty_cited_papers_raises(self):
        with pytest.raises(Exception):
            RelatedWorkResponse(
                related_work="A" * 100 + " [Paper Title].",
                cited_papers=[],
            )


class TestGenerateRelatedWork:
    @patch("src.generation.related_work._invoke_llm_for_related_work")
    @patch("src.generation.related_work.retrieval")
    def test_successful_generation(self, mock_retrieval, mock_llm):
        chunks = [{"paper_title": "Attention Is All You Need", "page_num": 1, "text": "transformer text", "doc_id": "d1"}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = "A" * 100 + " [Attention Is All You Need] describes transformers in detail."

        bm25 = MagicMock()
        with patch("src.generation.related_work.Config.validate"):
            result_text, cited = generate_related_work("transformers", bm25)

        assert "Attention Is All You Need" in cited
        assert len(result_text) >= 100

    @patch("src.generation.related_work._invoke_llm_for_related_work")
    @patch("src.generation.related_work.retrieval")
    def test_raises_when_no_chunks(self, mock_retrieval, mock_llm):
        mock_retrieval.return_value = []
        bm25 = MagicMock()
        with patch("src.generation.related_work.Config.validate"):
            with pytest.raises(ValueError, match="No relevant papers"):
                generate_related_work("query", bm25)

    @patch("src.generation.related_work._invoke_llm_for_related_work")
    @patch("src.generation.related_work.retrieval")
    def test_retries_with_feedback_on_bad_citation(self, mock_retrieval, mock_llm):
        chunks = [{"paper_title": "Real Paper", "page_num": 1, "text": "text", "doc_id": "d1"}]
        mock_retrieval.return_value = chunks

        good_response = "A" * 100 + " [Real Paper] is discussed here."
        mock_llm.side_effect = [
            "no brackets here at all " * 10,
            good_response,
        ]

        bm25 = MagicMock()
        with patch("src.generation.related_work.Config.validate"):
            result_text, cited = generate_related_work("query", bm25, max_retries=2)

        assert "Real Paper" in cited

    @patch("src.generation.related_work._invoke_llm_for_related_work")
    @patch("src.generation.related_work.retrieval")
    def test_raises_after_all_retries_fail(self, mock_retrieval, mock_llm):
        chunks = [{"paper_title": "Real Paper", "page_num": 1, "text": "text", "doc_id": "d1"}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = "no valid citations here " * 10

        bm25 = MagicMock()
        with patch("src.generation.related_work.Config.validate"):
            with pytest.raises(RuntimeError, match="Failed to generate"):
                generate_related_work("query", bm25, max_retries=2)