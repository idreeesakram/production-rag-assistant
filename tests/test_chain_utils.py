"""Tests for related_work utility functions."""
from unittest.mock import MagicMock
import pytest

from src.generation.related_work import (
    _extract_cited_papers_from_response,
    _find_closest_title,
    _build_related_work_prompt,
)


class TestFindClosestTitle:
    def test_exact_match(self):
        titles = ["Attention Is All You Need", "BERT: Pre-training"]
        assert _find_closest_title("Attention Is All You Need", titles) == "Attention Is All You Need"

    def test_fuzzy_match_typo(self):
        titles = ["Attention Is All You Need"]
        result = _find_closest_title("Attention Is All You Nedd", titles, threshold=0.8)
        assert result == "Attention Is All You Need"

    def test_no_match_below_threshold(self):
        titles = ["Attention Is All You Need"]
        assert _find_closest_title("quantum unicorns paper", titles, threshold=0.8) is None

    def test_empty_titles(self):
        assert _find_closest_title("anything", [], threshold=0.8) is None


class TestExtractCitedPapers:
    def test_extracts_valid_title(self):
        titles = ["Attention Is All You Need"]
        text = "As shown in [Attention Is All You Need], transformers work well."
        result = _extract_cited_papers_from_response(text, titles)
        assert result == ["Attention Is All You Need"]

    def test_ignores_fabricated_title(self):
        titles = ["Attention Is All You Need"]
        text = "As shown in [made-up paper about quantum unicorns], this is great."
        result = _extract_cited_papers_from_response(text, titles)
        assert result == []

    def test_ignores_source_keyword(self):
        titles = ["Attention Is All You Need"]
        text = "See [SOURCE] for details."
        result = _extract_cited_papers_from_response(text, titles)
        assert result == []

    def test_ignores_short_matches(self):
        titles = ["Attention Is All You Need"]
        text = "See [AI] for details."
        result = _extract_cited_papers_from_response(text, titles)
        assert result == []

    def test_deduplicates_citations(self):
        titles = ["Attention Is All You Need"]
        text = "[Attention Is All You Need] and again [Attention Is All You Need]."
        result = _extract_cited_papers_from_response(text, titles)
        assert result == ["Attention Is All You Need"]

    def test_multiple_valid_titles(self):
        titles = ["Attention Is All You Need", "BERT Pre-training"]
        text = "[Attention Is All You Need] uses [BERT Pre-training]."
        result = _extract_cited_papers_from_response(text, titles)
        assert set(result) == {"Attention Is All You Need", "BERT Pre-training"}

    def test_empty_text(self):
        assert _extract_cited_papers_from_response("", ["Attention Is All You Need"]) == []

    def test_no_brackets(self):
        assert _extract_cited_papers_from_response("no citations here", ["Paper A"]) == []


class TestBuildRelatedWorkPrompt:
    def test_includes_query(self):
        chunks = [{"paper_title": "Paper A", "page_num": 1, "text": "some text"}]
        prompt = _build_related_work_prompt("transformer architecture", chunks)
        assert "transformer architecture" in prompt

    def test_includes_paper_titles(self):
        chunks = [{"paper_title": "Attention Is All You Need", "page_num": 1, "text": "text"}]
        prompt = _build_related_work_prompt("query", chunks)
        assert "Attention Is All You Need" in prompt

    def test_includes_error_feedback_on_retry(self):
        chunks = [{"paper_title": "Paper A", "page_num": 1, "text": "text"}]
        prompt = _build_related_work_prompt("query", chunks, error_feedback="No citations found")
        assert "PREVIOUS ATTEMPT FAILED" in prompt
        assert "No citations found" in prompt

    def test_no_feedback_on_first_attempt(self):
        chunks = [{"paper_title": "Paper A", "page_num": 1, "text": "text"}]
        prompt = _build_related_work_prompt("query", chunks)
        assert "PREVIOUS ATTEMPT FAILED" not in prompt
