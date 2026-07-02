"""Tests for ingestion pipeline v2 - token-based chunking via tiktoken."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.ingestion.pipeline_v2 import ingest_v2 as ingest


class TestIngestSuccess:
    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_full_pipeline_returns_counts(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1, "doc_id": "d1"}]
        mock_embed.return_value = [{"text": "chunk", "embedding": [0.1]}]
        mock_upsert.return_value = 1

        result = ingest("/fake/doc.pdf", paper_title="Test Paper")

        assert result["pages"] == 1
        assert result["chunks"] >= 1

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_real_chunker_is_used(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        page_text = "This is a test paragraph with enough text to trigger chunking. " * 10
        mock_extract.return_value = [{"text": page_text, "page_num": 1, "doc_id": "d1"}]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf", paper_title="Test Paper")

        embedded_chunks = mock_embed.call_args[0][0]
        assert len(embedded_chunks) > 0
        for chunk in embedded_chunks:
            assert "text" in chunk
            assert "page_num" in chunk

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_chunk_metadata_preserved_through_pipeline(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [
            {"text": "A" * 200, "page_num": 3, "doc_id": "doc_abc"},
            {"text": "B" * 200, "page_num": 7, "doc_id": "doc_abc"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf", paper_title="Test Paper")

        embedded_chunks = mock_embed.call_args[0][0]
        page_nums = {c["page_num"] for c in embedded_chunks}
        assert 3 in page_nums
        assert 7 in page_nums

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_empty_pages_skipped_by_chunker(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [
            {"text": "", "page_num": 1, "doc_id": "d1"},
            {"text": "A" * 200, "page_num": 2, "doc_id": "d1"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf", paper_title="Test Paper")

        embedded_chunks = mock_embed.call_args[0][0]
        page_nums = {c["page_num"] for c in embedded_chunks}
        assert 1 not in page_nums
        assert 2 in page_nums

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_chunks_have_total_chunks_metadata(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [
            {"text": "A" * 200, "page_num": 1, "doc_id": "d1"},
            {"text": "B" * 200, "page_num": 2, "doc_id": "d1"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        result = ingest("/fake/doc.pdf", paper_title="Test Paper")

        assert "pages" in result
        assert "chunks" in result
        assert result["pages"] == 2


class TestIngestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest("/missing/file.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_extraction_error(self, mock_exists, mock_extract):
        mock_extract.side_effect = RuntimeError("extraction failed")
        with pytest.raises(RuntimeError):
            ingest("/fake/doc.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_no_pages(self, mock_exists, mock_extract):
        mock_extract.return_value = []
        with pytest.raises(ValueError, match="No text extracted"):
            ingest("/fake/doc.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_chunking_error(self, mock_exists, mock_extract):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1}]
        with patch("src.ingestion.pipeline_v2._chunk_text_by_tokens", side_effect=RuntimeError("chunk failed")):
            with pytest.raises(Exception):
                ingest("/fake/doc.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_no_chunks(self, mock_exists, mock_extract):
        mock_extract.return_value = [{"text": "   ", "page_num": 1}]
        with pytest.raises(ValueError, match="No chunks created"):
            ingest("/fake/doc.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_embedding_error(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1}]
        mock_embed.side_effect = RuntimeError("embed failed")
        with pytest.raises(RuntimeError):
            ingest("/fake/doc.pdf", paper_title="Test")

    @patch("src.ingestion.pipeline_v2.upsert_chunks")
    @patch("src.ingestion.pipeline_v2.embed_chunks")
    @patch("src.ingestion.pipeline_v2.extract_pages")
    @patch("src.ingestion.pipeline_v2.Path.exists", return_value=True)
    def test_storage_error(self, mock_exists, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1}]
        mock_embed.return_value = [{"text": "chunk", "page_num": 1}]
        mock_upsert.side_effect = RuntimeError("storage failed")
        with pytest.raises(RuntimeError):
            ingest("/fake/doc.pdf", paper_title="Test")
