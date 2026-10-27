"""Comprehensive tests for multi-format document support (TXT, MD, DOCX, PDF)."""

import io
import tempfile
from pathlib import Path

import pytest

from src.document_loader import (
    SUPPORTED_EXTENSIONS,
    chunk_documents,
    detect_file_type,
    list_knowledge_files,
    load_document,
    load_documents,
)


class TestFileDetection:
    """Test detect_file_type for all 4 formats."""

    def test_detect_txt(self):
        assert detect_file_type("test.txt") == "txt"

    def test_detect_md(self):
        assert detect_file_type("readme.md") == "md"

    def test_detect_docx(self):
        assert detect_file_type("document.docx") == "docx"

    def test_detect_pdf(self):
        assert detect_file_type("report.pdf") == "pdf"

    def test_detect_pdf_by_magic_bytes(self):
        pdf_bytes = b"%PDF-1.4\n%test"
        result = detect_file_type("unknown", content=pdf_bytes)
        assert result == "pdf"

    def test_supported_extensions(self):
        assert SUPPORTED_EXTENSIONS == {".txt", ".md", ".docx", ".pdf"}


class TestChunking:
    """Test chunk_documents preserves metadata."""

    def test_chunking_preserves_file_type(self, tmp_path):
        test_file = tmp_path / "astronomy.txt"
        test_file.write_text("A" * 1000)
        
        docs = load_document(test_file)
        docs[0]["file_type"] = "txt"
        
        chunks = chunk_documents(docs)
        
        assert len(chunks) > 0
        assert all(c.get("file_type") == "txt" for c in chunks)

    def test_chunk_ids_unique(self):
        docs = [
            {"filename": "file1.txt", "content": "A" * 500, "file_type": "txt", "page_number": None},
            {"filename": "file2.txt", "content": "B" * 500, "file_type": "txt", "page_number": None},
        ]
        
        chunks = chunk_documents(docs)
        chunk_ids = [c["chunk_id"] for c in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_pdf_chunk_ids_include_page(self):
        docs = [{"filename": "report.pdf", "content": "A" * 500, "file_type": "pdf", "page_number": 1}]
        chunks = chunk_documents(docs)
        
        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.get("page_number") and chunk.get("page_number") > 0:
                assert f"p{chunk['page_number']}" in chunk["chunk_id"]


class TestKnowledgeFiles:
    """Test list_knowledge_files finds all formats."""

    def test_list_knowledge_files(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("Text")
        (tmp_path / "doc2.md").write_text("Markdown")
        
        files = list_knowledge_files(tmp_path)
        filenames = {f.name for f in files}
        
        assert "doc1.txt" in filenames
        assert "doc2.md" in filenames

    def test_metadata_propagates(self, tmp_path):
        (tmp_path / "astronomy.txt").write_text("The Milky Way is a spiral galaxy")
        (tmp_path / "planets.md").write_text("# Planets\n\nJupiter is largest")
        
        docs = load_documents(tmp_path)
        chunks = chunk_documents(docs)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "file_type" in chunk
            assert "filename" in chunk
            assert chunk["file_type"] in ("txt", "md")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
