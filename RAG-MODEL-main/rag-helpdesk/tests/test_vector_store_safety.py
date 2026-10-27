"""Regression tests for vectorstore ingestion safety (uses temporary stores only)."""
from __future__ import annotations

import pytest

from src.vector_store import ChromaVectorStore, EmbeddingDimensionMismatchError


def _vec(dim: int, seed: float = 0.1) -> list[float]:
    return [seed] * dim


def test_add_documents_is_incremental(tmp_path):
    vs = ChromaVectorStore(persist_directory=tmp_path / "vs")
    dim = 128

    n_existing = 3
    vs.add_documents(
        texts=[f"doc-{i}" for i in range(n_existing)],
        embeddings=[_vec(dim, 0.1 + i) for i in range(n_existing)],
        ids=[f"id-{i}" for i in range(n_existing)],
    )
    assert vs.get_collection_count() == n_existing

    m_new = 2
    vs.add_documents(
        texts=[f"new-{i}" for i in range(m_new)],
        embeddings=[_vec(dim, 0.5 + i) for i in range(m_new)],
        ids=[f"new-id-{i}" for i in range(m_new)],
    )
    assert vs.get_collection_count() == n_existing + m_new


def test_dimension_mismatch_does_not_delete_collection(tmp_path):
    vs = ChromaVectorStore(persist_directory=tmp_path / "vs")
    vs.add_documents(["existing"], [_vec(3072)], ids=["keep-me"])
    assert vs.get_collection_count() == 1
    assert vs.get_collection_embedding_dimension() == 3072

    with pytest.raises(EmbeddingDimensionMismatchError):
        vs.add_documents(["new"], [_vec(768)], ids=["new-id"])

    assert vs.get_collection_count() == 1
    assert vs.get_collection_embedding_dimension() == 3072
    assert vs.collection_exists()
