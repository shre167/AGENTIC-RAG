"""
RAG Pipeline package.
"""
from __future__ import annotations

from src.rag_pipeline.config import (
    ABSTAIN_MESSAGE,
    ENABLE_RERANKER,
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    MAX_RERANK_CANDIDATES,
    MIN_RERANK_CANDIDATES,
    MODERATE_RRF_THRESHOLD,
    RERANK_ABSTAIN_FLOOR,
    RERANK_CANDIDATE_LIMIT,
    RRF_ABSTAIN_FLOOR,
    RRF_K,
    STRONG_RRF_THRESHOLD,
    SYSTEM_PROMPT,
)
from src.rag_pipeline.models import Candidate, Evidence
from src.rag_pipeline.pipeline import RAGPipeline
from src.rag_pipeline.query import is_greeting

__all__ = [
    "ABSTAIN_MESSAGE",
    "ENABLE_RERANKER",
    "MAX_CHUNKS_PER_SOURCE",
    "MAX_CONTEXT_CHARS",
    "MAX_CONTEXT_CHUNKS",
    "MAX_RERANK_CANDIDATES",
    "MIN_RERANK_CANDIDATES",
    "MODERATE_RRF_THRESHOLD",
    "RERANK_ABSTAIN_FLOOR",
    "RERANK_CANDIDATE_LIMIT",
    "RRF_ABSTAIN_FLOOR",
    "RRF_K",
    "STRONG_RRF_THRESHOLD",
    "SYSTEM_PROMPT",
    "Candidate",
    "Evidence",
    "RAGPipeline",
    "is_greeting",
]
