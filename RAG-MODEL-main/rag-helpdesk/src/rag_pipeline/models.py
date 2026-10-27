
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# RETRIEVAL CANDIDATE
# ======================================================================


@dataclass
class Candidate:
    """
    Represents one retrieved chunk throughout the retrieval pipeline.

    A Candidate is the common data structure shared by Dense retrieval,
    BM25 retrieval, RRF fusion, evidence evaluation, and context
    selection.

    Flow:

        Dense Retrieval
              +
        BM25 Retrieval
              ↓
          Candidate Pool
              ↓
          RRF Fusion
              ↓
        Evidence Evaluation
              ↓
        Context Selection
              ↓
        LLM Generation
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    key: str
    filename: str
    chunk_id: Any
    text: str

    # Original chunk metadata.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    dense_rank: int | None = None
    dense_distance: float | None = None

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    bm25_rank: int | None = None
    bm25_score: float | None = None

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    rrf_score: float = 0.0

    # ------------------------------------------------------------------
    # Candidate stage journey tracking
    # ------------------------------------------------------------------

    in_dense: bool = False
    in_bm25: bool = False
    in_rrf: bool = False
    in_final_evidence: bool = False
    in_llm_context: bool = False

    # ------------------------------------------------------------------
    # Optional reranking
    # ------------------------------------------------------------------
    #
    # CrossEncoder reranking is currently disabled in this project.
    # The field remains for backward compatibility and for preserving
    # the pipeline's data model.
    #

    rerank_score: float | None = None

    # ------------------------------------------------------------------
    # Retrieval agreement
    # ------------------------------------------------------------------

    @property
    def in_both_retrievers(self) -> bool:
        """
        Return True when the same chunk was retrieved by both
        Dense retrieval and BM25.
        """
        return (
            self.dense_rank is not None
            and self.bm25_rank is not None
        )

    # ------------------------------------------------------------------
    # Observatory representation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Candidate into a serializable dictionary.

        This is useful for the RAG Observatory because the UI can
        inspect the complete state of a candidate without duplicating
        conversion logic.
        """
        return {
            "key": self.key,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),

            # Hoist canonical chunk metadata to top level for observatory/UI access
            "book_name": self.metadata.get("book_name") or self.metadata.get("book_title") or self.filename,
            "book_title": self.metadata.get("book_title") or self.metadata.get("book_name") or "",
            "chapter_num": self.metadata.get("chapter_num"),
            "chapter_title": self.metadata.get("chapter_title") or "",
            "section_heading": self.metadata.get("section_heading") or self.metadata.get("section_title") or "",
            "section_title": self.metadata.get("section_title") or self.metadata.get("section_heading") or "",
            "page_start": self.metadata.get("page_start") or self.metadata.get("page_number"),
            "page_end": self.metadata.get("page_end"),
            "token_count": self.metadata.get("token_count"),
            "chunk_index": self.metadata.get("chunk_index"),
            "prev_chunk_id": self.metadata.get("prev_chunk_id"),
            "next_chunk_id": self.metadata.get("next_chunk_id"),

            # Dense retrieval
            "dense_rank": self.dense_rank,
            "dense_distance": self.dense_distance,
            "in_dense": self.in_dense or (self.dense_rank is not None),

            # BM25 retrieval
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "in_bm25": self.in_bm25 or (self.bm25_rank is not None),

            # RRF
            "rrf_score": self.rrf_score,
            "in_rrf": self.in_rrf or (self.rrf_score > 0),

            # Stage journey
            "in_final_evidence": self.in_final_evidence,
            "in_llm_context": self.in_llm_context,

            # Optional reranker
            "rerank_score": self.rerank_score,

            # Agreement
            "retrieval_agreement": self.in_both_retrievers,
        }


# ======================================================================
# EVIDENCE
# ======================================================================


@dataclass
class Evidence:
    """
    Represents the strength of the retrieved evidence.

    The evidence gate combines:

    - absolute RRF floor
    - RRF rank strength
    - retriever agreement
    - source diversity
    - supporting chunks
    - retrieval mode
    """

    level: str
    should_answer: bool
    top_score: float
    score_source: str
    supporting_chunks: int
    top_gap: float | None
    agreement: bool
    top_percentile: float | None
    retrieval_mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Convert evidence information into a UI/API-friendly dictionary."""
        return {
            "level": self.level,
            "should_answer": self.should_answer,
            "top_score": self.top_score,
            "score_source": self.score_source,
            "supporting_chunks": self.supporting_chunks,
            "top_gap": self.top_gap,
            "retrieval_agreement": self.agreement,
            "top_percentile": self.top_percentile,
            "retrieval_mode": self.retrieval_mode,
            "reason": self.reason,
        }


# ======================================================================
# EVALUATION LAYER DATA MODELS
# ======================================================================


@dataclass
class CriterionScore:
    """
    Evaluation score for one criteria.
    Score is 0-100, query-relative, deterministic, and explainable.
    """
    score: float
    label: str
    definition: str
    reason: str
    supporting_chunks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "label": self.label,
            "definition": self.definition,
            "reason": self.reason,
            "supporting_chunks": self.supporting_chunks,
            "metrics": self.metrics,
        }


@dataclass
class EvaluationResult:
    """Complete multi-criteria runtime evaluation."""
    retrieval_health_score: float
    criteria: dict[str, CriterionScore]
    summary: str
    supporting_chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_health_score": round(self.retrieval_health_score, 1),
            "summary": self.summary,
            "criteria": {
                name: crit.to_dict() for name, crit in self.criteria.items()
            },
            "supporting_chunks": self.supporting_chunks,
        }


# ======================================================================
# CITATIONS & GROUNDING DATA MODELS
# ======================================================================


@dataclass
class Citation:
    """
    Represents one verified claim-level citation pointing to an exact passage.
    """
    id: int
    marker: str  # e.g. "[1]"
    claim: str
    filename: str
    chunk_id: Any
    passage: str
    retrieval_source: str = ""
    status: str = "supported"  # "supported" | "unsupported" | "partially_supported" | "parser_failure"
    verification_reason: str = ""
    kb_version: int | str = 1
    # Explain why a lexical match was accepted or rejected.  This makes a
    # citation audit useful even when a passage shares topic words but not the
    # answer-bearing fact.
    support_score: float = 0.0
    lexical_overlap: float = 0.0
    direct_answer_support: bool = False

    file_type: str = ""
    page_number: int | None = None
    page_range: str | None = None

    @property
    def source_label(self) -> str:
        page_info = ""
        if self.page_range:
            page_info = f" — pp. {self.page_range}"
        elif self.page_number and self.page_number > 0:
            page_info = f" — p. {self.page_number}"
        return f"{self.filename}{page_info}" if self.filename else "Document"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "marker": self.marker,
            "claim": self.claim,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "passage": self.passage,
            "retrieval_source": self.retrieval_source,
            "status": self.status,
            "verification_reason": self.verification_reason,
            "kb_version": self.kb_version,
            "support_score": round(self.support_score, 3),
            "lexical_overlap": round(self.lexical_overlap, 3),
            "direct_answer_support": self.direct_answer_support,
            "file_type": self.file_type,
            "page_number": self.page_number,
            "page_range": self.page_range,
            "source_label": self.source_label,
        }


@dataclass
class CitationCoverage:
    """Summary of claim support and citation coverage."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    coverage_percentage: float
    has_unsupported: bool
    unsupported_claims_list: list[str] = field(default_factory=list)
    partially_supported_claims: int = 0
    parser_failure_claims: int = 0
    parser_failures_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "coverage_percentage": round(self.coverage_percentage, 1),
            "has_unsupported": self.has_unsupported,
            "unsupported_claims_list": self.unsupported_claims_list,
            "partially_supported_claims": self.partially_supported_claims,
            "parser_failure_claims": self.parser_failure_claims,
            "parser_failures_list": self.parser_failures_list,
        }


# ======================================================================
# EVIDENCE-BACKED CONFIDENCE DATA MODEL
# ======================================================================


@dataclass
class ConfidenceResult:
    """
    Evidence-backed confidence level and explainable checklist.
    Does NOT claim 'accuracy probability', but transparent evidence grounding.
    """
    level: str  # "HIGH" | "MEDIUM" | "LOW"
    score: float  # 0-100
    reasons: list[str] = field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 1),
            "reasons": self.reasons,
            "supporting_evidence": self.supporting_evidence,
        }


# ======================================================================
# KNOWLEDGE BASE STATE DATA MODEL
# ======================================================================


@dataclass
class KBState:
    """State of the knowledge base at query time."""
    version: int
    document_count: int
    chroma_chunks: int
    bm25_chunks: int
    indexes_consistent: bool
    last_updated: str
    consistency_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "document_count": self.document_count,
            "chroma_chunks": self.chroma_chunks,
            "bm25_chunks": self.bm25_chunks,
            "indexes_consistent": self.indexes_consistent,
            "last_updated": self.last_updated,
            "consistency_message": self.consistency_message,
        }
