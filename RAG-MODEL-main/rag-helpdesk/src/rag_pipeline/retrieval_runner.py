"""Reusable retrieval execution for the agent controller and pipeline."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from src.config import VECTOR_TOP_K, BM25_TOP_K
from src.rag_pipeline.config import RRF_K
from src.rag_pipeline.models import Candidate
from src.rag_pipeline.retrieval import (
    bm25_retrieve,
    chapter_retrieve,
    dense_retrieve,
    fuse_rrf,
    page_retrieve,
)

if TYPE_CHECKING:
    from src.rag_pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFilters:
    """Metadata constraints applied during one retrieval attempt."""

    page_start: int | None = None
    page_end: int | None = None
    book_match: str | None = None
    chapter_num: int | None = None
    require_page_match: bool = False
    require_chapter_match: bool = False


@dataclass
class RetrievalAttemptResult:
    """Outcome of one retrieval attempt (before evidence evaluation)."""

    search_text: str
    retrieval_mode: str
    filters: RetrievalFilters
    tools_called: list[str] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    fused: list[Candidate] = field(default_factory=list)
    query_embedding: list[float] | None = None
    bm25_query_tokens: list[str] = field(default_factory=list)
    raw_dense_results: list[dict[str, Any]] = field(default_factory=list)
    raw_bm25_results: list[dict[str, Any]] = field(default_factory=list)
    rrf_results_snapshot: list[dict[str, Any]] = field(default_factory=list)
    rrf_calculations: list[dict[str, Any]] = field(default_factory=list)
    page_info: dict[str, Any] | None = None
    chapter_info: dict[str, Any] | None = None
    page_match_found: bool = False
    chapter_match_found: bool = False
    retrieval_ms: float = 0.0
    fingerprint: str = ""


def _attempt_fingerprint(
    search_text: str,
    mode: str,
    filters: RetrievalFilters,
) -> str:
    return "|".join(
        [
            mode,
            search_text.strip().lower(),
            str(filters.page_start),
            str(filters.page_end),
            str(filters.book_match or ""),
            str(filters.chapter_num or ""),
            str(filters.require_page_match),
            str(filters.require_chapter_match),
        ]
    )


def _filter_fused_for_constraints(
    fused: list[Candidate],
    filters: RetrievalFilters,
) -> list[Candidate]:
    if filters.require_page_match:
        page_only = [
            c
            for c in fused
            if c.metadata.get("is_page_match")
        ]
        if page_only:
            return page_only
    if filters.require_chapter_match:
        chapter_only = [
            c
            for c in fused
            if c.metadata.get("is_chapter_match")
        ]
        if chapter_only:
            return chapter_only
    return fused


def execute_retrieval_attempt(
    pipeline: RAGPipeline,
    *,
    search_text: str,
    retrieval_mode: str,
    filters: RetrievalFilters,
    page_info: dict[str, Any] | None = None,
    chapter_info: dict[str, Any] | None = None,
    skip_page_chapter_prefetch: bool = False,
) -> RetrievalAttemptResult:
    """
    Run dense/BM25/hybrid retrieval plus optional page/chapter lookups.

    This mirrors the pipeline's retrieval stage but accepts explicit filters
    and mode so the controller can retry with different strategies.
    """
    mode_aliases = {
        "vector": "vector",
        "dense": "vector",
        "bm25": "bm25",
        "hybrid": "hybrid",
        "both": "hybrid",
    }
    retrieval_mode = mode_aliases.get(
        str(retrieval_mode).strip().lower(),
        "hybrid",
    )

    tools: list[str] = []
    t0 = time.perf_counter()

    result = RetrievalAttemptResult(
        search_text=search_text,
        retrieval_mode=retrieval_mode,
        filters=filters,
        page_info=page_info,
        chapter_info=chapter_info,
        fingerprint=_attempt_fingerprint(search_text, retrieval_mode, filters),
    )

    try:
        collection_count = pipeline.vector_store.get_collection_count()
    except Exception:
        collection_count = 0

    page_candidates: dict[str, Candidate] = {}
    chapter_candidates: dict[str, Candidate] = {}

    if not skip_page_chapter_prefetch:
        if filters.page_start is not None:
            tools.append("page_retrieve")
            page_retrieve(
                vector_store=pipeline.vector_store,
                page_start=filters.page_start,
                page_end=filters.page_end,
                book_title=filters.book_match,
                candidates=page_candidates,
            )
            result.page_match_found = bool(page_candidates)

        if filters.chapter_num is not None:
            tools.append("chapter_retrieve")
            chapter_retrieve(
                vector_store=pipeline.vector_store,
                chapter_num=filters.chapter_num,
                book_title=filters.book_match,
                candidates=chapter_candidates,
            )
            result.chapter_match_found = bool(chapter_candidates)

    if filters.require_page_match and not page_candidates:
        tools.append("page_not_found_short_circuit")
        result.tools_called = tools
        result.retrieval_ms = (time.perf_counter() - t0) * 1000
        return result

    is_pure_page = bool(
        page_info
        and page_info.get("is_pure_page_query")
        and page_candidates
    )
    is_pure_chapter = bool(
        chapter_info
        and chapter_info.get("is_chapter_query")
        and chapter_candidates
        and not page_candidates
    )

    candidates: dict[str, Candidate] = {}
    query_embedding: list[float] | None = None
    bm25_query_tokens: list[str] = []

    if is_pure_page:
        candidates.update(page_candidates)
        tools.append("page_only_short_circuit")
        if pipeline.bm25 is not None:
            try:
                bm25_query_tokens = pipeline.bm25._tokenize(search_text)
            except Exception:
                bm25_query_tokens = []
    elif is_pure_chapter:
        candidates.update(chapter_candidates)
        tools.append("chapter_only_short_circuit")
        if pipeline.bm25 is not None:
            try:
                bm25_query_tokens = pipeline.bm25._tokenize(search_text)
            except Exception:
                bm25_query_tokens = []
    else:
        need_dense = (
            collection_count > 0
            and retrieval_mode in {"vector", "hybrid"}
        )
        need_bm25 = retrieval_mode in {"bm25", "hybrid"}

        if retrieval_mode == "hybrid" and need_dense:
            tools.extend(["dense_retrieve", "bm25_retrieve"])

            def _run_dense() -> None:
                nonlocal query_embedding
                try:
                    query_embedding = pipeline.embedding_service.create_embedding(
                        search_text
                    )
                    if query_embedding:
                        dense = dense_retrieve(
                            pipeline.vector_store,
                            query_embedding,
                            pipeline.top_k,
                        )
                        for c in dense.values():
                            c.in_dense = True
                        candidates.update(dense)
                except Exception as exc:
                    logger.warning("Concurrent dense retrieval failed: %s", exc)

            def _run_bm25_tokenize() -> None:
                nonlocal bm25_query_tokens
                try:
                    if pipeline.bm25 is not None:
                        bm25_query_tokens = pipeline.bm25._tokenize(search_text)
                except Exception:
                    bm25_query_tokens = []

            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                f_dense = pool.submit(_run_dense)
                f_bm25t = pool.submit(_run_bm25_tokenize)
                f_dense.result()
                f_bm25t.result()

            if need_bm25:
                bm25_retrieve(pipeline.bm25, search_text, candidates)
                for c in candidates.values():
                    if c.bm25_rank is not None:
                        c.in_bm25 = True

        else:
            if need_dense:
                tools.append("dense_retrieve")
                try:
                    query_embedding = pipeline.embedding_service.create_embedding(
                        search_text
                    )
                except Exception as exc:
                    logger.warning("Query embedding failed: %s", exc)

                if query_embedding:
                    dense = dense_retrieve(
                        pipeline.vector_store,
                        query_embedding,
                        pipeline.top_k,
                    )
                    for c in dense.values():
                        c.in_dense = True
                    candidates.update(dense)

            if pipeline.bm25 is not None:
                try:
                    bm25_query_tokens = pipeline.bm25._tokenize(search_text)
                except Exception:
                    bm25_query_tokens = []

            if need_bm25:
                tools.append("bm25_retrieve")
                bm25_retrieve(pipeline.bm25, search_text, candidates)
                for c in candidates.values():
                    if c.bm25_rank is not None:
                        c.in_bm25 = True
            elif retrieval_mode == "vector" and not candidates and pipeline.bm25:
                tools.append("bm25_retrieve_fallback")
                bm25_retrieve(pipeline.bm25, search_text, candidates)
                for c in candidates.values():
                    if c.bm25_rank is not None:
                        c.in_bm25 = True

        if page_candidates:
            candidates.update(page_candidates)
        if chapter_candidates:
            candidates.update(chapter_candidates)

    result.candidates = candidates
    result.query_embedding = query_embedding
    result.bm25_query_tokens = bm25_query_tokens

    result.raw_dense_results = [
        c.to_dict()
        for c in sorted(
            candidates.values(),
            key=lambda c: (c.dense_rank or 9999),
        )
        if c.dense_rank is not None
    ]
    result.raw_bm25_results = [
        c.to_dict()
        for c in sorted(
            candidates.values(),
            key=lambda c: (c.bm25_rank or 9999),
        )
        if c.bm25_rank is not None
    ]

    if not candidates:
        result.retrieval_ms = (time.perf_counter() - t0) * 1000
        result.tools_called = tools
        return result

    if retrieval_mode == "hybrid":
        tools.append("rrf_fusion")
        fused = fuse_rrf(candidates)
        for calc in fused:
            dense_contribution = None
            bm25_contribution = None
            if calc.dense_rank is not None:
                dense_contribution = 1.0 / (RRF_K + calc.dense_rank)
            if calc.bm25_rank is not None:
                bm25_contribution = 1.0 / (RRF_K + calc.bm25_rank)
            result.rrf_calculations.append(
                {
                    "filename": calc.filename,
                    "chunk_id": calc.chunk_id,
                    "dense_rank": calc.dense_rank,
                    "bm25_rank": calc.bm25_rank,
                    "k": RRF_K,
                    "dense_contribution": dense_contribution,
                    "bm25_contribution": bm25_contribution,
                    "rrf_score": calc.rrf_score,
                }
            )
    elif retrieval_mode == "vector":
        fused = sorted(
            candidates.values(),
            key=lambda c: (c.dense_rank if c.dense_rank is not None else 9999),
        )
    elif retrieval_mode == "bm25":
        fused = sorted(
            candidates.values(),
            key=lambda c: (c.bm25_rank if c.bm25_rank is not None else 9999),
        )
    else:
        fused = []

    for c in fused:
        c.in_rrf = retrieval_mode == "hybrid"

    fused = _filter_fused_for_constraints(fused, filters)

    result.fused = fused
    result.rrf_results_snapshot = [c.to_dict() for c in fused]
    result.tools_called = tools
    result.retrieval_ms = (time.perf_counter() - t0) * 1000
    return result
