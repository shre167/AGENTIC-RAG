"""
RAG Observatory
===============

Runtime diagnostics and Retrieval Lab helpers for the Astronomy RAG system.

This module observes the runtime trace produced by RAGPipeline.

It provides:
- Query processing diagnostics
- Dense / vector retrieval diagnostics
- BM25 retrieval diagnostics
- Hybrid retrieval diagnostics
- Explicit RRF calculations
- Evidence-gate diagnostics
- Context optimization diagnostics
- Generation diagnostics
- Source statistics
- Dense vs BM25 comparison
- Safe trace export

This module does NOT perform retrieval itself.
It only reads and organizes the runtime trace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _safe(value: Any, default: Any = None) -> Any:
    """Return default when value is None."""
    return default if value is None else value


def _as_list(value: Any) -> List[Any]:
    """Safely convert a value into a list."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        return list(value)
    except Exception:
        return []


def _as_dict(value: Any) -> Dict[str, Any]:
    """Safely convert a value into a dictionary."""
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    try:
        return dict(value)
    except Exception:
        return {}


def _round(value: Any, digits: int = 6) -> Any:
    """Safely round numeric values."""
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _text(
    value: Any,
    max_chars: Optional[int] = None,
) -> str:
    """Safely convert a value to string."""
    if value is None:
        return ""

    result = str(value)

    if max_chars is not None:
        return result[:max_chars]

    return result


# ============================================================================
# RETRIEVAL MODES
# ============================================================================

MODE_ALIASES = {
    "hybrid": "hybrid",
    "both": "hybrid",

    "dense": "dense",
    "vector": "dense",
    "dense_only": "dense",
    "vector_only": "dense",

    "bm25": "bm25",
    "bm25_only": "bm25",
    "lexical": "bm25",
}


def normalize_retrieval_mode(mode: Any) -> str:
    """
    Normalize retrieval mode.

    Returns:
        hybrid
        dense
        bm25
    """
    value = _text(mode).strip().lower()

    return MODE_ALIASES.get(value, "hybrid")


def retrieval_mode_label(mode: Any) -> str:
    """Return a human-readable retrieval mode."""
    normalized = normalize_retrieval_mode(mode)

    labels = {
        "hybrid": "Hybrid — Dense + BM25 + RRF",
        "dense": "Dense / Vector Only",
        "bm25": "BM25 / Lexical Only",
    }

    return labels[normalized]


# ============================================================================
# QUERY OBSERVATORY
# ============================================================================

def get_query_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract query-processing information from the pipeline trace.
    """

    trace = _as_dict(trace)

    query = _as_dict(
        trace.get("query")
    )

    original = (
        query.get("original")
        or query.get("original_query")
        or query.get("question")
        or ""
    )

    prepared = (
        query.get("prepared")
        or query.get("prepared_query")
        or query.get("search_text")
        or ""
    )

    normalization = query.get(
        "normalization",
        query.get("normalized", ""),
    )

    bm25_tokens = query.get(
        "bm25_tokens",
        [],
    )

    if isinstance(bm25_tokens, str):
        bm25_tokens = bm25_tokens.split()

    bm25_tokens = _as_list(
        bm25_tokens
    )

    return {
        "original_query": _text(
            original
        ),
        "prepared_query": _text(
            prepared
        ),
        "normalization": _text(
            normalization
        ),
        "bm25_tokens": bm25_tokens,
        "bm25_token_count": len(
            bm25_tokens
        ),
        "is_multi_intent": bool(
            query.get(
                "is_multi_intent",
                False,
            )
        ),
        "is_greeting": bool(
            query.get(
                "greeting",
                False,
            )
        ),
    }


# ============================================================================
# DENSE RETRIEVAL
# ============================================================================

def get_dense_results(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract raw Dense / Chroma retrieval results.

    Expected source:

        trace["retrieval"]["raw_dense"]
    """

    trace = _as_dict(trace)

    retrieval = _as_dict(
        trace.get("retrieval")
    )

    raw_dense = retrieval.get(
        "raw_dense",
        [],
    )

    results: List[Dict[str, Any]] = []

    for index, item in enumerate(
        _as_list(raw_dense),
        start=1,
    ):
        row = _as_dict(item)

        rank = row.get(
            "dense_rank",
            row.get(
                "rank",
                index,
            ),
        )

        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        raw_b = row.get("book") or row.get("book_title") or meta.get("book_title") or row.get("filename") or ""
        clean_b = Path(raw_b).stem if str(raw_b).lower().endswith(".pdf") else str(raw_b)
        clean_b = clean_b.replace(" (1)", "").replace("_", " ").strip() or "Unknown"

        ch_num = row.get("chapter_num") if row.get("chapter_num") is not None else meta.get("chapter_num")
        ch_title = row.get("chapter_title") or meta.get("chapter_title") or ""

        results.append(
            {
                "rank": rank,

                "filename": row.get(
                    "filename",
                    "",
                ),

                "book": clean_b,
                "book_title": clean_b,
                "chapter_num": ch_num,
                "chapter": f"Ch. {ch_num}" if ch_num and int(ch_num) > 0 else ("Intro" if ch_num == 0 else "-"),
                "chapter_title": ch_title,

                "chunk_id": row.get(
                    "chunk_id",
                    "",
                ),

                "distance": _round(
                    row.get(
                        "dense_distance",
                        row.get(
                            "distance"
                        ),
                    )
                ),

                "score": _round(
                    row.get(
                        "score"
                    )
                ),

                "text": row.get(
                    "text",
                    "",
                ),

                "section_heading": row.get(
                    "section_heading",
                    "",
                ),

                "section_path": row.get(
                    "section_path",
                    "",
                ),

                "chunk_index": row.get(
                    "chunk_index"
                ),

                "total_section_chunks": row.get(
                    "total_section_chunks"
                ),

                "category": row.get(
                    "category",
                    "",
                ),
            }
        )

    return results


# ============================================================================
# BM25 RETRIEVAL
# ============================================================================

def get_bm25_results(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract raw BM25 retrieval results.

    Expected source:

        trace["retrieval"]["raw_bm25"]
    """

    trace = _as_dict(trace)

    retrieval = _as_dict(
        trace.get("retrieval")
    )

    raw_bm25 = retrieval.get(
        "raw_bm25",
        [],
    )

    results: List[Dict[str, Any]] = []

    for index, item in enumerate(
        _as_list(raw_bm25),
        start=1,
    ):
        row = _as_dict(item)

        rank = row.get(
            "bm25_rank",
            row.get(
                "rank",
                index,
            ),
        )

        results.append(
            {
                "rank": rank,

                "filename": row.get(
                    "filename",
                    "",
                ),

                "chunk_id": row.get(
                    "chunk_id",
                    "",
                ),

                "score": _round(
                    row.get(
                        "bm25_score",
                        row.get(
                            "score"
                        ),
                    )
                ),

                "text": row.get(
                    "text",
                    "",
                ),

                "section_heading": row.get(
                    "section_heading",
                    "",
                ),

                "section_path": row.get(
                    "section_path",
                    "",
                ),

                "chunk_index": row.get(
                    "chunk_index"
                ),

                "total_section_chunks": row.get(
                    "total_section_chunks"
                ),

                "category": row.get(
                    "category",
                    "",
                ),
            }
        )

    return results


# ============================================================================
# RRF
# ============================================================================

def calculate_rrf_contribution(
    rank: Optional[int],
    k: int = 60,
) -> float:
    """
    Calculate a single RRF contribution.

    Formula:

        1 / (K + rank)

    Missing rank = 0 contribution.
    """

    if rank is None:
        return 0.0

    try:
        rank = int(rank)
        k = int(k)

        if rank <= 0:
            return 0.0

        return 1.0 / (
            k + rank
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def calculate_rrf_score(
    dense_rank: Optional[int],
    bm25_rank: Optional[int],
    k: int = 60,
) -> Dict[str, Any]:
    """
    Produce a complete explainable RRF calculation.

    Example:

        Dense rank = 2
        BM25 rank = 1
        K = 60

        Dense:
            1 / (60 + 2)
            = 0.016129

        BM25:
            1 / (60 + 1)
            = 0.016393

        RRF:
            0.016129 + 0.016393
            = 0.032522
    """

    dense_contribution = (
        calculate_rrf_contribution(
            dense_rank,
            k,
        )
    )

    bm25_contribution = (
        calculate_rrf_contribution(
            bm25_rank,
            k,
        )
    )

    rrf_score = (
        dense_contribution
        + bm25_contribution
    )

    dense_formula = None
    bm25_formula = None

    if dense_rank is not None:
        dense_formula = (
            f"1 / ({k} + {dense_rank}) = "
            f"{dense_contribution:.6f}"
        )

    if bm25_rank is not None:
        bm25_formula = (
            f"1 / ({k} + {bm25_rank}) = "
            f"{bm25_contribution:.6f}"
        )

    formula_parts = []

    if dense_rank is not None:
        formula_parts.append(
            f"{dense_contribution:.6f}"
        )

    if bm25_rank is not None:
        formula_parts.append(
            f"{bm25_contribution:.6f}"
        )

    formula = ""

    if formula_parts:
        formula = (
            " + ".join(formula_parts)
            + f" = {rrf_score:.6f}"
        )

    return {
        "k": k,

        "dense_rank": dense_rank,

        "bm25_rank": bm25_rank,

        "dense_contribution": (
            dense_contribution
        ),

        "bm25_contribution": (
            bm25_contribution
        ),

        "dense_formula": dense_formula,

        "bm25_formula": bm25_formula,

        "formula": formula,

        "rrf_score": rrf_score,

        "agreement": (
            dense_rank is not None
            and bm25_rank is not None
        ),
    }


def _normalize_rrf_row(
    row: Any,
) -> Dict[str, Any]:
    """
    Normalize an existing RRF row.
    """

    row = _as_dict(row)

    k = row.get(
        "k",
        60,
    )

    dense_rank = row.get(
        "dense_rank"
    )

    bm25_rank = row.get(
        "bm25_rank"
    )

    calculated = calculate_rrf_score(
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        k=k,
    )

    result = {
        **calculated,
        **row,
    }

    result[
        "dense_contribution"
    ] = _round(
        result.get(
            "dense_contribution"
        ),
        6,
    )

    result[
        "bm25_contribution"
    ] = _round(
        result.get(
            "bm25_contribution"
        ),
        6,
    )

    result[
        "rrf_score"
    ] = _round(
        result.get(
            "rrf_score"
        ),
        6,
    )

    return result


def get_rrf_calculations(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Get actual RRF calculations.

    If the pipeline already records:

        trace["retrieval"]["rrf_calculation"]

    those rows are used.

    Otherwise the calculations are reconstructed from:

        trace["retrieval"]["rrf_results"]
    """

    trace = _as_dict(trace)

    retrieval = _as_dict(
        trace.get("retrieval")
    )

    calculations = retrieval.get(
        "rrf_calculation",
        [],
    )

    if calculations:

        return [
            _normalize_rrf_row(
                row
            )
            for row in _as_list(
                calculations
            )
        ]

    results = retrieval.get(
        "rrf_results",
        [],
    )

    if not results:
        return []

    k = retrieval.get(
        "rrf_k",
        60,
    )

    output: List[Dict[str, Any]] = []

    for index, item in enumerate(
        _as_list(results),
        start=1,
    ):
        row = _as_dict(item)

        dense_rank = row.get(
            "dense_rank"
        )

        bm25_rank = row.get(
            "bm25_rank"
        )

        calculation = (
            calculate_rrf_score(
                dense_rank=dense_rank,
                bm25_rank=bm25_rank,
                k=k,
            )
        )

        # Round for display so reconstructed rows match the precision of
        # rows the pipeline recorded itself.
        for field in (
            "dense_contribution",
            "bm25_contribution",
            "rrf_score",
        ):
            calculation[field] = _round(
                calculation[field],
                6,
            )

        calculation.update(
            {
                "final_rank": row.get(
                    "final_rank",
                    row.get(
                        "rank",
                        index,
                    ),
                ),

                "filename": row.get(
                    "filename",
                    "",
                ),

                "chunk_id": row.get(
                    "chunk_id",
                    "",
                ),

                "text": row.get(
                    "text",
                    "",
                ),

                "retrieval_agreement": row.get(
                    "retrieval_agreement",
                    calculation[
                        "agreement"
                    ],
                ),
            }
        )

        output.append(
            calculation
        )

    return output


# ============================================================================
# RETRIEVAL OBSERVATORY
# ============================================================================

def get_retrieval_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the complete retrieval diagnostics object.
    """

    trace = _as_dict(trace)

    retrieval = _as_dict(
        trace.get("retrieval")
    )

    # The pipeline may record the mode under any of these keys depending
    # on which code path produced the trace. The first non-empty value
    # wins; nothing is invented when all of them are absent.
    raw_mode = None

    for key in (
        "selected_mode",
        "mode",
        "retrieval_mode",
    ):
        value = retrieval.get(key)

        if value:
            raw_mode = value
            break

    if raw_mode is None:
        raw_mode = trace.get(
            "retrieval_mode",
            "hybrid",
        )

    mode = normalize_retrieval_mode(
        raw_mode
    )

    dense_results = (
        get_dense_results(
            trace
        )
    )

    bm25_results = (
        get_bm25_results(
            trace
        )
    )

    rrf_calculations = (
        get_rrf_calculations(
            trace
        )
    )

    rrf_enabled = bool(
        retrieval.get(
            "rrf_enabled",
            mode == "hybrid",
        )
    )

    return {
        "mode": mode,

        "mode_label": retrieval_mode_label(
            mode
        ),

        "dense_enabled": (
            mode in {
                "dense",
                "hybrid",
            }
        ),

        "bm25_enabled": (
            mode in {
                "bm25",
                "hybrid",
            }
        ),

        "rrf_enabled": rrf_enabled,

        "rrf_k": retrieval.get(
            "rrf_k",
            60,
        ),

        "dense_count": len(
            dense_results
        ),

        "bm25_count": len(
            bm25_results
        ),

        "rrf_count": len(
            rrf_calculations
        ),

        "candidate_count": retrieval.get(
            "candidate_count"
        ),

        "fused_count": retrieval.get(
            "fused_count"
        ),

        "reranked": bool(
            retrieval.get(
                "reranked",
                False,
            )
        ),

        "reranker_enabled": bool(
            retrieval.get(
                "reranker_enabled",
                False,
            )
        ),

        "query_embedding_dim": retrieval.get(
            "query_embedding_dim"
        ),

        "vector_top_k": retrieval.get(
            "vector_top_k"
        ),

        "bm25_top_k": retrieval.get(
            "bm25_top_k"
        ),

        "hybrid_top_k": retrieval.get(
            "hybrid_top_k"
        ),

        "raw_dense": dense_results,

        "raw_bm25": bm25_results,

        "rrf_calculation": rrf_calculations,
    }


# ============================================================================
# EVIDENCE OBSERVATORY
# ============================================================================

def get_evidence_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract evidence-gate diagnostics.

    Supports both:

        trace["evidence_gate"]

    and:

        trace["evidence"]
    """

    trace = _as_dict(trace)

    evidence = _as_dict(
        trace.get(
            "evidence_gate"
        )
        or trace.get(
            "evidence"
        )
    )

    return {
        "level": evidence.get(
            "level",
            "unknown",
        ),

        "should_answer": bool(
            evidence.get(
                "should_answer",
                False,
            )
        ),

        "top_score": _round(
            evidence.get(
                "top_score"
            )
        ),

        "score_source": evidence.get(
            "score_source",
            "",
        ),

        "supporting_chunks": evidence.get(
            "supporting_chunks",
            0,
        ),

        "top_gap": _round(
            evidence.get(
                "top_gap"
            )
        ),

        "agreement": bool(
            evidence.get(
                "agreement",
                False,
            )
        ),

        "top_percentile": _round(
            evidence.get(
                "top_percentile"
            )
        ),

        "retrieval_mode": evidence.get(
            "retrieval_mode",
            "",
        ),

        "reason": evidence.get(
            "reason",
            "",
        ),
    }


# ============================================================================
# CONTEXT OPTIMIZER
# ============================================================================

def get_context_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract context optimization information.
    """

    trace = _as_dict(trace)

    optimizer = _as_dict(
        trace.get(
            "token_optimizer"
        )
    )

    context = _as_dict(
        trace.get(
            "context"
        )
    )

    return {
        "input_chunks": optimizer.get(
            "input_chunks",
            optimizer.get(
                "candidate_count",
                0,
            ),
        ),

        "selected_chunks": optimizer.get(
            "selected_chunks",
            context.get(
                "selected_chunks",
                context.get(
                    "chunks",
                    0,
                ),
            ),
        ),

        "removed_chunks": optimizer.get(
            "removed_chunks",
            0,
        ),

        "input_chars": optimizer.get(
            "input_chars",
            0,
        ),

        "selected_chars": optimizer.get(
            "selected_chars",
            context.get(
                "context_chars",
                context.get(
                    "chars",
                    0,
                ),
            ),
        ),

        "source_diversity": optimizer.get(
            "source_diversity",
            context.get(
                "unique_sources",
                context.get(
                    "sources",
                    0,
                ),
            ),
        ),

        "max_chunks": optimizer.get(
            "max_chunks"
        ),

        "max_chars": optimizer.get(
            "max_chars"
        ),

        "context_chars": context.get(
            "context_chars",
            context.get(
                "chars",
                0,
            ),
        ),

        "unique_sources": context.get(
            "unique_sources",
            context.get(
                "sources",
                [],
            ),
        ),
    }


# ============================================================================
# GENERATION OBSERVATORY
# ============================================================================

def get_generation_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract LLM generation diagnostics.

    API keys are deliberately ignored.
    """

    trace = _as_dict(trace)

    generation = _as_dict(
        trace.get(
            "generation"
        )
    )

    return {
        "status": generation.get(
            "status",
            "unknown",
        ),

        "model": generation.get(
            "model",
            "",
        ),

        "base_url": generation.get(
            "base_url",
            "",
        ),

        "question": generation.get(
            "question",
            "",
        ),

        "context_chars": generation.get(
            "context_chars",
            0,
        ),

        "message_count": generation.get(
            "message_count",
            0,
        ),

        "temperature": generation.get(
            "temperature"
        ),

        "response_chars": generation.get(
            "response_chars",
            0,
        ),

        "response_preview": generation.get(
            "response_preview",
            "",
        ),

        "explanation": generation.get(
            "explanation",
            "",
        ),

        "procedural": bool(
            generation.get(
                "procedural",
                False,
            )
        ),

        "error": generation.get(
            "error",
            "",
        ),
    }


# ============================================================================
# ANSWER OBSERVATORY
# ============================================================================

def get_answer_observatory(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a compact representation of one complete RAG execution.
    """

    trace = _as_dict(trace)

    query = get_query_observatory(
        trace
    )

    retrieval = get_retrieval_observatory(
        trace
    )

    evidence = get_evidence_observatory(
        trace
    )

    context = get_context_observatory(
        trace
    )

    generation = get_generation_observatory(
        trace
    )

    sources = context.get(
        "unique_sources",
        [],
    )

    if not isinstance(
        sources,
        list,
    ):
        sources = []

    return {
        "query": query,

        "retrieval": retrieval,

        "evidence": evidence,

        "context": context,

        "generation": generation,

        "sources": sources,

        "retrieved_count": (
            retrieval.get(
                "candidate_count"
            )
            or retrieval.get(
                "dense_count",
                0,
            )
            or retrieval.get(
                "bm25_count",
                0,
            )
        ),

        "trace_available": bool(
            trace
        ),
    }


# ============================================================================
# DENSE VS BM25 COMPARISON
# ============================================================================

def compare_retrieval_results(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare Dense and BM25 results for the same chunks.

    Example:

        filename
        chunk_id
        dense_rank
        bm25_rank
        agreement
    """

    dense = get_dense_results(
        trace
    )

    bm25 = get_bm25_results(
        trace
    )

    combined: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # ------------------------------------------------------------------
    # Dense results
    # ------------------------------------------------------------------

    for row in dense:

        filename = row.get(
            "filename",
            "",
        )

        chunk_id = row.get(
            "chunk_id",
            "",
        )

        key = (
            f"{filename}::"
            f"{chunk_id}"
        )

        if key not in combined:
            combined[key] = {
                "filename": filename,
                "chunk_id": chunk_id,
                "text": row.get(
                    "text",
                    "",
                ),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "bm25_score": None,
                "agreement": False,
            }

        combined[key][
            "dense_rank"
        ] = row.get(
            "rank"
        )

        combined[key][
            "dense_distance"
        ] = row.get(
            "distance"
        )

    # ------------------------------------------------------------------
    # BM25 results
    # ------------------------------------------------------------------

    for row in bm25:

        filename = row.get(
            "filename",
            "",
        )

        chunk_id = row.get(
            "chunk_id",
            "",
        )

        key = (
            f"{filename}::"
            f"{chunk_id}"
        )

        if key not in combined:
            combined[key] = {
                "filename": filename,
                "chunk_id": chunk_id,
                "text": row.get(
                    "text",
                    "",
                ),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "bm25_score": None,
                "agreement": False,
            }

        combined[key][
            "bm25_rank"
        ] = row.get(
            "rank"
        )

        combined[key][
            "bm25_score"
        ] = row.get(
            "score"
        )

    # ------------------------------------------------------------------
    # Agreement
    # ------------------------------------------------------------------

    for row in combined.values():

        row["agreement"] = (
            row["dense_rank"] is not None
            and row["bm25_rank"] is not None
        )

    return list(
        combined.values()
    )


# ============================================================================
# SOURCE STATISTICS
# ============================================================================

def get_source_statistics(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate retrieval results by source file.
    """

    retrieval = get_retrieval_observatory(
        trace
    )

    dense = retrieval.get(
        "raw_dense",
        [],
    )

    bm25 = retrieval.get(
        "raw_bm25",
        [],
    )

    stats: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    for row in dense:

        filename = row.get(
            "filename",
            "",
        )

        if not filename:
            continue

        if filename not in stats:
            stats[filename] = {
                "filename": filename,
                "dense_hits": 0,
                "bm25_hits": 0,
                "total_hits": 0,
                "best_dense_rank": None,
                "best_bm25_rank": None,
            }

        item = stats[filename]

        item[
            "dense_hits"
        ] += 1

        rank = row.get(
            "rank"
        )

        if (
            rank is not None
            and (
                item[
                    "best_dense_rank"
                ] is None
                or rank
                < item[
                    "best_dense_rank"
                ]
            )
        ):
            item[
                "best_dense_rank"
            ] = rank

        item[
            "total_hits"
        ] += 1

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    for row in bm25:

        filename = row.get(
            "filename",
            "",
        )

        if not filename:
            continue

        if filename not in stats:
            stats[filename] = {
                "filename": filename,
                "dense_hits": 0,
                "bm25_hits": 0,
                "total_hits": 0,
                "best_dense_rank": None,
                "best_bm25_rank": None,
            }

        item = stats[filename]

        item[
            "bm25_hits"
        ] += 1

        rank = row.get(
            "rank"
        )

        if (
            rank is not None
            and (
                item[
                    "best_bm25_rank"
                ] is None
                or rank
                < item[
                    "best_bm25_rank"
                ]
            )
        ):
            item[
                "best_bm25_rank"
            ] = rank

        item[
            "total_hits"
        ] += 1

    return sorted(
        stats.values(),
        key=lambda item: item[
            "total_hits"
        ],
        reverse=True,
    )


# ============================================================================
# CHUNK HELPERS
# ============================================================================

def format_chunk_identifier(
    filename: Any,
    chunk_id: Any,
) -> str:
    """Return a readable chunk identifier."""

    filename = (
        _text(filename)
        or "unknown"
    )

    chunk_id = (
        _text(chunk_id)
        or "unknown"
    )

    return (
        f"{filename} / "
        f"{chunk_id}"
    )


def truncate_text(
    text: Any,
    max_chars: int = 500,
) -> str:
    """Truncate text for tables/cards."""

    text = _text(text)

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        .rstrip()
        + "…"
    )


# ============================================================================
# COMPLETE OBSERVATORY SNAPSHOT
# ============================================================================

def build_observatory_snapshot(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the complete runtime snapshot.

    This is the main function the Streamlit UI should use.
    """

    trace = _as_dict(trace)

    answer = get_answer_observatory(
        trace
    )

    return {
        "query": answer[
            "query"
        ],

        "retrieval": answer[
            "retrieval"
        ],

        "evidence": answer[
            "evidence"
        ],

        "context": answer[
            "context"
        ],

        "generation": answer[
            "generation"
        ],

        "sources": answer[
            "sources"
        ],

        "source_statistics": (
            get_source_statistics(
                trace
            )
        ),

        "retrieval_comparison": (
            compare_retrieval_results(
                trace
            )
        ),

        "evaluation": get_evaluation_diagnostics(trace),
        "citations": get_citations_diagnostics(trace),
        "citation_coverage": get_citation_coverage_diagnostics(trace),
        "grounding": get_grounding_diagnostics(trace),
        "confidence": get_confidence_diagnostics(trace),
        "kb_state": get_kb_state_diagnostics(trace),
        "candidate_journey": get_candidate_journey_diagnostics(trace),

        "trace_available": answer[
            "trace_available"
        ],
    }


def get_evaluation_diagnostics(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    trace = _as_dict(trace)
    return _as_dict(trace.get("evaluation"))


def get_citations_diagnostics(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace = _as_dict(trace)
    return _as_list(trace.get("citations"))


def get_citation_coverage_diagnostics(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    trace = _as_dict(trace)
    return _as_dict(trace.get("citation_coverage"))


def get_grounding_diagnostics(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the explicit Claim -> Evidence -> Citation post-generation chain."""
    trace = _as_dict(trace)
    grounding = _as_dict(trace.get("grounding"))
    if grounding:
        return grounding
    # Backward-compatible reconstruction for traces recorded before grounding.
    citations = _as_list(trace.get("citations"))
    return {
        "stage": "post_generation",
        "claim_count": len(citations),
        "claim_to_citation": [
            {
                "claim": item.get("claim", ""),
                "citation": item.get("marker", ""),
                "status": item.get("status", "unknown"),
                "evidence": {
                    "filename": item.get("filename", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "passage": item.get("passage", ""),
                },
            }
            for item in citations
        ],
    }


def get_confidence_diagnostics(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    trace = _as_dict(trace)
    return _as_dict(trace.get("confidence"))


def get_kb_state_diagnostics(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    trace = _as_dict(trace)
    return _as_dict(trace.get("kb_state"))


def get_candidate_journey_diagnostics(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace = _as_dict(trace)
    return _as_list(trace.get("candidate_journey"))


# ============================================================================
# COMPACT STATUS
# ============================================================================

def get_observatory_status(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return a compact status object for the top of the Retrieval Lab.
    """

    snapshot = build_observatory_snapshot(
        trace
    )

    retrieval = snapshot[
        "retrieval"
    ]

    evidence = snapshot[
        "evidence"
    ]

    generation = snapshot[
        "generation"
    ]

    return {
        "trace_available": snapshot[
            "trace_available"
        ],

        "retrieval_mode": retrieval[
            "mode"
        ],

        "retrieval_mode_label": retrieval[
            "mode_label"
        ],

        "dense_enabled": retrieval[
            "dense_enabled"
        ],

        "bm25_enabled": retrieval[
            "bm25_enabled"
        ],

        "rrf_enabled": retrieval[
            "rrf_enabled"
        ],

        "rrf_k": retrieval[
            "rrf_k"
        ],

        "dense_results": retrieval[
            "dense_count"
        ],

        "bm25_results": retrieval[
            "bm25_count"
        ],

        "rrf_results": retrieval[
            "rrf_count"
        ],

        "evidence": evidence[
            "level"
        ],

        "should_answer": evidence[
            "should_answer"
        ],

        "generation_status": generation[
            "status"
        ],

        "model": generation[
            "model"
        ],
    }


# ============================================================================
# SAFE TRACE EXPORT
# ============================================================================

def export_trace_json(
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return a sanitized trace suitable for JSON export/debugging.

    Secret-looking fields are removed.
    """

    trace = _as_dict(trace)

    secret_words = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "credential",
    )

    def sanitize(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            dict,
        ):

            result = {}

            for key, item in value.items():

                key_text = str(
                    key
                ).lower()

                if any(
                    secret_word
                    in key_text
                    for secret_word
                    in secret_words
                ):
                    continue

                result[key] = sanitize(
                    item
                )

            return result

        if isinstance(
            value,
            list,
        ):
            return [
                sanitize(item)
                for item in value
            ]

        if isinstance(
            value,
            tuple,
        ):
            return [
                sanitize(item)
                for item in value
            ]

        return value

    return sanitize(
        trace
    )

# ============================================================================
# CONTEXT CHUNKS (UI HELPER)
# ============================================================================

def get_context_chunks(
    trace: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return the actual chunks that were passed to the LLM.

    The pipeline may record these under any of:

        trace["context"]["final_chunks"]
        trace["context"]["selected"]
        trace["context"]["chunks"]          (only when it is a list)
        trace["token_optimizer"]["selected"]

    Nothing is fabricated. If the trace does not carry the chunk bodies,
    an empty list is returned and the UI is expected to say so.
    """

    trace = _as_dict(trace)

    context = _as_dict(
        trace.get("context")
    )

    optimizer = _as_dict(
        trace.get("token_optimizer")
    )

    raw: Any = None

    for container, key in (
        (context, "final_chunks"),
        (context, "selected_chunks"),
        (context, "selected"),
        (context, "chunks"),
        (optimizer, "final_chunks"),
        (optimizer, "selected"),
    ):
        value = container.get(key)

        # "chunks"/"selected_chunks" are integers in some traces.
        if isinstance(value, list) and value:
            raw = value
            break

    if raw is None:
        return []

    results: List[Dict[str, Any]] = []

    for index, item in enumerate(
        _as_list(raw),
        start=1,
    ):
        row = _as_dict(item)

        text = _text(
            row.get("text", "")
        )

        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        book = (
            row.get("book")
            or row.get("book_title")
            or meta.get("book_title")
            or meta.get("book_name")
            or row.get("filename", "")
        )
        if str(book).lower().endswith(".pdf"):
            book = str(book)[:-4].replace(" (1)", "").strip()

        ch_num = row.get("chapter") or row.get("chapter_num") or meta.get("chapter_num")
        sec = row.get("section") or row.get("section_heading") or meta.get("section_heading") or meta.get("section_title") or ""
        p_start = row.get("page_start") or row.get("page_number") or meta.get("page_start") or meta.get("page_number")
        p_end = row.get("page_end") or meta.get("page_end") or p_start
        p_range = row.get("page_range")
        if not p_range and p_start:
            p_range = f"{p_start}–{p_end}" if p_end and p_end != p_start else str(p_start)

        results.append(
            {
                "position": index,
                "rank": row.get("rank", index),
                "filename": row.get("filename") or meta.get("filename") or "",
                "book": book,
                "book_title": book,
                "book_name": row.get("book_name") or meta.get("book_name") or row.get("filename", ""),
                "chapter": ch_num if ch_num and int(ch_num) > 0 else None,
                "chapter_num": ch_num,
                "chapter_title": row.get("chapter_title") or meta.get("chapter_title", ""),
                "section": sec,
                "section_heading": sec,
                "page_start": p_start,
                "page_end": p_end,
                "page_range": p_range or "-",
                "token_count": row.get("token_count") or meta.get("token_count", 0),
                "chunk_id": row.get("chunk_id", ""),
                "text": text,
                "chars": len(text),
                "score": _round(row.get("rrf_score", row.get("score"))),
                "rrf_score": _round(row.get("rrf_score")),
                "dense_rank": row.get("dense_rank"),
                "dense_distance": _round(row.get("dense_distance")),
                "bm25_rank": row.get("bm25_rank"),
                "bm25_score": _round(row.get("bm25_score")),
                "source_label": row.get("source_label", ""),
                "metadata": meta,
            }
        )

    # Back-fill source_label using rich formatting on the assembled rows
    for r in results:
        if not r.get("source_label"):
            r["source_label"] = _rich_source_label(r)

    return results


def _rich_source_label(row: Dict[str, Any]) -> str:
    """Build a human-readable source label from chunk metadata."""
    book = (
        row.get("book_title")
        or row.get("book_name")
        or row.get("book")
        or row.get("filename")
        or "Unknown"
    )
    if str(book).lower().endswith(".pdf"):
        book = str(book)[:-4].replace(" (1)", "").strip()
    parts = [book]
    ch = row.get("chapter_num") or row.get("chapter")
    ch_title = row.get("chapter_title") or ""
    if ch:
        try:
            ch_int = int(ch)
            if ch_int > 0:
                ch_part = f"Chapter {ch_int}"
                if ch_title:
                    ch_part += f" — {ch_title}"
                parts.append(ch_part)
        except (ValueError, TypeError):
            pass
    p_start = row.get("page_start")
    p_end = row.get("page_end")
    if p_start:
        try:
            p_start_i = int(p_start)
            p_end_i = int(p_end) if p_end else p_start_i
            if p_end_i > p_start_i:
                parts.append(f"Pages {p_start_i}–{p_end_i}")
            else:
                parts.append(f"Page {p_start_i}")
        except (ValueError, TypeError):
            pass
    return " — ".join(parts)


def get_supporting_sources(
    trace: Optional[Dict[str, Any]],
    response: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return the sources that actually backed the answer.

    These are the sources represented in the final context, which is a
    different thing from the full candidate set produced by retrieval.
    """

    chunks = get_context_chunks(trace)

    ordered: List[str] = []

    for row in chunks:
        label = _rich_source_label(row)
        if label and label not in ordered:
            ordered.append(label)

    if ordered:
        return ordered

    context = get_context_observatory(trace)

    unique = context.get(
        "unique_sources"
    )

    if isinstance(unique, list):

        for item in unique:
            if isinstance(item, dict):
                filename = _rich_source_label(item)
            else:
                filename = _text(item)

            if filename and filename not in ordered:
                ordered.append(filename)

    if ordered:
        return ordered

    response = _as_dict(response)

    for item in _as_list(
        response.get("sources")
    ):
        if isinstance(item, dict):
            filename = _rich_source_label(item)
        else:
            filename = _text(item)

        if filename and filename not in ordered:
            ordered.append(filename)

    return ordered


def get_candidate_chunks(
    trace: Optional[Dict[str, Any]],
    response: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return the retrieval candidates for the current mode.

    Candidates are what retrieval produced. They are NOT the same as the
    supporting sources, and the UI labels them separately.
    """

    retrieval = get_retrieval_observatory(trace)

    mode = retrieval.get("mode", "hybrid")

    rows: List[Dict[str, Any]] = []

    if mode == "hybrid":

        for row in retrieval.get("rrf_calculation", []):

            rows.append(
                {
                    "rank": row.get("final_rank"),
                    "filename": row.get("filename", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "text": row.get("text", ""),
                    "score": row.get("rrf_score"),
                    "score_label": "RRF score",
                }
            )

    elif mode == "dense":

        for row in retrieval.get("raw_dense", []):

            rows.append(
                {
                    "rank": row.get("rank"),
                    "filename": row.get("filename", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "text": row.get("text", ""),
                    "score": row.get("distance"),
                    "score_label": "Vector distance",
                }
            )

    else:

        for row in retrieval.get("raw_bm25", []):

            rows.append(
                {
                    "rank": row.get("rank"),
                    "filename": row.get("filename", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "text": row.get("text", ""),
                    "score": row.get("score"),
                    "score_label": "BM25 score",
                }
            )

    if rows:
        return rows

    # Fall back to whatever the response itself carried.
    response = _as_dict(response)

    for index, item in enumerate(
        _as_list(
            response.get("retrieved_chunks")
        ),
        start=1,
    ):
        row = _as_dict(item)

        rows.append(
            {
                "rank": row.get("rank", index),
                "filename": row.get("filename", ""),
                "chunk_id": row.get("chunk_id", ""),
                "text": row.get("text", ""),
                "score": _round(
                    row.get("score")
                ),
                "score_label": "Score",
            }
        )

    return rows


def describe_evidence_level(
    level: Any,
) -> str:
    """
    Return a display label for an evidence level.

    Unknown levels are passed through rather than guessed at.
    """

    value = _text(level).strip().lower()

    labels = {
        "strong": "Strong",
        "high": "Strong",
        "moderate": "Moderate",
        "medium": "Moderate",
        "weak": "Weak",
        "low": "Weak",
        "none": "No sufficient evidence",
        "insufficient": "No sufficient evidence",
        "unknown": "Not available",
        "": "Not available",
    }

    return labels.get(
        value,
        _text(level).title(),
    )


def has_trace(
    response: Optional[Dict[str, Any]],
) -> bool:
    """Return True when a response carries a usable runtime trace."""

    response = _as_dict(response)

    return bool(
        _as_dict(
            response.get("trace")
        )
    )
