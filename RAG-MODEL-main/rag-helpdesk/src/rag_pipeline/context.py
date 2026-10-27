from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.rag_pipeline.config import (
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
)
from src.rag_pipeline.models import Candidate
from src.rag_pipeline.query import get_retrieval_mode, normalize_text


def deduplicate_candidates(
    candidates: list[Candidate],
) -> list[Candidate]:
    """
    Remove duplicate candidates by stable key and normalized text.

    Empty candidates are ignored.
    """
    seen_keys: set[str] = set()
    seen_text: set[str] = set()

    unique: list[Candidate] = []

    for candidate in candidates:
        if not candidate.text or not candidate.text.strip():
            continue

        if candidate.key in seen_keys:
            continue

        normalized_text = normalize_text(candidate.text)

        if not normalized_text:
            continue

        if normalized_text in seen_text:
            continue

        seen_keys.add(candidate.key)
        seen_text.add(normalized_text)

        unique.append(candidate)

    return unique


def _safe_rank(
    value: int | None,
) -> int:
    """
    Missing retrieval ranks are placed after ranked candidates.
    """
    if value is None:
        return 999999

    try:
        return int(value)
    except (TypeError, ValueError):
        return 999999


def _safe_score(
    value: float | None,
    default: float = float("-inf"),
) -> float:
    """
    Safely convert a score to float.
    """
    if value is None:
        return default

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(score):
        return default

    return score


def _candidate_sort_key(
    candidate: Candidate,
    retrieval_mode: str,
    reranked: bool,
) -> tuple[Any, ...]:
    """
    Return a ranking key appropriate for the active retrieval mode.

    Hybrid:
        RRF score

    Vector:
        Dense rank first, then dense distance

    BM25:
        BM25 rank first, then BM25 score

    Reranking:
        Reranker score takes priority when available.
    """

    if reranked:
        return (
            _safe_score(
                candidate.rerank_score,
            ),
        )

    if retrieval_mode == "vector":
        # Dense rank is the primary ordering signal.
        #
        # Lower distance is better, so negate it because the
        # final sort is descending.
        distance = _safe_score(
            candidate.dense_distance,
            default=float("inf"),
        )

        distance_key = (
            -distance
            if math.isfinite(distance)
            else float("-inf")
        )

        return (
            -_safe_rank(candidate.dense_rank),
            distance_key,
        )

    if retrieval_mode == "bm25":
        # BM25 rank is the primary ordering signal.
        # Higher BM25 score is used as a secondary signal.
        return (
            -_safe_rank(candidate.bm25_rank),
            _safe_score(candidate.bm25_score),
        )

    # Hybrid / default.
    return (
        _safe_score(candidate.rrf_score),
    )


def select_context_candidates(
    candidates: list[Candidate],
    reranked: bool,
    diversify_sources: bool = False,
    retrieval_mode: str | None = None,
) -> list[Candidate]:
    """
    Select the candidates that will be passed into the context builder.

    The ranking strategy depends on the actual retrieval mode:

        vector  -> dense rank / distance
        bm25    -> BM25 rank / score
        hybrid  -> RRF score

    This prevents Vector/BM25-only retrieval from incorrectly
    depending on an RRF score that does not exist in those modes.
    """

    if not candidates:
        return []

    # --------------------------------------------------------------
    # DETERMINE RETRIEVAL MODE
    # --------------------------------------------------------------

    mode = (
        str(retrieval_mode).strip().lower()
        if retrieval_mode
        else get_retrieval_mode(candidates)
    )

    mode_aliases = {
        "dense": "vector",
        "vector": "vector",
        "bm25": "bm25",
        "hybrid": "hybrid",
        "both": "hybrid",
    }

    mode = mode_aliases.get(
        mode,
        "hybrid",
    )

    # --------------------------------------------------------------
    # SORT
    # --------------------------------------------------------------

    ranked = sorted(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            mode,
            reranked,
        ),
        reverse=True,
    )

    # --------------------------------------------------------------
    # DEDUPLICATE BEFORE LIMITING
    # --------------------------------------------------------------
    #
    # This allows a lower-ranked candidate to replace a duplicate
    # candidate instead of losing that slot completely.
    # --------------------------------------------------------------

    ranked = deduplicate_candidates(ranked)

    if not ranked:
        return []

    # --------------------------------------------------------------
    # SOURCE DIVERSIFICATION
    # --------------------------------------------------------------

    if diversify_sources:
        selected: list[Candidate] = []
        source_counts: dict[str, int] = {}
        by_source: dict[str, list[Candidate]] = {}
        for candidate in ranked:
            src = (
                candidate.filename
                or candidate.metadata.get("filename")
                or candidate.metadata.get("book_title")
                or candidate.metadata.get("source_name")
                or candidate.metadata.get("source")
                or f"source-{candidate.chunk_id}"
            )
            by_source.setdefault(src, []).append(candidate)

        # Round-robin selection across sources so all books get representation
        max_limit = MAX_CONTEXT_CHUNKS
        while len(selected) < max_limit and any(by_source.values()):
            added_any = False
            for src in list(by_source.keys()):
                if by_source[src] and source_counts.get(src, 0) < MAX_CHUNKS_PER_SOURCE:
                    cand = by_source[src].pop(0)
                    selected.append(cand)
                    source_counts[src] = source_counts.get(src, 0) + 1
                    added_any = True
                    if len(selected) >= max_limit:
                        break
                elif not by_source[src] or source_counts.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
                    by_source.pop(src, None)
            if not added_any:
                break

        ranked = selected

    else:
        # ----------------------------------------------------------
        # NORMAL SOURCE LIMITING
        # ----------------------------------------------------------

        selected = []
        source_counts = {}

        for candidate in ranked:
            source = (
                candidate.filename
                or candidate.metadata.get("filename")
                or candidate.metadata.get("book_title")
                or candidate.metadata.get("source_name")
                or candidate.metadata.get("source")
                or f"source-{candidate.chunk_id}"
            )

            count = source_counts.get(
                source,
                0,
            )

            if count >= MAX_CHUNKS_PER_SOURCE:
                continue

            selected.append(candidate)
            source_counts[source] = count + 1

            if len(selected) >= MAX_CONTEXT_CHUNKS:
                break

        ranked = selected

    # --------------------------------------------------------------
    # CONTEXT CHARACTER LIMIT
    # --------------------------------------------------------------

    final_candidates: list[Candidate] = []
    total_chars = 0

    for candidate in ranked:
        text = (
            candidate.text
            or ""
        ).strip()

        if not text:
            continue

        separator_chars = 2 if final_candidates else 0
        projected_chars = (
            total_chars
            + separator_chars
            + len(text)
        )

        if projected_chars > MAX_CONTEXT_CHARS:
            # If the context is already non-empty, stop here.
            if final_candidates:
                break

            # If even the first chunk is too large, keep a
            # truncated version rather than returning no context.
            available = max(
                0,
                MAX_CONTEXT_CHARS,
            )

            if available <= 0:
                break

            candidate.text = text[:available]
            final_candidates.append(candidate)
            break

        final_candidates.append(candidate)
        total_chars = projected_chars

        if len(final_candidates) >= MAX_CONTEXT_CHUNKS:
            break

    return final_candidates


def build_context(
    candidates: list[Candidate],
) -> tuple[
    str,
    list[str],
    list[dict[str, Any]],
]:
    """
    Build the final LLM context and observability metadata.

    Returns:
        context:
            Text passed to the LLM.

        source_names:
            Unique source filenames.

        retrieved_chunks:
            Structured information about each selected chunk.
    """

    if not candidates:
        return (
            "",
            [],
            [],
        )

    context_parts: list[str] = []
    source_names: list[str] = []
    seen_sources: set[str] = set()
    retrieved_chunks: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        text = (
            candidate.text
            or ""
        ).strip()

        if not text:
            continue

        filename = (
            candidate.filename
            or candidate.metadata.get(
                "filename",
                "",
            )
            or ""
        )

        metadata = candidate.metadata or {}

        file_type = metadata.get("file_type") or (Path(filename).suffix.lstrip(".").lower() if filename else "")
        page_num = metadata.get("page_number") or metadata.get("page_start")
        try:
            page_number = int(page_num) if page_num and int(page_num) > 0 else None
        except (ValueError, TypeError):
            page_number = None

        book_title = (
            metadata.get("book_title")
            or metadata.get("book_name")
            or metadata.get("source_name")
            or metadata.get("source")
            or filename
            or f"Document {index}"
        )
        if book_title.lower().endswith(".pdf"):
            clean_title = book_title[:-4].replace(" (1)", "").strip()
        else:
            clean_title = book_title.replace(" (1)", "").strip()
        clean_title = clean_title or f"Document {index}"

        chapter_num = metadata.get("chapter_num")
        chapter_title = (metadata.get("chapter_title") or "").strip()
        page_start = metadata.get("page_start") or page_number
        page_end = metadata.get("page_end") or page_start

        parts = []
        if clean_title:
            parts.append(clean_title)
        if chapter_num and int(chapter_num) > 0:
            ch_part = f"Chapter {chapter_num}"
            if chapter_title:
                ch_part += f": {chapter_title}"
            parts.append(ch_part)
        if page_start and int(page_start) > 0:
            if page_end and int(page_end) > int(page_start):
                parts.append(f"Pages {page_start}–{page_end}")
            else:
                parts.append(f"Page {page_start}")

        if parts:
            source_display = " — ".join(parts)
        else:
            source_display = filename or f"Document {index}"

        if source_display not in seen_sources:
            seen_sources.add(source_display)
            source_names.append(source_display)

        context_parts.append(
            f"[Source {index}: {source_display}]\n"
            f"{text}"
        )

        page_range_str = ""
        if page_start and int(page_start) > 0:
            if page_end and int(page_end) > int(page_start):
                page_range_str = f"{page_start}–{page_end}"
            else:
                page_range_str = str(page_start)

        retrieved_chunks.append(
            {
                "rank": index,
                "filename": filename,
                "book": clean_title,
                "book_title": clean_title,
                "book_name": metadata.get("book_name") or filename,
                "book_id": metadata.get("book_id", ""),
                "chapter": chapter_num if chapter_num and int(chapter_num) > 0 else None,
                "chapter_num": chapter_num,
                "chapter_title": metadata.get("chapter_title", ""),
                "section": metadata.get("section_heading") or metadata.get("section_title") or "",
                "section_heading": metadata.get("section_heading") or metadata.get("section_title") or "",
                "section_title": metadata.get("section_title") or metadata.get("section_heading") or "",
                "page_start": page_start,
                "page_end": page_end,
                "page_range": page_range_str,
                "token_count": metadata.get("token_count", 0),
                "character_count": len(text),
                "chunk_id": candidate.chunk_id,
                "chunk_index": metadata.get("chunk_index", 0),
                "prev_chunk_id": metadata.get("prev_chunk_id", ""),
                "next_chunk_id": metadata.get("next_chunk_id", ""),
                "text": text,
                "metadata": metadata,
                "file_type": file_type,
                "page_number": page_number if page_number is not None else 0,
                "source_label": source_display,
                "dense_rank": candidate.dense_rank,
                "dense_distance": candidate.dense_distance,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "rrf_score": candidate.rrf_score,
                "rerank_score": candidate.rerank_score,
                "in_both_retrievers": (
                    candidate.in_both_retrievers
                ),
            }
        )

    context = "\n\n".join(
        context_parts
    )

    return (
        context,
        source_names,
        retrieved_chunks,
    )