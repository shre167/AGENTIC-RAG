from __future__ import annotations
from typing import Any

from src.rag_pipeline.config import (
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_text(chunk: Any) -> str:
    """Get text from either a Candidate object or a dictionary."""
    if hasattr(chunk, "text"):
        return str(getattr(chunk, "text", "") or "")

    if isinstance(chunk, dict):
        return str(chunk.get("text", "") or "")

    return str(chunk or "")


def _get_source(chunk: Any) -> str:
    """Get source filename from either a Candidate object or a dictionary."""
    if hasattr(chunk, "filename"):
        return str(getattr(chunk, "filename", "") or "unknown")

    if isinstance(chunk, dict):
        metadata = chunk.get("metadata", {}) or {}

        return str(
            chunk.get("filename")
            or chunk.get("source")
            or metadata.get("source")
            or "unknown"
        )

    return "unknown"


def optimize_context(
    chunks: list[Any],
    search_text: str = "",
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> dict[str, Any]:
    """
    Reduce retrieved candidates to a manageable context while preserving
    the original Candidate objects.

    IMPORTANT:
    Defaults now come from src.rag_pipeline.config, the same module used
    by context.py and fallback.py. Previously this function defined its
    own hardcoded defaults (18000 chars / 5 chunks / 3 per source), and
    pipeline.py's call site never overrode them, so this second-pass
    optimizer silently applied different, smaller limits than the ones
    already enforced earlier in the pipeline by select_context_candidates().
    That mismatch capped every answer at 5 chunks regardless of the
    configured MAX_CONTEXT_CHUNKS value.

    Compatible with the existing pipeline.py contract.
    """

    max_context_chars = _safe_int(max_context_chars, MAX_CONTEXT_CHARS)
    max_chunks = _safe_int(max_chunks, MAX_CONTEXT_CHUNKS)
    max_chunks_per_source = _safe_int(max_chunks_per_source, MAX_CHUNKS_PER_SOURCE)

    if max_context_chars <= 0:
        max_context_chars = MAX_CONTEXT_CHARS

    if max_chunks <= 0:
        max_chunks = MAX_CONTEXT_CHUNKS

    if max_chunks_per_source <= 0:
        max_chunks_per_source = MAX_CHUNKS_PER_SOURCE

    search_text = str(search_text or "").strip()

    input_chunks = len(chunks)

    input_chars = sum(
        len(_get_text(chunk))
        for chunk in chunks
    )

    selected: list[Any] = []
    removed: list[Any] = []

    steps: list[dict[str, Any]] = []

    source_counts: dict[str, int] = {}

    current_chars = 0

    for chunk in chunks:

        source = _get_source(chunk)
        text = _get_text(chunk)

        # Maximum number of chunks.
        if len(selected) >= max_chunks:
            removed.append(chunk)

            steps.append(
                {
                    "action": "removed",
                    "reason": "max_chunks",
                    "source": source,
                }
            )

            continue

        # Maximum chunks from one source.
        if source_counts.get(source, 0) >= max_chunks_per_source:
            removed.append(chunk)

            steps.append(
                {
                    "action": "removed",
                    "reason": "max_chunks_per_source",
                    "source": source,
                }
            )

            continue

        chunk_chars = len(text)
        new_chars = current_chars + chunk_chars

        # Context character limit.
        if new_chars > max_context_chars:

            # Keep the first candidate even if it is individually
            # larger than the context limit.
            if not selected:
                selected.append(chunk)

                source_counts[source] = (
                    source_counts.get(source, 0) + 1
                )

                current_chars += chunk_chars

                steps.append(
                    {
                        "action": "kept",
                        "reason": "first_candidate",
                        "source": source,
                        "chars": chunk_chars,
                    }
                )

            else:
                removed.append(chunk)

                steps.append(
                    {
                        "action": "removed",
                        "reason": "max_context_chars",
                        "source": source,
                        "chars": chunk_chars,
                    }
                )

            continue

        selected.append(chunk)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        current_chars = new_chars

        steps.append(
            {
                "action": "kept",
                "reason": "within_limits",
                "source": source,
                "chars": chunk_chars,
            }
        )

    output_chunks = len(selected)

    output_chars = sum(
        len(_get_text(chunk))
        for chunk in selected
    )

    removed_chunks = len(removed)

    stats = {
        "input_chunks": input_chunks,
        "output_chunks": output_chunks,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "removed_chunks": removed_chunks,
    }

    return {
        "selected": selected,
        "removed": removed,
        "input_chunks": input_chunks,
        "output_chunks": output_chunks,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "stats": stats,
        "search_text": search_text,
        "steps": steps,
    }