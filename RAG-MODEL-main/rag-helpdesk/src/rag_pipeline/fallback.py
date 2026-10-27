from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from src.document_loader import chunk_documents, load_documents
from src.rag_pipeline.config import (
    ABSTAIN_MESSAGE,
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
)
from src.rag_pipeline.models import Evidence
from src.rag_pipeline.query import candidate_key, normalize_text

logger = logging.getLogger(__name__)


# ======================================================================
# FALLBACK QUERY SUPPORT
# ======================================================================

def _query_tokens(search_text: str) -> set[str]:
    """Extract meaningful query terms for fallback validation."""

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "i", "in", "is", "it", "me", "my", "of",
        "on", "or", "that", "the", "this", "to", "was", "what",
        "when", "where", "which", "who", "why", "with", "you",
        "your", "tell", "give", "explain", "describe", "about",
        "hi", "hello", "hey", "help", "please",
    }

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            search_text.lower(),
        )
        if len(token) > 2
        and token not in stopwords
    }


def _has_fallback_support(
    text: str,
    search_text: str,
) -> bool:
    """
    Require meaningful lexical overlap before allowing fallback
    retrieval to reach the LLM.

    This prevents a generic BM25/keyword match from becoming an
    answer simply because some weak term appeared in a chunk.
    """

    query_tokens = _query_tokens(search_text)

    if not query_tokens:
        return False

    text_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            (text or "").lower(),
        )
    )

    overlap = len(
        query_tokens & text_tokens
    )

    if len(query_tokens) == 1:
        return overlap >= 1

    return (
        overlap / len(query_tokens)
    ) >= 0.5


# ======================================================================
# KEYWORD SEARCH (LAST RESORT)
# ======================================================================

def keyword_search(
    terms: list[str],
    knowledge_base_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Scan raw documents for keywords when index is unavailable."""

    try:
        docs = load_documents(
            knowledge_base_path
        )

        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    except Exception as exc:
        logger.warning(
            "Keyword search failed: %s",
            exc,
        )
        return []

    normalized_terms = []

    for term in terms:
        if not term:
            continue

        cleaned = re.sub(
            r"\s+",
            " ",
            term.lower().strip(),
        )

        if cleaned:
            normalized_terms.append(cleaned)

    if not normalized_terms:
        return []

    scored: list[
        tuple[float, dict[str, Any]]
    ] = []

    # Use the complete search text for support validation.
    search_text = " ".join(
        normalized_terms
    )

    for chunk in chunks:
        text = chunk.get(
            "text",
            "",
        )

        if not text:
            continue

        # Do not allow weak keyword matches into the fallback.
        if not _has_fallback_support(
            text,
            search_text,
        ):
            continue

        text_lower = text.lower()
        score = 0.0

        for term in normalized_terms:
            if term in text_lower:
                score += (
                    1.0
                    + (
                        text_lower.count(term)
                        * 0.1
                    )
                )

        if score > 0.0:
            scored.append(
                (score, chunk)
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results: list[
        dict[str, Any]
    ] = []

    seen_keys: set[str] = set()
    seen_text: set[str] = set()

    for score, document in scored:
        filename = str(
            document.get(
                "filename",
                "unknown",
            )
        )

        chunk_id = document.get(
            "chunk_id",
            0,
        )

        key = candidate_key(
            filename,
            chunk_id,
        )

        norm = normalize_text(
            document.get(
                "text",
                "",
            )
        )

        if (
            key in seen_keys
            or norm in seen_text
        ):
            continue

        seen_keys.add(key)
        seen_text.add(norm)

        document["score"] = score

        results.append(document)

        if len(results) >= top_k:
            break

    return results


# ======================================================================
# ANSWER FROM DOCUMENTS
# ======================================================================

def answer_from_documents(
    question: str,
    documents: list[dict[str, Any]],
    ask_llm_fn: Callable[[str, str], str],
    retrieval_mode: str = "keyword_fallback",
) -> dict[str, Any]:
    """Build context from fallback documents, call LLM, and return response dict."""

    unique_documents = []

    seen_keys: set[str] = set()
    seen_text: set[str] = set()

    for document in documents:
        text = (
            document.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            continue

        filename = str(
            document.get(
                "filename",
                "unknown",
            )
        )

        chunk_id = document.get(
            "chunk_id",
            0,
        )

        key = candidate_key(
            filename,
            chunk_id,
        )

        norm = normalize_text(text)

        if (
            key in seen_keys
            or norm in seen_text
        ):
            continue

        seen_keys.add(key)
        seen_text.add(norm)

        unique_documents.append(
            document
        )

    unique_documents = unique_documents[
        :MAX_CONTEXT_CHUNKS
    ]

    if not unique_documents:
        return {
            "answer": ABSTAIN_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
            "num_retrieved": 0,
            "evidence": Evidence(
                level="none",
                should_answer=False,
                top_score=0.0,
                score_source="fallback",
                supporting_chunks=0,
                top_gap=None,
                agreement=False,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason=(
                    "No usable fallback documents "
                    "were found."
                ),
            ).as_dict(),
        }

    selected_documents = []

    source_counts: dict[str, int] = {}
    current_chars = 0

    for document in unique_documents:
        filename = str(
            document.get(
                "filename",
                "unknown",
            )
        )

        count = source_counts.get(
            filename,
            0,
        )

        if count >= MAX_CHUNKS_PER_SOURCE:
            continue

        text = (
            document.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            continue

        if not selected_documents:
            if len(text) > MAX_CONTEXT_CHARS:
                document = dict(document)

                document["text"] = (
                    text[:MAX_CONTEXT_CHARS]
                )

                text = document["text"]

            selected_documents.append(
                document
            )

            source_counts[filename] = 1
            current_chars = len(text)

            continue

        if (
            current_chars + len(text)
            > MAX_CONTEXT_CHARS
        ):
            continue

        selected_documents.append(
            document
        )

        source_counts[filename] = (
            count + 1
        )

        current_chars += len(text)

    if not selected_documents:
        return {
            "answer": ABSTAIN_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
            "num_retrieved": 0,
            "evidence": Evidence(
                level="none",
                should_answer=False,
                top_score=0.0,
                score_source="fallback",
                supporting_chunks=0,
                top_gap=None,
                agreement=False,
                top_percentile=None,
                retrieval_mode=retrieval_mode,
                reason=(
                    "No fallback context fit "
                    "the context limits."
                ),
            ).as_dict(),
        }

    context = "\n\n---\n\n".join(
        f"Source: {doc.get('filename', 'unknown')}\n"
        f"Section: {doc.get('section_heading', '')}\n"
        f"{doc.get('text', '')}"
        for doc in selected_documents
    )

    sources = list(
        dict.fromkeys(
            str(
                doc.get(
                    "filename",
                    "unknown",
                )
            )
            for doc in selected_documents
        )
    )

    retrieved_chunks = [
        {
            "filename": doc.get(
                "filename",
                "unknown",
            ),
            "chunk_id": doc.get(
                "chunk_id",
                0,
            ),
            "text": doc.get(
                "text",
                "",
            ),
            "score": float(
                doc.get(
                    "score",
                    0.0,
                )
                or 0.0
            ),
            "section_heading": doc.get(
                "section_heading",
                "",
            ),
            "section_path": doc.get(
                "section_path",
                "",
            ),
            "section_level": doc.get(
                "section_level",
                0,
            ),
            "chunk_index": doc.get(
                "chunk_index",
                0,
            ),
        }
        for doc in selected_documents
    ]

    answer = ask_llm_fn(
        question,
        context,
    )

    answer += (
        f"\n\nConfidence: Weak evidence "
        f"({retrieval_mode})"
    )

    top_score = float(
        selected_documents[0].get(
            "score",
            0.0,
        )
        or 0.0
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "num_retrieved": len(
            retrieved_chunks
        ),
        "evidence": Evidence(
            level="weak",
            should_answer=True,
            top_score=top_score,
            score_source="fallback",
            supporting_chunks=len(
                selected_documents
            ),
            top_gap=None,
            agreement=False,
            top_percentile=None,
            retrieval_mode=retrieval_mode,
            reason=(
                "Answer generated using a "
                "fallback lexical retriever."
            ),
        ).as_dict(),
    }


# ======================================================================
# ANSWER VIA KEYWORD FALLBACK
# ======================================================================

def answer_via_keyword_fallback(
    question: str,
    search_text: str,
    bm25: Any,
    knowledge_base_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ask_llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """Execute fallback retrieval using BM25 or raw keyword search."""

    if bm25 is not None:
        try:
            results = bm25.query(
                search_text,
                top_k=MAX_CONTEXT_CHUNKS,
            )

            documents = []

            for result in results:
                text = (
                    result.get(
                        "text",
                        "",
                    )
                    or ""
                ).strip()

                if not text:
                    continue

                # IMPORTANT:
                # BM25 fallback must still have actual lexical
                # support before reaching the LLM.
                if not _has_fallback_support(
                    text,
                    search_text,
                ):
                    continue

                documents.append({
                    "filename": result.get(
                        "filename",
                        "unknown",
                    ),
                    "text": text,
                    "chunk_id": result.get(
                        "chunk_id",
                        0,
                    ),
                    "score": float(
                        result.get(
                            "bm25_score",
                            0.0,
                        )
                        or 0.0
                    ),
                    "section_heading": result.get(
                        "section_heading",
                        "",
                    ),
                    "section_path": result.get(
                        "section_path",
                        "",
                    ),
                    "section_level": result.get(
                        "section_level",
                        0,
                    ),
                    "chunk_index": result.get(
                        "chunk_index",
                        0,
                    ),
                })

            if documents:
                return answer_from_documents(
                    question,
                    documents,
                    ask_llm_fn=ask_llm_fn,
                    retrieval_mode="bm25_fallback",
                )

        except Exception as exc:
            logger.warning(
                "BM25 fallback failed: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # RAW KEYWORD FALLBACK
    # ------------------------------------------------------------------

    keyword_docs = keyword_search(
        [search_text],
        knowledge_base_path=knowledge_base_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not keyword_docs:
        return {
            "answer": ABSTAIN_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
            "num_retrieved": 0,
            "evidence": Evidence(
                level="none",
                should_answer=False,
                top_score=0.0,
                score_source="keyword",
                supporting_chunks=0,
                top_gap=None,
                agreement=False,
                top_percentile=None,
                retrieval_mode="keyword_fallback",
                reason=(
                    "No relevant information was found "
                    "in the knowledge base."
                ),
            ).as_dict(),
        }

    return answer_from_documents(
        question,
        keyword_docs,
        ask_llm_fn=ask_llm_fn,
        retrieval_mode="keyword_fallback",
    )
