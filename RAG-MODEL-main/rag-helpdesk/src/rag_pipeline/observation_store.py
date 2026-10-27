"""
Persistent JSONL store for reproducible query observations.

Each line is a self-contained, structured observation record with:
  - raw + normalized query, query_id, timestamp
  - KB/index state + embedding model
  - Dense results: source, chunk_id, rank, score/distance
  - BM25 results: source, chunk_id, rank, score
  - RRF rank/score + retriever membership
  - Selected evidence + final rank
  - Context sent to LLM (sources, char/chunk counts)
  - Answer, citations, claimâ†’evidence mapping
  - Model + latency/token data when available
  - Live evaluation (retrieval + answer quality)

Design invariants:
  - Only what the pipeline recorded is stored; nothing is invented.
  - The raw trace is preserved as-is for full reproducibility.
  - All fields degrade gracefully when a trace sub-key is absent.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_OBSERVATION_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------------

def observation_path(vectorstore_path: Path) -> Path:
    return vectorstore_path / "query_observations.jsonl"


# ---------------------------------------------------------------------------
# STRUCTURED OBSERVATION BUILDER
# ---------------------------------------------------------------------------

def _clean_book_name(item: dict) -> str:
    raw = item.get("book_title") or item.get("book_name") or item.get("book") or item.get("filename") or ""
    stem = Path(raw).stem if str(raw).lower().endswith(".pdf") else str(raw)
    clean = stem.replace(" (1)", "").replace("_", " ").strip()
    return clean or "Unknown"


def _extract_dense_results(trace: dict) -> list[dict]:
    """Minimal reproducible dense result rows."""
    rows = []
    for i, item in enumerate(
        (trace.get("retrieval") or {}).get("raw_dense") or [], start=1
    ):
        if not isinstance(item, dict):
            continue
        rows.append({
            "rank": item.get("dense_rank", item.get("rank", i)),
            "source": item.get("filename", ""),
            "book": _clean_book_name(item),
            "chapter_num": item.get("chapter_num"),
            "chapter_title": item.get("chapter_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "distance": item.get("dense_distance", item.get("distance")),
            "score": item.get("score"),
            "section": item.get("section_heading") or item.get("section_title") or "",
        })
    return rows


def _extract_bm25_results(trace: dict) -> list[dict]:
    """Minimal reproducible BM25 result rows."""
    rows = []
    for i, item in enumerate(
        (trace.get("retrieval") or {}).get("raw_bm25") or [], start=1
    ):
        if not isinstance(item, dict):
            continue
        rows.append({
            "rank": item.get("bm25_rank", item.get("rank", i)),
            "source": item.get("filename", ""),
            "book": _clean_book_name(item),
            "chapter_num": item.get("chapter_num"),
            "chapter_title": item.get("chapter_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "score": item.get("bm25_score", item.get("score")),
            "section": item.get("section_heading") or item.get("section_title") or "",
        })
    return rows


def _extract_rrf_results(trace: dict) -> list[dict]:
    """RRF rank/score + retriever membership for each fused candidate."""
    rows = []
    retrieval = trace.get("retrieval") or {}
    for i, item in enumerate(retrieval.get("rrf_results") or [], start=1):
        if not isinstance(item, dict):
            continue
        rows.append({
            "final_rank": item.get("final_rank", item.get("rank", i)),
            "source": item.get("filename", ""),
            "book": _clean_book_name(item),
            "chapter_num": item.get("chapter_num"),
            "chapter_title": item.get("chapter_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "rrf_score": item.get("rrf_score"),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "dense_contribution": item.get("dense_contribution"),
            "bm25_contribution": item.get("bm25_contribution"),
            "in_dense": item.get("in_dense", item.get("dense_rank") is not None),
            "in_bm25": item.get("in_bm25", item.get("bm25_rank") is not None),
            "retrieval_agreement": item.get("retrieval_agreement", False),
            "section": item.get("section_heading") or item.get("section_title") or "",
        })
    return rows


def _extract_selected_evidence(trace: dict) -> list[dict]:
    """Final selected evidence with stage flags."""
    rows = []
    for item in trace.get("selected_evidence") or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "source": item.get("filename", ""),
            "book": _clean_book_name(item),
            "chapter_num": item.get("chapter_num"),
            "chapter_title": item.get("chapter_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "rrf_score": item.get("rrf_score"),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "in_dense": item.get("in_dense", False),
            "in_bm25": item.get("in_bm25", False),
            "retrieval_agreement": item.get("retrieval_agreement", False),
            "text_preview": (item.get("text") or "")[:200],
            "section": item.get("section_heading") or item.get("section_title") or "",
        })
    return rows


def _extract_context_summary(trace: dict) -> dict:
    """Context sent to LLM â€” sources, char/chunk counts, no raw text."""
    ctx = trace.get("context") or {}
    chunks = ctx.get("final_chunks") or []
    return {
        "chars": ctx.get("chars", ctx.get("context_chars", 0)),
        "chunks": ctx.get("chunks", len(chunks) if isinstance(chunks, list) else 0),
        "sources": ctx.get("sources", []),
    }


def _extract_citation_map(trace: dict) -> list[dict]:
    """Claim â†’ evidence mapping from post-generation grounding."""
    rows = []
    grounding = trace.get("grounding") or {}
    for item in grounding.get("claim_to_citation") or trace.get("citations") or []:
        if not isinstance(item, dict):
            continue
        # Support both grounding chain format and Citation.to_dict() format
        evidence = item.get("evidence") or {}
        rows.append({
            "claim": item.get("claim", ""),
            "citation": item.get("citation", item.get("marker", "")),
            "status": item.get("status", "unknown"),
            "evidence_source": evidence.get("filename", item.get("filename", "")),
            "evidence_chunk_id": evidence.get("chunk_id", item.get("chunk_id", "")),
            "support_score": item.get("support_score"),
            "lexical_overlap": item.get("lexical_overlap"),
            "direct_answer_support": item.get("direct_answer_support", False),
        })
    return rows


def _extract_kb_state(trace: dict) -> dict:
    """KB/index state snapshot embedded in the trace."""
    kb = trace.get("kb_state") or {}
    return {
        "version": kb.get("version"),
        "document_count": kb.get("document_count"),
        "chroma_chunks": kb.get("chroma_chunks"),
        "bm25_chunks": kb.get("bm25_chunks"),
        "indexes_consistent": kb.get("indexes_consistent"),
        "last_updated": kb.get("last_updated", ""),
    }

def _extract_retrieval_breakdown(trace: dict) -> dict:
    """Compute aggregate retrieval statistics by book and chapter.
    
    Returns a dict with:
      - total_books: Number of unique books retrieved
      - total_chapters: Number of unique (book, chapter) pairs
      - breakdown: List of {book, chapter, chapter_title, chunk_count} sorted by book/chapter
    """
    breakdown_dict = {}  # key: (book, chapter) -> {count, title}
    
    # Collect from all retrieval stages
    for item in (trace.get("selected_evidence") or []):
        if not isinstance(item, dict):
            continue
        book = _clean_book_name(item)
        chapter = item.get("chapter_num")
        title = item.get("chapter_title", "")
        
        key = (book, chapter)
        if key not in breakdown_dict:
            breakdown_dict[key] = {"count": 0, "title": title}
        breakdown_dict[key]["count"] += 1
    
    # Build result list
    breakdown = []
    for (book, chapter), data in sorted(breakdown_dict.items()):
        breakdown.append({
            "book": book,
            "chapter": chapter,
            "chapter_title": data["title"],
            "chunk_count": data["count"],
        })
    
    return {
        "total_books": len(set(b for b, c in breakdown_dict.keys())),
        "total_chapters": len(breakdown_dict),
        "breakdown": breakdown,
    }


def _extract_generation_meta(trace: dict) -> dict:
    """Model + token/latency data when available."""
    gen = trace.get("generation") or {}
    return {
        "model": gen.get("model", ""),
        "status": gen.get("status", ""),
        "context_chars": gen.get("context_chars"),
        "response_chars": gen.get("response_chars"),
        "temperature": gen.get("temperature"),
        "message_count": gen.get("message_count"),
    }


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def record_observation(
    vectorstore_path: Path,
    question: str,
    response: dict[str, Any],
    embedding_model: str | None,
    latency_ms: float,
) -> dict[str, Any]:
    """
    Build a complete structured observation and append it to the JSONL store.

    Returns the observation dict so callers (e.g. pipeline.answer_question)
    can attach it back to the trace.
    """
    trace = response.setdefault("trace", {})
    retrieval = trace.get("retrieval") or {}
    query_trace = trace.get("query") or {}

    observation = {
        # â”€â”€ Identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "query_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_query": question,
        "normalized_query": (
            query_trace.get("prepared")
            or query_trace.get("normalized_query")
            or question
        ),
        # â”€â”€ KB / index state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "embedding_model": embedding_model or "unknown",
        "kb_state": _extract_kb_state(trace),
        # â”€â”€ Retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "retrieval_mode": retrieval.get("mode") or retrieval.get("selected_mode") or "hybrid",
        "rrf_k": retrieval.get("rrf_k"),
        "dense_results": _extract_dense_results(trace),
        "bm25_results": _extract_bm25_results(trace),
        "rrf_results": _extract_rrf_results(trace),
        "selected_evidence": _extract_selected_evidence(trace),
        "retrieval_breakdown": _extract_retrieval_breakdown(trace),
        # â”€â”€ Context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "context": _extract_context_summary(trace),
        # â”€â”€ Answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "answer_preview": (response.get("raw_answer") or response.get("answer") or "")[:500],
        "citation_map": _extract_citation_map(trace),
        "citation_coverage": trace.get("citation_coverage") or response.get("citation_coverage") or {},
        # â”€â”€ Generation metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "generation": _extract_generation_meta(trace),
        "latency_ms": round(latency_ms, 1),
        # â”€â”€ Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "evaluation": response.get("evaluation") or {},
        "answer_evaluation": response.get("answer_evaluation") or {},
        "evidence": (trace.get("evidence_gate") or response.get("evidence") or {}),
        "confidence": trace.get("confidence") or response.get("confidence") or {},
        "response_type": response.get("response_type", "unknown"),
        # â”€â”€ Full trace (for drill-down and reproducibility) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "trace": trace,
    }

    # Write observation metadata back into the trace for Observatory display
    trace["observation"] = {
        k: observation[k]
        for k in ("query_id", "timestamp", "latency_ms", "embedding_model")
    }

    path = observation_path(vectorstore_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _OBSERVATION_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, ensure_ascii=False, default=str) + "\n")

    return observation


def update_observation_fields(
    vectorstore_path: Path,
    query_id: str,
    fields: dict[str, Any],
) -> bool:
    """Update fields on a saved observation, preserving its other trace data."""
    path = observation_path(vectorstore_path)
    if not path.exists():
        return False

    with _OBSERVATION_LOCK:
        updated_lines: list[str] = []
        found = False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                updated_lines.append(line)
                continue

            if row.get("query_id") == query_id:
                row.update(fields)
                if "deepeval" in fields:
                    trace = row.setdefault("trace", {})
                    if isinstance(trace, dict):
                        trace["deepeval"] = fields["deepeval"]
                found = True
            updated_lines.append(
                json.dumps(row, ensure_ascii=False, default=str)
            )

        if found:
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                "\n".join(updated_lines) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        return found


def load_observations(
    vectorstore_path: Path,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Load the most recent `limit` observations, newest first.

    Malformed lines are skipped silently.
    """
    path = observation_path(vectorstore_path)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows[-limit:]))


def delete_observation(vectorstore_path: Path, query_id: str) -> bool:
    """Remove a single observation by query_id. Returns True if found."""
    path = observation_path(vectorstore_path)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    found = False
    for line in lines:
        try:
            row = json.loads(line)
            if row.get("query_id") == query_id:
                found = True
                continue
        except json.JSONDecodeError:
            pass
        kept.append(line)
    if found:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return found


