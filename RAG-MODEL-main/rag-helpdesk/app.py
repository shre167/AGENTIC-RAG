from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
from src.document_loader import (
    chunk_documents,
    get_chunking_config,
    list_txt_files,
    list_knowledge_files,
    load_documents,
)
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline.kb_state import get_current_kb_state
from src.rag_pipeline.observation_store import load_observations
from src.rag_pipeline.observatory import (
    build_observatory_snapshot,
    describe_evidence_level,
    export_trace_json,
    get_candidate_chunks,
    get_context_chunks,
    get_supporting_sources,
    has_trace,
    retrieval_mode_label,
)
from src.ui.astronomy_theme import (
    inject_global_css,
    render_arch_arrow,
    render_arch_legend,
    render_arch_node,
    render_arch_plane_open,
    render_architecture_node,
    render_chunk_map,
    render_citation_card,
    render_citation_coverage_badge,
    render_confidence_card,
    render_control_label,
    render_evaluation_criterion_card,
    render_evidence_badge,
    render_formula_card,
    render_html,
    render_kb_state_card,
    render_kv_rows,
    render_mode_chip,
    render_mode_description,
    render_nav_group_label,
    render_page_header,
    render_panel_card,
    render_pipeline_node,
    render_retrieval_health_card,
    render_section_title,
    render_sidebar_divider,
    render_simulation_tag,
    render_sources_card,
    render_stage_rail,
    render_stat_grid,
    render_status_card,
    render_status_row,
    render_trace_query_card,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# These keys are the exact strings handed to the backend. They are not
# display strings and must not be renamed.
MODE_ORDER = ["vector", "bm25", "hybrid"]

MODES = {
    "vector": {
        "segment": "VECTOR",
        "short": "Vector",
        "caption": (
            "Dense semantic retrieval using the Chroma vector index."
        ),
    },
    "bm25": {
        "segment": "BM25",
        "short": "BM25",
        "caption": (
            "Lexical retrieval using the persistent BM25 index."
        ),
    },
    "hybrid": {
        "segment": "HYBRID",
        "short": "Hybrid",
        "caption": (
            "Dense + lexical retrieval fused using Reciprocal Rank Fusion."
        ),
    },
}

SEGMENT_TO_MODE = {MODES[key]["segment"]: key for key in MODE_ORDER}

PAGE_MISSION = "Mission Control"
PAGE_CHAT_HISTORY = "Chat History"
PAGE_LAB = "Retrieval Lab"
PAGE_OBSERVATION = "Observation & Evaluation"
PAGE_DASHBOARD = "Evaluation Dashboard"
PAGE_COMPARISON = "Query Comparison"
PAGE_INGEST = "Ingestion Pipeline"
PAGE_CHUNKS = "Database"
PAGE_ARCH = "System Architecture"

# (group label, [(page, icon)])
NAV_GROUPS = [
    ("Database & History", [
        (PAGE_CHAT_HISTORY, ""),
        (PAGE_CHUNKS, ""),
    ]),
    ("Mission Control", [
        (PAGE_MISSION, ""),
    ]),
    ("Retrieval", [(PAGE_LAB, "")]),
    ("Knowledge Base", [(PAGE_INGEST, "")]),
    ("Evaluation", [
        (PAGE_OBSERVATION, "Observation"),
        (PAGE_DASHBOARD, "Dashboard"),
        (PAGE_COMPARISON, "Compare"),
    ]),
    ("System", [(PAGE_ARCH, "")]),
]

PAGES = [page for _, items in NAV_GROUPS for page, _ in items]


def _sidebar_nav_label(marker: str, page_name: str, icon: str) -> str:
    """Build sidebar button text; skip text icons that duplicate the page name."""
    if icon and not icon.isascii():
        return f"{marker}  {icon}  {page_name}"
    return f"{marker}  {page_name}"


INGESTION_STAGES = [
    ("", "Document", "Loaded"),
    ("", "Text Cleaning", "Normalized"),
    ("", "Chunking", "Segmented"),
    ("", "Embedding", "Vectorized"),
    ("", "Chroma", "Stored"),
    ("", "BM25", "Indexed"),
    ("", "Knowledge Ready", "Queryable"),
]

SIMULATION_STEPS = [
    ("01", "", "DOCUMENT DETECTED",
     "A TXT or Markdown file is discovered in the knowledge base "
     "directory and queued for processing."),
    ("02", "", "TEXT EXTRACTED",
     "Raw text is read and normalized. Whitespace is collapsed and "
     "section headings are detected so they can be kept as metadata."),
    ("03", "", "CONTENT CHUNKED",
     "Paragraphs are merged into token-sized chunks (target ~350, max ~450). "
     "Oversized paragraphs are split with a small token carry-over."),
    ("04", "", "EMBEDDING GENERATED",
     "Each chunk is converted into a numerical vector by the configured "
     "embedding model, capturing meaning rather than exact wording."),
    ("05", "", "STORED IN CHROMA",
     "Vectors and metadata are written to the persistent Chroma "
     "collection, which survives restarts."),
    ("06", "", "INDEXED IN BM25",
     "The same chunks are added to the persistent BM25 index for exact "
     "term matching, keeping rare names and identifiers findable."),
]

# Component descriptions for the architecture page. These describe the
# system as actually implemented - nothing here is aspirational.
ARCH_DESCRIPTIONS = [
    ("Document Loader",
     "Reads TXT and Markdown files from the knowledge base directory and "
     "detects their structure."),
    ("Text Processing",
     "Normalizes raw text and records section headings as metadata before "
     "the document is split."),
    ("Chunking",
     "Packs paragraphs into token-sized chunks (~350 target, ~450 max) for "
     "both semantic and lexical retrieval."),
    ("Embedding Model",
     "Turns each chunk into a vector using the model named in the "
     "environment configuration."),
    ("Chroma Vector Store",
     "Persistent vector store used for dense semantic retrieval."),
    ("BM25 Index",
     "Lexical retrieval layer used for keyword and token-level matching. "
     "It requires no embedding call."),
    ("Query Preparation",
     "Normalizes the incoming question, expands aliases and produces the "
     "token list BM25 searches with."),
    ("Dense Search",
     "Embeds the prepared query and finds the nearest chunk vectors in "
     "Chroma."),
    ("BM25 Search",
     "Scores chunks by term overlap with the prepared query."),
    ("RRF Fusion",
     "Combines ranked dense and BM25 results using Reciprocal Rank "
     "Fusion. Used in Hybrid mode only. It is a ranking signal, not a "
     "confidence or probability."),
    ("Evidence Gate",
     "Evaluates whether the retrieved information is sufficient to "
     "answer, and decides whether the system should answer at all."),
    ("Context Selection",
     "Selects which of the surviving chunks are passed into generation, "
     "subject to chunk and character budgets."),
    ("Generation",
     "Uses the configured LLM to produce the final grounded response."),
    ("Answer",
     "The response returned to the user, together with its supporting "
     "sources and the runtime trace the Retrieval Lab reads."),
    ("Observation & Trace",
     "Captures full candidate journeys, index statistics, and retrieval states."),
    ("Retrieval Health Evaluation",
     "Evaluates 7 relative retrieval criteria at runtime, prioritizing direct "
     "answer support and relevance over agreement."),
    ("Claim Grounding & Citations",
     "Verifies factual claims against retrieved passages, mapping bracketed "
     "citations to chunk sources and flagging unsupported claims."),
    ("Evidence Confidence",
     "Computes evidence-backed confidence (HIGH/MEDIUM/LOW) with explainable reasons."),
    ("Knowledge Base State",
     "Tracks versioning, document counts, and Chroma/BM25 index synchronization."),
]

RETRIEVAL_DIAGRAM = """\
                    USER QUERY
                        
                        
                Query Preparation
                        
             
                                  
       Dense Retrieval           BM25
                                  
             
                        
                       RRF
                        
                        
                Evidence Evaluation
                        
                        
                 Context Selection
                        
                        
                  LLM Generation
                        
                        
                      Answer"""

INGESTION_DIAGRAM = """\
Documents
    
    
Cleaning
    
    
Chunking
    
    
                   
Embeddings        BM25
    
    
Chroma"""


# ============================================================================
# STREAMLIT COMPATIBILITY
# ============================================================================

def _layout_kwargs() -> dict:
    """
    Full-width keyword arguments for the installed Streamlit version.

    Streamlit 1.49 replaced use_container_width=True with width="stretch"
    and deprecated the old name. Resolving this at runtime keeps the same
    source file working across both machines without edits.
    """
    try:
        parts = st.__version__.split(".")
        version = (int(parts[0]), int(parts[1]))
    except Exception:  # noqa: BLE001
        return {"use_container_width": True}

    if version >= (1, 49):
        return {"width": "stretch"}

    return {"use_container_width": True}


WIDE = _layout_kwargs()


def _dataframe(rows) -> None:
    """Render a table full width, without version-specific warnings."""
    st.dataframe(rows, hide_index=True, **WIDE)


def _extract_metadata(row: dict) -> dict:
    """Extract book, chapter, section metadata from a retrieval result row."""
    # Book
    book = row.get("book_title") or row.get("book") or ""
    if not book and row.get("filename"):
        book = Path(row.get("filename")).stem
    book = book.replace(" (1)", "").replace("_", " ").strip() or "Unknown"
    
    # Chapter
    chapter = row.get("chapter")
    if not chapter:
        ch_num = row.get("chapter_num")
        if ch_num is not None:
            try:
                ch_int = int(ch_num)
                chapter = f"Ch. {ch_num}" if ch_int > 0 else ("Intro" if ch_int == 0 else "-")
            except (ValueError, TypeError):
                chapter = "-"
        else:
            chapter = "-"
    
    # Section
    section = row.get("section_heading") or ""
    
    return {
        "book": book,
        "chapter": chapter,
        "section": section,
    }


def _segmented_control(options: list, default: str, key: str) -> str:
    """
    Render a real segmented control.

    Uses st.segmented_control where available (Streamlit 1.40+), then
    st.pills, and finally falls back to a horizontal radio which the
    stylesheet strips of its radio circles. In every case the returned
    value is one of `options`.
    """
    if hasattr(st, "segmented_control"):
        try:
            value = st.segmented_control(
                "Retrieval engine",
                options,
                default=default,
                key=key,
                label_visibility="collapsed",
                **WIDE,
            )
        except TypeError:
            # Older Streamlit accepts segmented_control but not the
            # width/use_container_width kwarg  retry without it rather
            # than losing the control entirely.
            try:
                value = st.segmented_control(
                    "Retrieval engine",
                    options,
                    default=default,
                    key=key,
                    label_visibility="collapsed",
                )
            except TypeError:
                value = None

        return value if value in options else default

    if hasattr(st, "pills"):
        try:
            value = st.pills(
                "Retrieval engine",
                options,
                default=default,
                key=key,
                label_visibility="collapsed",
            )
            return value if value in options else default
        except TypeError:
            pass

    value = st.radio(
        "Retrieval engine",
        options,
        index=options.index(default),
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )

    return value if value in options else default


# ============================================================================
# BACKEND ACCESS HELPERS
# ============================================================================

def _pipeline() -> RAGPipeline:
    """Return (and cache) a RAGPipeline in session_state."""
    import inspect
    cached_pipe = st.session_state.get("pipeline")
    # Verify cached pipeline has on_event support in answer_question
    is_outdated = False
    if cached_pipe is not None:
        try:
            sig = inspect.signature(cached_pipe.answer_question)
            if "on_event" not in sig.parameters:
                is_outdated = True
        except Exception:
            is_outdated = True

    if (
        "pipeline" not in st.session_state
        or st.session_state.get("force_reload", False)
        or is_outdated
    ):
        st.session_state.pipeline = RAGPipeline(
            knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
            vectorstore_path=Path(VECTORSTORE_DIR),
        )
        st.session_state.force_reload = False
        # Reset KB snapshot so stale-chunk check runs on next status call
        st.session_state.pop("_kb_snapshot", None)

    return st.session_state.pipeline


def _kb_file_snapshot(kb_path: Path) -> frozenset:
    """Return a frozenset of (name, mtime) pairs for all KB documents."""
    try:
        return frozenset(
            (p.name, p.stat().st_mtime)
            for p in kb_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}
        )
    except Exception:
        return frozenset()


def _pipeline_status(pipeline: RAGPipeline) -> dict:
    """Return cached pipeline status, refreshing at most every 5 seconds.

    Also detects file deletions from the knowledge_base/ directory and
    purges stale chunks automatically — without blocking navigation.
    """
    import time as _time

    now = _time.monotonic()
    cache_age = now - st.session_state.get("_status_ts", 0.0)

    # ── Stale-chunk detection (fast: only stat calls, no Chroma I/O) ──
    kb_path = Path(pipeline.knowledge_base_path)
    current_snapshot = _kb_file_snapshot(kb_path)
    last_snapshot = st.session_state.get("_kb_snapshot")

    if last_snapshot is not None and current_snapshot != last_snapshot:
        # Files were added or removed — purge stale chunks immediately
        try:
            sync_result = pipeline.sync_deleted_files()
            removed = sync_result.get("removed", [])
            if removed:
                names = ", ".join(r["filename"] for r in removed)
                st.session_state.ingestion_notice = {
                    "type": "info",
                    "message": f"Auto-removed stale chunks for: {names}",
                }
                # Force status refresh so sidebar counts update
                cache_age = 999.0
        except Exception:
            pass

    st.session_state["_kb_snapshot"] = current_snapshot

    # ── Return cached status when it is fresh enough ──────────────────
    if cache_age < 5.0 and "pipeline_status" in st.session_state:
        return st.session_state["pipeline_status"]

    try:
        status = pipeline.status()
    except Exception as exc:  # noqa: BLE001
        st.session_state.status_error = str(exc)
        return st.session_state.get("pipeline_status", {})

    st.session_state.pop("status_error", None)
    st.session_state["pipeline_status"] = status or {}
    st.session_state["_status_ts"] = now
    return st.session_state["pipeline_status"]


def _collection_count(pipeline: RAGPipeline):
    """Return cached Chroma collection count (refreshed alongside status)."""
    # Prefer the count embedded in the cached status dict
    cached = st.session_state.get("pipeline_status", {})
    if "collection_count" in cached:
        return cached["collection_count"]
    try:
        return int(pipeline.vector_store.get_collection_count())
    except Exception:  # noqa: BLE001
        return None


def _bm25_indexed_count(status: dict):
    """
    Number of chunks in the BM25 index, if the backend reports it.

    Returns None rather than a guess when status exposes no value.
    """
    for key in (
        "bm25_chunks",
        "bm25_documents",
        "bm25_count",
        "bm25_indexed_chunks",
        "bm25_size",
    ):
        value = status.get(key)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None


def _llm_model(status: dict):
    """
    Configured LLM model name.

    Read from pipeline status first, then from the environment, using the
    existing variable names. Never hardcoded.
    """
    for key in ("llm_model", "generation_model", "model"):
        value = status.get(key)

        if value:
            return str(value)

    for env_name in ("LLM_MODEL", "GENERATION_MODEL"):
        value = os.getenv(env_name)

        if value:
            return value

    return None


def _chunk_config(pipeline: RAGPipeline) -> tuple[int | None, int | None]:
    """Target and max token limits used by the canonical chunker."""
    _ = pipeline
    cfg = get_chunking_config()
    return cfg.get("target_tokens"), cfg.get("max_tokens")


def _kb_signature(kb_path: Path) -> tuple:
    """
    Fingerprint of the knowledge base directory.

    Used as a cache key so chunk previews refresh when files change.
    """
    signature = []

    try:
        for path in sorted(list_knowledge_files(kb_path)):
            stat = path.stat()
            signature.append((path.name, stat.st_size, int(stat.st_mtime)))
    except Exception:  # noqa: BLE001
        return tuple()

    return tuple(signature)


@st.cache_data(show_spinner=False)
def _compute_chunks(
    kb_path: str,
    chunk_size: int,
    chunk_overlap: int,
    signature: tuple,
) -> list:
    """
    Load canonical chunks for inspection.
    Uses precomputed canonical chunks cache if available for instant UI rendering.
    """
    cache_path = Path(VECTORSTORE_DIR) / "canonical_chunks_cache.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached and isinstance(cached, list):
                for c in cached:
                    if not c.get("filename"):
                        c["filename"] = c.get("book_name") or "untitled_document.pdf"
                return [c for c in cached if (c.get("text") or "").strip()]
        except Exception:
            pass

    documents = load_documents(Path(kb_path))

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for c in chunks:
        if not c.get("filename"):
            c["filename"] = c.get("book_name") or "untitled_document.pdf"

    return [c for c in chunks if (c.get("text") or "").strip()]


def _load_chunks(pipeline: RAGPipeline) -> list:
    """Load all chunks for inspection, prioritizing the live ingested database (Chroma)."""
    # 1. Prioritize live Chroma vector store so all ingested books and chapters appear
    if pipeline and hasattr(pipeline, "vector_store"):
        try:
            cnt = pipeline.vector_store.get_collection_count()
            if cnt > 0:
                data = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
                if data and data.get("ids"):
                    chroma_chunks = []
                    ids = data["ids"]
                    docs = data.get("documents") or []
                    metas = data.get("metadatas") or []
                    for i, cid in enumerate(ids):
                        m = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                        doc = docs[i] if i < len(docs) and docs[i] else ""
                        fn = m.get("filename") or m.get("book_name") or ""
                        bt = m.get("book_title") or m.get("book") or (Path(fn).stem if fn else "")
                        chroma_chunks.append({
                            **m,
                            "chunk_id": cid,
                            "text": doc,
                            "filename": fn,
                            "book_title": bt,
                            "book": bt,
                        })
                    if chroma_chunks:
                        return chroma_chunks
        except Exception:
            pass

    # 2. Fallback to cache or on-demand chunking
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    size, overlap = _chunk_config(pipeline)

    if size is None or overlap is None:
        return []

    try:
        return _compute_chunks(
            str(kb_path),
            int(size),
            int(overlap),
            _kb_signature(kb_path),
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Chunk preview unavailable: {exc}")
        return []


def _document_characters(path: Path):
    """Character count of a knowledge base file."""
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None


def _na(value):
    """Map empty values to None so the UI renders a muted N/A."""
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    return value


def _service_error(title: str, detail: str = "") -> None:
    """
    Show a professional error panel instead of a raw stack trace.

    The underlying exception text is kept behind an expander so it is
    available for debugging without being pushed at a normal user.
    """
    st.error(
        f" {title}\n\n"
        "The configured retrieval, embedding or generation service could "
        "not be reached. Check the environment and API configuration."
    )

    if detail:
        with st.expander("Technical detail"):
            st.code(str(detail), language="text")


# ============================================================================
# RESPONSE / TRACE HELPERS
# ============================================================================

def _build_meta(response: dict, requested_mode: str) -> dict:
    """
    Build display metadata for one answer.

    Everything here is read from the runtime trace. Nothing is synthesized.
    """
    trace = response.get("trace") or {}

    snapshot = build_observatory_snapshot(trace) if trace else {}

    evidence = snapshot.get("evidence", {}) if snapshot else {}
    retrieval = snapshot.get("retrieval", {}) if snapshot else {}

    raw_dense = retrieval.get("raw_dense") or []
    raw_bm25 = retrieval.get("raw_bm25") or []
    rrf_res = retrieval.get("rrf_results") or retrieval.get("rrf_calculation") or []
    ctx_chunks = get_context_chunks(trace)
    all_cand = get_candidate_chunks(trace, response)

    return {
        "requested_mode": requested_mode,
        "mode_label": retrieval.get(
            "mode_label",
            retrieval_mode_label(requested_mode),
        ),
        "evidence_level": evidence.get("level"),
        "evidence_label": describe_evidence_level(evidence.get("level")),
        "supporting_sources": get_supporting_sources(trace, response),
        "context_chunks": ctx_chunks,
        "candidates": all_cand,
        "candidate_count": retrieval.get("candidate_count") or len(all_cand),
        "dense_candidate_count": len(raw_dense),
        "bm25_candidate_count": len(raw_bm25),
        "rrf_candidate_count": retrieval.get("fused_count") or len(rrf_res) or len(all_cand),
        "final_context_chunk_count": len(ctx_chunks),
        "retrieval_time_ms": retrieval.get("retrieval_ms") or (response.get("observation", {}).get("latency_ms") if response else None),
        "trace_available": bool(trace),
        "error": response.get("error", ""),
        "evaluation": response.get("evaluation") or snapshot.get("evaluation", {}),
        "citations": response.get("citations") or snapshot.get("citations", []),
        "citation_coverage": response.get("citation_coverage") or snapshot.get("citation_coverage", {}),
        "confidence": response.get("confidence") or snapshot.get("confidence", {}),
        "kb_state": response.get("kb_state") or snapshot.get("kb_state", {}),
        "candidate_journey": trace.get("candidate_journey") or snapshot.get("candidate_journey", []),
        "llm_ms": trace.get("llm_ms") or response.get("llm_ms"),
        "retrieval_ms": trace.get("retrieval_ms") or response.get("retrieval_ms"),
        "deepeval_results": response.get("deepeval_results") or response.get("deepeval") or {},
        "langfuse_trace_id": (
            response.get("langfuse_trace_id")
            or (response.get("observability") or {}).get("langfuse_trace_id")
            or (response.get("observation", {}) or {}).get("langfuse_trace_id")
        ),
        "response_type": response.get("response_type"),
        "agent_activity": response.get("agent_activity") or (response.get("trace") or {}).get("agent_activity") or {},
        "events": response.get("events", []),
    }


def _render_agent_activity(agent_activity: dict) -> None:
    """Render the Agent Activity panel with controller decision details."""
    if not agent_activity:
        return

    with st.expander("🤖 Agent Activity", expanded=False):
        # Question type and plan
        col1, col2 = st.columns(2)
        col1.write(f"**Question Type:** {agent_activity.get('question_type', 'unknown')}")
        col2.write(f"**Retrieval Plan:** {agent_activity.get('retrieval_plan', 'N/A')}")

        # Final decision
        final_decision = agent_activity.get("final_decision", "unknown")
        decision_emoji = "✅" if final_decision == "answer" else "⏭"
        st.write(f"**Final Decision:** {decision_emoji} {final_decision.upper()}")

        # Attempts
        attempts = agent_activity.get("attempts", [])
        if attempts:
            st.markdown("#### Retrieval Attempts")
            for attempt in attempts:
                attempt_num = attempt.get("attempt", 1)
                mode = attempt.get("retrieval_mode", "unknown")
                sufficient = attempt.get("evidence_sufficient", False)
                level = attempt.get("evidence_level", "none")
                reason = attempt.get("evidence_reason", "")
                changes = attempt.get("changes_from_previous", "")

                with st.expander(
                    f"Attempt {attempt_num} — {mode} — {level.upper()} "
                    f"({'✓ Sufficient' if sufficient else '✗ Insufficient'})",
                    expanded=(attempt_num == 1),
                ):
                    # Tools called
                    tools = attempt.get("tools_called", [])
                    if tools:
                        st.caption(f"**Tools:** {', '.join(tools)}")

                    # Search text
                    search_text = attempt.get("search_text", "")
                    if search_text:
                        st.caption(f"**Search:** {search_text}")

                    # Retrieval metrics
                    ret_ms = attempt.get("retrieval_ms", 0)
                    candidates = attempt.get("candidate_count", 0)
                    fused = attempt.get("fused_count", 0)
                    st.caption(
                        f"**Latency:** {ret_ms:.0f}ms | "
                        f"**Candidates:** {candidates} | **Fused:** {fused}"
                    )

                    # Evidence assessment
                    if reason:
                        st.caption(f"**Evidence:** {reason}")

                    # Retry information
                    if attempt_num > 1:
                        if changes:
                            st.caption(f"**Changed:** {changes}")
                        retry_reason = attempt.get("retry_reason", "")
                        if retry_reason:
                            st.caption(f"**Retry Reason:** {retry_reason}")

        # Timings and assessment
        st.markdown("#### Assessment")
        timings = agent_activity.get("timings_ms", {})
        col1, col2 = st.columns(2)
        col1.metric(
            "Controller Total",
            f"{timings.get('controller_total', 0):.0f} ms",
        )
        col2.metric(
            "Last Retrieval",
            f"{timings.get('last_retrieval', 0):.0f} ms",
        )


def _render_chunk_list(chunks: list, mode_label: str = "") -> None:
    """Render chunks exposing complete canonical metadata and clean formatting."""
    for index, chunk in enumerate(chunks, start=1):
        book = (
            chunk.get("book")
            or chunk.get("book_title")
            or chunk.get("book_name")
            or chunk.get("filename")
            or chunk.get("source")
            or f"Document {index}"
        )
        chunk_id = chunk.get("chunk_id") or f"chunk_{index}"
        ch = chunk.get("chapter") or chunk.get("chapter_num")
        ch_str = f"Chapter {ch}" if ch and int(ch) > 0 else ""
        sec = chunk.get("section") or chunk.get("section_heading") or chunk.get("section_title") or ""
        p_range = chunk.get("page_range") or (f"pp. {chunk.get('page_start')}–{chunk.get('page_end')}" if chunk.get('page_start') else "")

        title_parts = [f"**{index}. {book}**"]
        if ch_str:
            title_parts.append(f"*{ch_str}*")
        if p_range and p_range != "-":
            title_parts.append(f"`{p_range}`")
        if mode_label:
            title_parts.append(f"[{mode_label}]")

        st.markdown(" — ".join(title_parts))

        # Canonical Metadata Badges
        badges = [f"**Chunk ID:** `{chunk_id}`"]
        if sec:
            badges.append(f"**Section:** {sec}")
        if chunk.get("dense_rank") is not None:
            dist_str = f" (dist: {chunk.get('dense_distance')})" if chunk.get("dense_distance") is not None else ""
            badges.append(f"**Dense:** #{chunk['dense_rank']}{dist_str}")
        if chunk.get("bm25_rank") is not None:
            score_str = f" (score: {chunk.get('bm25_score')})" if chunk.get("bm25_score") is not None else ""
            badges.append(f"**BM25:** #{chunk['bm25_rank']}{score_str}")
        if chunk.get("rrf_score") is not None and float(chunk.get("rrf_score") or 0) > 0:
            badges.append(f"**RRF Score:** {chunk['rrf_score']}")
        elif chunk.get("score") is not None:
            badges.append(f"**Score:** {chunk['score']}")

        st.caption(" | ".join(badges))

        text = chunk.get("text") or ""
        if text:
            with st.expander(f"Inspect Text — {len(text)} characters", expanded=(index == 1)):
                # Rich metadata panel
                meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
                ch = chunk.get("chapter") or chunk.get("chapter_num")
                ch_title = chunk.get("chapter_title") or ""
                sec = chunk.get("section") or chunk.get("section_heading") or chunk.get("section_title") or ""
                p_start = chunk.get("page_start")
                p_end   = chunk.get("page_end") or p_start
                tok     = chunk.get("token_count")
                cid     = chunk.get("chunk_id") or ""

                meta_c1.caption(f"**Book:** {chunk.get('book') or chunk.get('book_title') or '—'}")
                meta_c2.caption(f"**Chapter:** {f'Ch. {ch}' if ch else '—'}{f' — {ch_title}' if ch_title else ''}")
                meta_c3.caption(f"**Section:** {sec[:40] if sec else '—'}")
                meta_c4.caption(f"**Pages:** {f'{p_start}–{p_end}' if p_start else '—'} | **Tokens:** {tok or '—'}")

                st.code(text, language="text")

                with st.expander("Raw metadata", expanded=False):
                    st.json({k: v for k, v in chunk.items() if k != "text" and v is not None and v != "" and v != 0})
        else:
            st.caption("Chunk text is not present in the trace.")
        st.markdown("---")


def _render_agent_activity(agent_activity: dict) -> None:
    """Render agent controller reasoning, attempts, and decisions."""
    if not agent_activity or not agent_activity.get("question_type"):
        return

    q_type = agent_activity.get("question_type", "general")
    decision = agent_activity.get("final_decision", "answer")
    decision_display = "ANSWER \u2705" if decision == "answer" else "ABSTAIN \u274c"
    attempt_count = agent_activity.get("attempt_count", len(agent_activity.get("attempts", [])))
    mode = agent_activity.get("retrieval_mode", "")
    plan = agent_activity.get("retrieval_plan", "")

    with st.expander("\U0001f916 Agent Activity", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Question Type", q_type)
        c2.metric("Final Decision", decision_display)
        c3.metric("Attempts", attempt_count)
        c4.metric("Active Mode", mode or "\u2014")

        if plan:
            st.caption(f"**Retrieval Plan:** `{plan}`")

        attempts = agent_activity.get("attempts", [])
        for att in attempts:
            att_num = att.get("attempt", 1)
            att_mode = att.get("retrieval_mode", mode)
            att_level = att.get("evidence_level", "none")
            att_suff = "\u2705 Sufficient" if att.get("evidence_sufficient") else "\u274c Insufficient"

            with st.expander(f"Attempt {att_num} ({att_mode}) \u2014 {att_suff}", expanded=(len(attempts) == 1)):
                tools_called = att.get("tools_called") or []
                st.caption(f"**Mode:** `{att_mode}` | **Tools:** {', '.join(tools_called) or '\u2014'}")
                st.markdown(f"**Search Text:** `{att.get('search_text', '')}`")

                sc1, sc2, sc3 = st.columns(3)
                sc1.caption(f"**Candidates:** {att.get('candidate_count', 0)}")
                sc2.caption(f"**Fused:** {att.get('fused_count', 0)}")
                lat = att.get("retrieval_ms", 0.0)
                sc3.caption(f"**Latency:** {lat:.1f} ms" if isinstance(lat, (int, float)) else f"**Latency:** {lat} ms")

                st.markdown(f"**Evidence Level:** `{att_level}` ({att_suff})")
                if att.get("evidence_reason"):
                    st.caption(f"**Reason:** {att.get('evidence_reason')}")

                if att.get("page_match_found") is not None or att.get("chapter_match_found") is not None:
                    pm = "Yes" if att.get("page_match_found") else "No"
                    cm = "Yes" if att.get("chapter_match_found") else "No"
                    st.caption(f"**Page Match Found:** {pm} | **Chapter Match Found:** {cm}")

                if att.get("retry_reason"):
                    st.markdown(f"\u26a0\ufe0f **Why Retry:** {att.get('retry_reason')}")
                if att.get("changes_from_previous"):
                    st.info(f"\U0001f504 **Changes from Previous:** {att.get('changes_from_previous')}")

        timings = agent_activity.get("timings_ms", {})
        if timings:
            st.markdown("---")
            tc1, tc2 = st.columns(2)
            tc1.caption(f"\u23f1\ufe0f **Controller Total:** {timings.get('controller_total', 0)} ms")
            tc2.caption(f"\u23f1\ufe0f **Last Retrieval:** {timings.get('last_retrieval', 0)} ms")


def _render_event(event: dict) -> None:
    """Render a single live controller execution event."""
    event_type = event.get("type")

    if event_type == "plan":
        q_type = event.get("question_type", "general")
        plan = event.get("retrieval_plan", "")
        expl = event.get("explanation", "")
        st.markdown(f"📋 **Planning:** Question classified as `{q_type}`")
        if plan:
            st.caption(f"**Strategy:** `{plan}`")
        if expl:
            st.info(f"💡 **Controller Rationale:** {expl}")

    elif event_type in ("retrieval", "retrieval_start"):
        att = event.get("attempt", 1)
        mode = event.get("mode", "")
        search_text = event.get("search_text", "")
        st.markdown(f"🔍 **Attempt {att} — Dispatching `{mode}` Retrieval**")
        if search_text:
            st.caption(f"**Search Query:** `{search_text}`")

    elif event_type == "tool_calls":
        tools = event.get("tools_called") or []
        tools_str = ", ".join(f"`{t}`" for t in tools) if tools else "—"
        st.markdown(f"⚙️ **Tools Executed:** {tools_str}")
        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Retrieved:** {event.get('candidate_count', 0)} chunks")
        c2.caption(f"**Fused Pool:** {event.get('fused_count', 0)} chunks")
        c3.caption(f"**Tool Latency:** {event.get('retrieval_ms', 0):.1f} ms")
        if event.get("page_match_found") or event.get("chapter_match_found"):
            matches = []
            if event.get("page_match_found"):
                matches.append("Page match ✅")
            if event.get("chapter_match_found"):
                matches.append("Chapter match ✅")
            st.caption(" | ".join(matches))

    elif event_type == "evidence":
        level = str(event.get("level", "none")).upper()
        sufficient = event.get("sufficient", False)
        emoji = "✅" if sufficient else "⚠️"
        status_text = "Sufficient to answer" if sufficient else "Insufficient evidence"
        score = event.get("top_score")
        score_src = event.get("score_source")
        score_info = f" | Top {score_src} score: `{score}`" if score else ""
        st.markdown(f"{emoji} **Evidence Evaluation:** `{level}` ({status_text}{score_info})")
        if event.get("reason"):
            st.caption(f"**Evaluation Detail:** {event.get('reason')}")

    elif event_type == "adaptive_decision":
        att = event.get("attempt", 2)
        change = event.get("strategy_change", "")
        new_mode = event.get("new_mode", "")
        new_q = event.get("new_search_text", "")
        reason = event.get("reason", "")
        st.warning(f"🔄 **Adaptive Decision (Attempt {att}):** {change or reason}")
        if new_q:
            st.caption(f"**Revised Query:** `{new_q}` | **New Mode:** `{new_mode}`")

    elif event_type == "retry":
        if event.get("reason"):
            st.caption(f"⚠️ *{event.get('reason')}*")

    elif event_type == "decision":
        decision = event.get("decision")
        level = str(event.get("evidence_level", "none")).capitalize()
        reason = event.get("reason", "")
        if decision == "answer":
            st.success(f"✨ **Final Decision:** ACCEPTED ({level} evidence) — proceeding to generate answer.")
        else:
            st.error(f"⏹️ **Final Decision:** ABSTAIN ({level} evidence) — {reason or 'refusing to hallucinate without sufficient evidence'}.")

    elif event_type == "generation_start":
        chunks = event.get("context_chunks", 0)
        chunks_info = f" from {chunks} context chunks" if chunks else ""
        st.markdown(f"🤖 **Generating grounded answer with LLM**{chunks_info}...")

    elif event_type == "generation_complete":
        st.caption(f"✅ Answer generated in {event.get('duration_ms', 0):.0f} ms")

    elif event_type == "citations_start":
        st.markdown("🔗 **Verifying claim citations against retrieved passages...**")

    elif event_type == "citations_complete":
        supported = event.get("supported", 0)
        unsupported = event.get("unsupported", 0)
        total = event.get("total", supported + unsupported)
        if unsupported == 0 and total > 0:
            st.caption(f"✅ All {total} extracted claims directly grounded in evidence.")
        elif total > 0:
            st.caption(f"⚠️ Citations: {supported}/{total} claims supported by retrieved context.")

    elif event_type == "confidence_start":
        st.caption("📈 Computing evidence-backed confidence score...")

    elif event_type == "confidence_complete":
        st.caption(f"🎯 **Confidence:** `{event.get('level')}` ({event.get('score', 0):.0%})")


def _render_answer_extras(meta: dict) -> None:
    """Render mode, evidence, confidence, citations, sources and retrieved context under an answer."""
    _render_agent_activity(meta.get("agent_activity") or {})

    chip_col, evidence_col = st.columns([1, 1])

    with chip_col:
        render_html(render_mode_chip(meta.get("mode_label", "")))

    with evidence_col:
        if meta.get("trace_available"):
            render_html(
                render_evidence_badge(
                    meta.get("evidence_label", "Not available"),
                    str(meta.get("evidence_level") or "none"),
                )
            )

    confidence = meta.get("confidence")
    if confidence:
        render_html(render_confidence_card(confidence))

    coverage = meta.get("citation_coverage")
    kb_state = meta.get("kb_state")
    if coverage or kb_state:
        sub_col1, sub_col2 = st.columns([1, 1])
        with sub_col1:
            if coverage:
                render_html(render_citation_coverage_badge(coverage))
        with sub_col2:
            if kb_state:
                v = kb_state.get("version", 1)
                sync = " Synced" if kb_state.get("indexes_consistent", True) else " Mismatch"
                st.caption(f"Knowledge Base: **v{v}** ({sync})")

    sources = meta.get("supporting_sources") or []

    if sources:
        render_html(render_sources_card(sources))
    else:
        st.caption("No supporting sources were selected for this answer.")

    citations = meta.get("citations") or []
    if citations:
        with st.expander(f"Claim Citations & Evidence Passages ({len(citations)})", expanded=False):
            for c in citations:
                render_html(render_citation_card(c))

    # ========================================================================
    # RETRIEVAL METRICS & CITATION TRACE BREAKDOWN (Requirements 4 & 5)
    # ========================================================================
    context_chunks = meta.get("context_chunks") or []
    candidates = meta.get("candidates") or []
    citations = meta.get("citations") or []

    retrieved_count = meta.get("candidate_count") or len(candidates) or len(context_chunks)
    sent_count = len(context_chunks)

    # Chunks actually cited as evidence in verified claims
    cited_chunk_ids = {str(c.get("chunk_id")) for c in citations if c.get("chunk_id")}
    cited_chunks = [c for c in context_chunks if str(c.get("chunk_id")) in cited_chunk_ids]
    unused_chunks = [c for c in context_chunks if str(c.get("chunk_id")) not in cited_chunk_ids]
    cited_count = len(cited_chunks)
    unused_count = len(unused_chunks)

    st.markdown("#### 🔬 Retrieval Results & Evidence Trace")

    ret_ms = meta.get("retrieval_time_ms") or meta.get("retrieval_ms")
    col1, col2 = st.columns(2)
    col1.metric("Retrieval Latency", f"{ret_ms:.0f} ms" if isinstance(ret_ms, (int, float)) and ret_ms > 0 else "—")
    llm_ms = meta.get("llm_ms") or meta.get("llm_time_ms")
    col2.metric("Response Tokens", f"{meta.get('output_tokens', 0)} tokens" if meta.get('output_tokens') else "—")

    # Chapter distribution of retrieved context
    context_chunks = meta.get("context_chunks") or []
    if context_chunks:
        chapter_dist = {}
        for c in context_chunks:
            ch = c.get("chapter") or c.get("chapter_num")
            try:
                ch_int = int(ch) if ch else None
            except (ValueError, TypeError):
                ch_int = None
            key = f"Ch. {ch_int}" if ch_int else "Unknown"
            chapter_dist[key] = chapter_dist.get(key, 0) + 1

        if len(chapter_dist) > 1:
            with st.expander("📚 Retrieved Context by Chapter", expanded=False):
                st.caption("Distribution of final context chunks across book chapters.")
                st.bar_chart(chapter_dist)

    if cited_chunks:
        with st.expander(f"✅ Chunks Used as Citation Evidence ({cited_count})", expanded=True):
            st.caption("These chunks were sent to the LLM and directly cited to verify factual statements in the answer.")
            _render_chunk_list(cited_chunks, mode_label="Cited Evidence")

    if unused_chunks:
        with st.expander(f"ℹ️ Retrieved Chunks Not Cited ({unused_count})", expanded=False):
            st.caption("These chunks were passed to the LLM as background/continuity context, but were not directly cited in the final claim evidence.")
            _render_chunk_list(unused_chunks, mode_label="Uncited Context")

    if not context_chunks and candidates:
        with st.expander(f"Retrieved Candidates Pool ({len(candidates)})"):
            _render_chunk_list(candidates, mode_label=meta.get("mode_label", ""))

    if meta.get("error"):
        _service_error("GENERATION SERVICE UNAVAILABLE", meta["error"])


# ============================================================================
# NOTICES
# ============================================================================

def _show_ingestion_notice() -> None:
    """Display a one-time ingestion result after a rerun."""
    notice = st.session_state.pop("ingestion_notice", None)

    if not notice:
        return

    notice_type = notice.get("type", "info")
    message = notice.get("message", "")

    if notice_type == "success":
        st.success(message)
    elif notice_type == "warning":
        st.warning(message)
    elif notice_type == "error":
        st.error(message)
    else:
        st.info(message)


# ============================================================================
# INGESTION
# ============================================================================

def render_ingestion_health_panel(result: dict) -> None:
    """
    Ingestion Health Panel:
    Displays real-time document telemetry:
    - Pages processed
    - OCR pages
    - Image captions / figures
    - Chunks created
    - Skipped sections
    """
    if not result:
        return

    st.markdown("#### Ingestion Status")
    filename = result.get("_filename") or result.get("filename") or "Document"
    skipped = bool(result.get("skipped"))

    if skipped:
        st.info(f"**{filename}** is up-to-date (SHA-256 fingerprint verified). Embeddings preserved.")
    else:
        st.success(f"**{filename}** successfully indexed into the Knowledge Base.")

    c1, c2, c3, c4, c5 = st.columns(5)
    
    pages = result.get("pages_processed")
    if pages is None:
        pages = result.get("total_pages", "—")
    c1.metric("Pages", pages, help="Total pages processed")

    ocr_pages = result.get("ocr_pages", 0)
    c2.metric("OCR Pages", ocr_pages, help="Pages requiring OCR")

    captions = result.get("image_captions", 0)
    c3.metric("Figures", captions, help="Images and captions detected")

    chunks_created = result.get("chunks_created", result.get("chunks", "—"))
    c4.metric("Chunks", chunks_created, help="Total chunks created")

    skipped_sec = result.get("skipped_sections", 0)
    c5.metric("Skipped", skipped_sec, help="Blank sections skipped")

    # Detailed metrics row
    t1, t2, t3, t4 = st.columns(4)
    chunks_added = result.get("chunks_added", "—")
    t1.caption(f"**Chunks Added:** {chunks_added}")
    
    collection_count = result.get("collection_count", "—")
    t2.caption(f"**Total Indexed:** {collection_count}")
    
    ingest_time = result.get("ingestion_time_ms")
    t3.caption(f"**Latency:** {f'{ingest_time:.0f} ms' if ingest_time is not None else '—'}")
    
    hash_val = result.get("hash")
    short_hash = f"{hash_val[:10]}..." if hash_val else "—"
    t4.caption(f"**Fingerprint:** `{short_hash}`")


def _ingest_uploaded_file(pipeline: RAGPipeline, uploaded, rerun: bool = False):
    """
    Run incremental ingestion for an uploaded file with transparent real-time status steps.
    """
    raw_bytes: bytes = uploaded.getvalue()

    with st.status(f"Ingesting {uploaded.name}...", expanded=True) as status:
        status.write("📄 Reading file and computing SHA-256 fingerprint...")

        def _on_progress(step_message: str, frac: float):
            status.write(step_message)

        try:
            result = pipeline.ingest_file(
                file_path=uploaded.name,
                content=raw_bytes,
                progress_callback=_on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"❌ Ingestion failed for {uploaded.name}", state="error")
            _service_error("INGESTION SERVICE UNAVAILABLE", exc)
            st.caption("No existing data was removed.")
            return None

        is_skipped = result.get("skipped")
        complete_label = (
            f"⏭ {uploaded.name} already up-to-date (hash match)"
            if is_skipped
            else f"✅ Successfully ingested {uploaded.name}!"
        )
        status.update(label=complete_label, state="complete", expanded=False)

    result = dict(result or {})
    result["_filename"] = uploaded.name
    result["_characters"] = len(raw_bytes)

    st.session_state.last_ingestion = result

    if result.get("skipped"):
        message = (
            f"{uploaded.name} is already up-to-date. "
            "No embeddings were changed."
        )
        notice_type = "info"
    else:
        message = (
            f"{uploaded.name} added: "
            f"{result.get('chunks_added', 'unknown')} new chunks indexed. "
            f"Total indexed: {result.get('collection_count', 'unknown')}."
        )
        notice_type = "success"

    st.session_state.ingestion_notice = {
        "type": notice_type,
        "message": message,
    }

    # Mark that the health panel should be shown after rerun.
    # Rendering it here (inside the popover) would be lost on rerun,
    # so we persist the flag and render it in the main chat area instead.
    st.session_state.show_ingestion_panel = True

    # Invalidate the status cache so sidebar chunk count refreshes after rerun
    st.session_state.pop("pipeline_status", None)
    st.session_state.pop("_status_ts", None)
    st.session_state.pop("_kb_snapshot", None)

    if rerun:
        st.rerun()

    return result


def _replay_ingestion_stages(result: dict) -> None:
    """
    Replay the ingestion stages as a visualization.

    This animation is drawn after ingestion has already finished. It is
    not a live event stream. Stage details show real values from the
    ingestion result, and are blank where the backend reports nothing.
    """
    details = [
        result.get("_filename"),
        None,
        result.get("chunks_created", result.get("chunks")),
        None,
        result.get("collection_count"),
        result.get("bm25_chunks", result.get("bm25_synced")),
        "Queryable",
    ]

    placeholder = st.empty()
    progress = st.progress(0.0)

    total = len(INGESTION_STAGES)

    for step in range(1, total + 1):
        stages = []

        for index, stage in enumerate(INGESTION_STAGES):
            state = "done" if index < step else "pending"
            detail = details[index]

            stages.append(
                (state, stage[1], detail if detail is not None else "")
            )

        with placeholder.container():
            render_html(render_stage_rail(stages))

        progress.progress(step / total)
        time.sleep(0.12)

    progress.empty()


def _render_ingestion_result(result: dict) -> None:
    """Show the real values returned by pipeline.ingest_file(...)."""
    render_html(render_section_title("Ingestion result"))

    skipped = bool(result.get("skipped"))

    if skipped:
        st.info(
            "This file was unchanged, so incremental ingestion skipped it. "
            "No embeddings were recomputed."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", result.get("_characters", "N/A"))

    chunks_created = result.get("chunks_created", result.get("chunks"))
    col2.metric(
        "Chunks created",
        chunks_created if chunks_created is not None else "N/A",
    )

    chunks_added = result.get("chunks_added")
    col3.metric(
        "Chunks added",
        chunks_added if chunks_added is not None else "N/A",
    )

    render_html(
        render_kv_rows(
            [
                ("File", _na(result.get("_filename"))),
                ("Skipped (unchanged)", "Yes" if skipped else "No"),
                ("Collection count", _na(result.get("collection_count"))),
                (
                    "BM25 synchronization",
                    _na(
                        result.get(
                            "bm25_synced",
                            result.get("bm25_status"),
                        )
                    ),
                ),
            ]
        )
    )

    missing = [
        key
        for key in ("chunks_created", "bm25_synced")
        if key not in result
    ]

    if missing:
        st.caption(
            "Fields shown as N/A are not reported by the current ingestion "
            "backend. They are not estimated here."
        )


# ============================================================================
# PAGE  MISSION CONTROL
# ============================================================================

def _render_overview_cards(
    status: dict,
    collection_count,
    llm_model,
) -> None:
    """Four status cards driven entirely by real application state."""
    col1, col2, col3, col4 = st.columns(4)

    index_ready = bool(status.get("index_available"))
    bm25_ready = bool(status.get("bm25_available"))
    embedding_model = status.get("embedding_model")

    with col1:
        render_html(
            render_status_card(
                "Knowledge Core",
                collection_count,
                "Indexed chunks",
            )
        )

    with col2:
        render_html(
            render_status_card(
                "Vector Index",
                "ONLINE" if index_ready else "OFFLINE",
                "Chroma",
                state="online" if index_ready else "offline",
                small=True,
            )
        )

    with col3:
        render_html(
            render_status_card(
                "BM25 Index",
                "ONLINE" if bm25_ready else "OFFLINE",
                "Lexical search",
                state="online" if bm25_ready else "offline",
                small=True,
            )
        )

    with col4:
        render_html(
            render_status_card(
                "Language Model",
                llm_model if llm_model else None,
                "Configured model",
                state="online" if llm_model else "unknown",
                small=True,
            )
        )

    st.caption(
        f"Embedding model: {embedding_model or 'N/A'}  "
        f"Source documents: {status.get('txt_files_found', 'N/A')}"
    )


def _render_mission_control(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    render_html(
        render_page_header(
            "ASTRONOMY OBSERVATORY",
            "Mission Control",
            "Knowledge Intelligence & Retrieval System",
        )
    )

    _render_overview_cards(status, collection_count, _llm_model(status))

    st.write("")

    # System diagnostics
    with st.expander("🔧 System Diagnostics", expanded=False):
        diag_c1, diag_c2, diag_c3 = st.columns(3)

        # Chroma
        try:
            chroma_count = pipeline.vector_store.get_collection_count()
            chroma_dim   = pipeline.vector_store.get_collection_embedding_dimension()
            chroma_name  = pipeline.vector_store.collection_name
        except Exception:
            chroma_count, chroma_dim, chroma_name = 0, None, "knowledge_base"

        diag_c1.metric("Chroma Collection", chroma_name)
        diag_c1.metric("Vector Count", chroma_count)
        diag_c1.metric("Embedding Dimension", chroma_dim or "—")

        # BM25 & embedding model
        bm25_status = "Online" if status.get("bm25_available") else "Offline"
        embed_model = status.get("embedding_model") or "—"
        diag_c2.metric("BM25 Status", bm25_status)
        diag_c2.metric("Embedding Model", str(embed_model)[:30])
        diag_c2.metric("Source Documents", status.get("txt_files_found", "—"))

        # KB state
        kb_state_obj = get_current_kb_state(pipeline) if hasattr(pipeline, "vectorstore_path") else {}
        kb_version = kb_state_obj.get("version", "—") if isinstance(kb_state_obj, dict) else getattr(kb_state_obj, "version", "—")
        indexes_consistent = kb_state_obj.get("indexes_consistent", True) if isinstance(kb_state_obj, dict) else getattr(kb_state_obj, "indexes_consistent", True)
        diag_c3.metric("KB Version", kb_version)
        diag_c3.metric("Index Consistency", "✅ Synced" if indexes_consistent else "⚠️ Mismatch")
        diag_c3.metric("Vectorstore Path", str(pipeline.vectorstore_path)[-30:])

    render_html(render_section_title("Astronomy Knowledge Assistant"))

    st.caption(
        "Ask questions about planets, missions, cosmology, telescopes, "
        "astrophysics and space technology."
    )

    # ----------------------------------------------------------------
    # RETRIEVAL ENGINE - segmented control wired to the real backend
    # ----------------------------------------------------------------
    current = st.session_state.retrieval_mode

    if current not in MODE_ORDER:
        current = "hybrid"

    render_html(render_control_label("Retrieval Engine"))

    segment = _segmented_control(
        [MODES[key]["segment"] for key in MODE_ORDER],
        MODES[current]["segment"],
        key="retrieval_segment",
    )

    selected_mode = SEGMENT_TO_MODE.get(segment, current)
    st.session_state.retrieval_mode = selected_mode

    render_html(render_mode_description(MODES[selected_mode]["caption"]))

    if selected_mode in ("vector", "hybrid") and not status.get(
        "index_available"
    ):
        st.warning(
            "The vector index is not available. BM25 mode remains usable "
            "if the lexical index is present."
        )

    if selected_mode in ("bm25", "hybrid") and not status.get(
        "bm25_available"
    ):
        st.warning("The BM25 index is not available for this query.")

    # ----------------------------------------------------------------
    # INGESTION HEALTH PANEL (shown once after + upload)
    # ----------------------------------------------------------------
    if st.session_state.pop("show_ingestion_panel", False):
        result = st.session_state.get("last_ingestion")
        if result:
            render_ingestion_health_panel(result)
            st.divider()

    # ----------------------------------------------------------------
    # HISTORY
    # ----------------------------------------------------------------
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = _load_persistent_chat_history(pipeline.vectorstore_path)

    if st.session_state.chat_history:
        _, clear_col = st.columns([8.8, 1.2])
        with clear_col:
            if st.button("Clear", key="clear_chat_mc", help="Clear conversation history"):
                st.session_state.chat_history = []
                _clear_persistent_chat_history(pipeline.vectorstore_path)
                st.rerun()

    if not st.session_state.chat_history:
        st.info(
            "No queries have been run yet. Ask a question below to bring "
            "the retrieval pipeline online."
        )

        if not status.get("index_available"):
            st.warning(
                "No vector index found. Use Sync Knowledge Base in the "
                "sidebar, or run: python ingest.py"
            )

    for idx, entry in enumerate(st.session_state.chat_history):
        with st.chat_message(entry["role"]):
            # Add delete button for each message pair (user + assistant)
            if entry["role"] == "user":
                msg_col, del_col = st.columns([0.95, 0.05])
                with msg_col:
                    st.markdown(entry["content"])
                with del_col:
                    if st.button("🗑️", key=f"del_msg_{idx}", help="Delete this conversation"):
                        # Remove both user message and the following assistant response
                        to_remove = [idx]
                        if idx + 1 < len(st.session_state.chat_history) and st.session_state.chat_history[idx + 1]["role"] == "assistant":
                            to_remove.append(idx + 1)
                        for i in sorted(to_remove, reverse=True):
                            st.session_state.chat_history.pop(i)
                        _save_persistent_chat_history(pipeline.vectorstore_path, st.session_state.chat_history)
                        st.rerun()
            else:
                events = entry.get("meta", {}).get("events")
                if events:
                    with st.status("🤖 Agent Execution Trace", state="complete", expanded=False):
                        for ev in events:
                            _render_event(ev)
                if entry.get("meta", {}).get("response_type") == "error":
                    st.error(entry["content"])
                else:
                    st.markdown(entry["content"])

            if entry["role"] == "assistant":
                _render_answer_extras(entry.get("meta", {}))

    # ----------------------------------------------------------------
    # NEW QUESTION
    # ----------------------------------------------------------------
    question = _render_custom_composer(pipeline)

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Real-time live execution trace driven directly by the controller
        status_box = st.status("🤖 Planning & executing...", expanded=True)

        def handle_live_event(event: dict):
            with status_box:
                _render_event(event)
                etype = event.get("type")
                if etype == "plan":
                    status_box.update(
                        label=f"📋 Planning: {event.get('question_type', '')} ({event.get('retrieval_plan', '')})",
                        state="running",
                    )
                elif etype in ("retrieval", "retrieval_start"):
                    status_box.update(
                        label=f"🔍 Attempt {event.get('attempt')}: Executing {event.get('mode')} retrieval...",
                        state="running",
                    )
                elif etype == "tool_calls":
                    tools = ", ".join(event.get("tools_called", []))
                    status_box.update(
                        label=f"⚙️ Tools: {tools} ({event.get('fused_count', 0)} chunks)",
                        state="running",
                    )
                elif etype == "evidence":
                    suff = "Sufficient" if event.get("sufficient") else "Insufficient"
                    status_box.update(
                        label=f"⚖️ Evidence: {str(event.get('level', '')).upper()} ({suff})",
                        state="running",
                    )
                elif etype in ("retry", "adaptive_decision"):
                    status_box.update(
                        label=f"🔄 Adapting strategy & retrying (Attempt {event.get('attempt', 1)})...",
                        state="running",
                    )
                elif etype == "decision":
                    if event.get("decision") == "answer":
                        status_box.update(
                            label="✨ Evidence accepted — generating answer...",
                            state="running",
                        )
                    else:
                        status_box.update(
                            label="⏹️ Insufficient evidence — abstaining...",
                            state="running",
                        )
                elif etype == "generation_start":
                    status_box.update(
                        label="✍️ Generating grounded answer...",
                        state="running",
                    )
                elif etype == "citations_start":
                    status_box.update(
                        label="🔗 Verifying claim citations...",
                        state="running",
                    )
                elif etype == "confidence_start":
                    status_box.update(
                        label="🎯 Computing confidence...",
                        state="running",
                    )

        _app_query_t0 = time.perf_counter()
        try:
            try:
                response = pipeline.answer_question(
                    question,
                    retrieval_mode=selected_mode,
                    on_event=handle_live_event,
                )
            except TypeError as te:
                if "on_event" in str(te):
                    # In case of stale instance or monkeypatch, fallback safely
                    response = pipeline.answer_question(
                        question,
                        retrieval_mode=selected_mode,
                    )
                else:
                    raise te
        except Exception as exc:  # noqa: BLE001
            _err_elapsed_ms = (time.perf_counter() - _app_query_t0) * 1000
            response = {
                "answer": (
                    f"Query execution failed ({type(exc).__name__}: {exc}). "
                    "If the embedding or generation service is unavailable, "
                    "BM25 mode does not require embeddings and may still work."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "error": str(exc),
                "response_type": "error",
                "retrieval_mode": selected_mode,
                "retrieval_ms": round(_err_elapsed_ms, 2),
                "trace": {
                    "retrieval": {
                        "mode": selected_mode,
                        "retrieval_ms": round(_err_elapsed_ms, 2),
                        "llm_ms": 0.0,
                    }
                },
                "events": [],
            }

        # Mark complete
        resp_type = response.get("response_type")
        stop_reason = (response.get("agent_activity") or {}).get("stop_reason", "")
        if resp_type == "error":
            status_box.update(label="❌ Generation error", state="error", expanded=True)
        elif stop_reason == "page_not_found":
            status_box.update(label="📄 Page not found — abstained", state="complete", expanded=True)
        elif response.get("answer") and resp_type != "abstain":
            status_box.update(label="✅ Answer generated & grounded", state="complete", expanded=False)
        else:
            status_box.update(label="⏹️ Abstained (insufficient evidence)", state="complete", expanded=False)

        st.session_state.last_response = response

        answer = response.get("answer", "No answer available.")
        meta = _build_meta(response, selected_mode)
        meta["events"] = response.get("events", [])

        if resp_type == "error":
            st.error(answer)
        else:
            st.markdown(answer)
        _render_answer_extras(meta)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_entry = {"role": "user", "content": question, "timestamp": now_str}
    assistant_entry = {
        "role": "assistant",
        "content": answer,
        "meta": meta,
        "timestamp": now_str,
    }
    st.session_state.chat_history.append(user_entry)
    st.session_state.chat_history.append(assistant_entry)
    _save_persistent_chat_turn(pipeline.vectorstore_path, user_entry, assistant_entry)

    st.rerun()


# ============================================================================
# COMPOSER
# ============================================================================

def _render_custom_composer(pipeline: RAGPipeline):
    """
    Render the astronomy composer.

    The question input and send button are inside a Streamlit form, so
    pressing Enter submits the question just like clicking the arrow.
    A versioned widget key creates a fresh empty input after submission.
    """
    if "composer_version" not in st.session_state:
        st.session_state.composer_version = 0

    plus_col, composer_col = st.columns([0.75, 10.25], gap="small")

    # ----------------------------------------------------------------
    # PLUS / KNOWLEDGE UPLOAD
    # ----------------------------------------------------------------
    with plus_col:
        with st.popover("+", help="Add knowledge to the knowledge base"):
            st.markdown("**Add knowledge**")
            st.caption(
                "Upload a TXT, Markdown, DOCX, or PDF file. Indexing is incremental, "
                "so existing knowledge is preserved."
            )

            uploaded = st.file_uploader(
                "Choose a file",
                type=["txt", "md", "docx", "pdf"],
                key="composer_file_uploader",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                st.caption(f"{uploaded.name}")

                if st.button(
                    "Add to Knowledge Base",
                    key="composer_add_file",
                    **WIDE,
                ):
                    _ingest_uploaded_file(pipeline, uploaded, rerun=True)

    # ----------------------------------------------------------------
    # QUESTION COMPOSER
    # ----------------------------------------------------------------
    send_clicked = False
    question = ""

    with composer_col:
        with st.form(
            key=f"question_form_{st.session_state.composer_version}",
            clear_on_submit=False,
            border=False,
        ):
            input_col, send_col = st.columns([9.35, 0.65], gap="small")

            with input_col:
                question = st.text_input(
                    "Ask a question",
                    placeholder="Ask the universe something...",
                    key=(
                        f"custom_question_"
                        f"{st.session_state.composer_version}"
                    ),
                    label_visibility="collapsed",
                )

            with send_col:
                send_clicked = st.form_submit_button(
                    "→",
                    help="Send (or press Enter)",
                    **WIDE,
                )

    if send_clicked:
        question = (question or "").strip()

        if not question:
            return None

        # Do not assign directly to the current widget's session-state key.
        # Instead, create a new widget key on the next rerun. This clears
        # the input without triggering Streamlit's widget-state exception.
        st.session_state.composer_version += 1

        return question

    return None


# ============================================================================
# PAGE  RETRIEVAL LAB
# ============================================================================

def _render_retrieval_pipeline_flow(retrieval: dict) -> None:
    """
    Draw the retrieval flow for the mode that actually ran.

    Vector mode does not show BM25, BM25 mode does not show dense search,
    and RRF only appears when fusion actually happened.
    """
    dense = bool(retrieval.get("dense_enabled"))
    bm25 = bool(retrieval.get("bm25_enabled"))
    rrf = bool(retrieval.get("rrf_enabled"))

    render_html(render_section_title("Retrieval flow for this query"))

    top = st.columns(3)

    with top[1]:
        render_html(
            render_arch_node("", "Query", "user input", "processing")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Query Preparation", "normalize / tokenize", "processing"
            )
        )
        render_html(render_arch_arrow())

    # Retrieval branch  only the retrievers that ran.
    if dense and bm25:
        branch = st.columns(2)

        with branch[0]:
            render_html(
                render_arch_node(
                    "", "Dense Search", "Chroma vectors", "retrieval"
                )
            )

        with branch[1]:
            render_html(
                render_arch_node(
                    "", "BM25 Search", "lexical match", "retrieval"
                )
            )

    else:
        single = st.columns(3)

        with single[1]:
            if dense:
                render_html(
                    render_arch_node(
                        "", "Dense Search", "Chroma vectors", "retrieval"
                    )
                )
            elif bm25:
                render_html(
                    render_arch_node(
                        "", "BM25 Search", "lexical match", "retrieval"
                    )
                )
            else:
                render_html(
                    render_arch_node(
                        "", "No retriever recorded", "", "retrieval"
                    )
                )

    tail = st.columns(3)

    with tail[1]:
        render_html(render_arch_arrow())

        if rrf:
            render_html(
                render_arch_node(
                    "",
                    "RRF Fusion",
                    f"k = {retrieval.get('rrf_k', 'N/A')}",
                    "retrieval",
                )
            )
        else:
            render_html(
                render_arch_node(
                    "", "RRF  Not used", "single retriever", "retrieval"
                )
            )

        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Evidence Gate", "sufficiency check", "decision"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Context Selection", "chunk budget", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Generation", "configured LLM", "generation"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Answer", "sources + trace", "generation"
            )
        )


def _render_retrieval_lab(response) -> None:
    render_html(
        render_page_header(
            "RUNTIME DIAGNOSTICS",
            "Retrieval Lab",
            "Every panel below reads the trace of the most recent query. "
            "Opening this page runs no retrieval and calls no model.",
        )
    )

    if not response:
        st.info(
            " No retrieval trace available.\n\n"
            "Run a query from Mission Control to inspect the retrieval "
            "pipeline here."
        )
        return

    if not has_trace(response):
        st.warning(
            "The last response carried no runtime trace, so there is "
            "nothing to observe. This usually means the query failed "
            "before retrieval ran."
        )

        if response.get("error"):
            _service_error("RETRIEVAL SERVICE UNAVAILABLE", response["error"])

        return

    trace = response.get("trace") or {}
    snapshot = build_observatory_snapshot(trace)

    retrieval = snapshot.get("retrieval", {})

    _lab_section_query(snapshot.get("query", {}))
    _lab_section_mode(retrieval)
    _render_retrieval_pipeline_flow(retrieval)

    if retrieval.get("dense_enabled"):
        _lab_section_dense(retrieval)

    if retrieval.get("bm25_enabled"):
        _lab_section_bm25(retrieval)

    if retrieval.get("rrf_enabled"):
        _lab_section_rrf(retrieval)
    else:
        render_html(render_section_title("Reciprocal Rank Fusion"))
        st.info(
            "RRF  Not used. Fusion applies to Hybrid mode only, where "
            "two ranked lists exist to combine."
        )

    _lab_section_evidence(snapshot.get("evidence", {}))
    _lab_section_retrieval_breakdown(response)
    _lab_section_context(trace, snapshot.get("context", {}))
    _lab_section_generation(snapshot.get("generation", {}))
    _lab_section_grounding_chain(snapshot.get("query", {}), snapshot.get("grounding", {}))

    with st.expander("Raw trace (sanitized)"):
        st.caption(
            "Secret-looking fields are stripped by "
            "observatory.export_trace_json()."
        )
        st.json(export_trace_json(trace))


def _lab_section_grounding_chain(query: dict, grounding: dict) -> None:
    """Render the auditable Query -> Evidence -> Claim -> Citation chain."""
    render_html(render_section_title("Grounding chain"))
    st.caption("Query  selected evidence passage  generated claim  citation verification")
    rows = []
    for item in grounding.get("claim_to_citation", []):
        evidence = item.get("evidence", {})
        rows.append({
            "Query": query.get("original", ""),
            "Evidence": f"{evidence.get('filename', '')} / {evidence.get('chunk_id', '')}",
            "Claim": item.get("claim", ""),
            "Citation": item.get("citation", ""),
            "Verdict": item.get("status", ""),
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No generated claims were available for post-generation grounding.")


def _render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Evaluation Dashboard", "Aggregate live heuristics from recorded observations; these are not accuracy metrics."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    table = [{"query": row["raw_query"], "latency_ms": row["latency_ms"], "retrieval_health": row.get("evaluation", {}).get("retrieval_health_score"), "citation_coverage": row.get("answer_evaluation", {}).get("citation_coverage", {}).get("score")} for row in rows]
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_benchmark_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Benchmark", "Labelled queries are evaluated separately from live heuristic observations."))
    st.info("Add labelled cases with query, relevant_documents/relevant_chunks, expected_facets, and optional reference_answer. Benchmark calculation is available in src/rag_pipeline/benchmark.py.")


def _render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare two recorded executions without rerunning retrieval."))
    rows = load_observations(pipeline.vectorstore_path)
    if len(rows) < 2:
        st.info("Run at least two queries to compare observations.")
        return
    labels = [f"{row['timestamp']}  {row['raw_query']}" for row in rows]
    left, right = st.columns(2)
    with left:
        st.json(rows[labels.index(st.selectbox("First query", labels, key="compare_a"))])
    with right:
        st.json(rows[labels.index(st.selectbox("Second query", labels, index=1, key="compare_b"))])


# ============================================================================
# PAGE  INGESTION PIPELINE
# ============================================================================

def _render_ingestion_architecture() -> None:
    """Visual ingestion architecture using themed nodes."""
    top = st.columns(3)

    with top[1]:
        render_html(
            render_arch_node("", "Document", "TXT / MD", "storage")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Document Loader", "read + detect", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Text Processing", "normalize", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Chunking", "token pack", "processing"
            )
        )
        render_html(render_arch_arrow("  "))

    branch = st.columns(2)

    with branch[0]:
        render_html(
            render_arch_node(
                "", "Embedding Model", "vectorize", "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Chroma Vector Store", "dense index", "storage"
            )
        )

    with branch[1]:
        render_html(
            render_arch_node(
                "", "BM25", "term weighting", "retrieval"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "BM25 Index", "lexical index", "storage"
            )
        )


def _render_ingestion_simulation(pipeline: RAGPipeline) -> None:
    """
    Educational, read-only walkthrough of the ingestion stages.

    This section never calls ingest_file, ingest_documents, or any write
    path. It only reads configuration and the already-computed chunk
    preview, so it cannot modify Chroma, BM25 or the knowledge base.
    """
    st.warning(
        "EDUCATIONAL INGESTION SIMULATION  ILLUSTRATIVE ONLY. "
        "This walkthrough does not modify Chroma, BM25, the vectorstore "
        "or the knowledge base."
    )

    render_html(render_simulation_tag("Simulation  no database writes"))

    size, overlap = _chunk_config(pipeline)
    chunks = _load_chunks(pipeline)

    if st.button(" Run simulation", key="sim_run"):
        placeholder = st.empty()
        progress = st.progress(0.0)

        for step in range(1, len(SIMULATION_STEPS) + 1):
            stages = []

            for index, item in enumerate(SIMULATION_STEPS):
                state = "done" if index < step else "pending"
                stages.append((state, f"{item[0]}  {item[2]}", ""))

            with placeholder.container():
                render_html(render_stage_rail(stages))

            progress.progress(step / len(SIMULATION_STEPS))
            time.sleep(0.25)

        progress.empty()
        st.success("Simulation complete. No data was written.")

    for number, icon, title, description in SIMULATION_STEPS:
        with st.expander(f"{number}  {icon}  {title}"):
            st.write(description)

            if title == "CONTENT CHUNKED":
                render_html(
                    render_kv_rows(
                        [
                            ("Target tokens (pack goal)", _na(size)),
                            ("Max tokens per chunk", _na(overlap)),
                        ]
                    )
                )

                if chunks:
                    st.caption(
                        "A real chunk produced by the backend's chunker:"
                    )
                    st.code(
                        (chunks[0].get("text") or "")[:400],
                        language="text",
                    )


def _render_ingestion_page(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    render_html(
        render_page_header(
            "KNOWLEDGE INGESTION",
            "Ingestion Pipeline",
            "How documents become searchable knowledge.",
        )
    )

    # ----------------------------------------------------------------
    # REAL KNOWLEDGE CORE STATUS
    # ----------------------------------------------------------------
    render_html(render_section_title("Knowledge core"))

    size, overlap = _chunk_config(pipeline)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_html(
            render_status_card(
                "Documents",
                status.get("txt_files_found"),
                "Source files",
            )
        )

    with col2:
        render_html(
            render_status_card(
                "Chunks",
                collection_count,
                "Indexed",
            )
        )

    with col3:
        ready = bool(status.get("index_available"))
        render_html(
            render_status_card(
                "Vector Index",
                "ONLINE" if ready else "OFFLINE",
                "Chroma",
                state="online" if ready else "offline",
                small=True,
            )
        )

    with col4:
        ready = bool(status.get("bm25_available"))
        render_html(
            render_status_card(
                "BM25 Index",
                "ONLINE" if ready else "OFFLINE",
                "Lexical",
                state="online" if ready else "offline",
                small=True,
            )
        )

    render_html(
        render_kv_rows(
            [
                ("Target tokens", _na(size)),
                ("Max tokens", _na(overlap)),
                ("BM25 indexed chunks", _bm25_indexed_count(status)),
                ("Embedding model", _na(status.get("embedding_model"))),
                ("Vectorstore path", _na(status.get("vectorstore_path"))),
                ("Knowledge base path", str(KNOWLEDGE_BASE_DIR)),
                ("Reranking", "Not enabled in this system"),
            ]
        )
    )

    tab_arch, tab_sim, tab_upload = st.tabs(
        ["Ingestion architecture", "Educational simulation", "Real ingestion"]
    )

    with tab_arch:
        render_html(render_simulation_tag("Architecture diagram"))
        _render_ingestion_architecture()
        st.write("")
        with st.expander("Text version of this diagram"):
            st.code(INGESTION_DIAGRAM, language="text")

    with tab_sim:
        _render_ingestion_simulation(pipeline)

    with tab_upload:
        _ingestion_tab_real(pipeline)


def _ingestion_tab_real(pipeline: RAGPipeline) -> None:
    st.markdown("### 📥 Ingest New Document")
    st.caption("Upload a PDF, TXT, DOCX, or Markdown file. Large PDFs are supported. Duplicate files are automatically detected and skipped.")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["txt", "md", "docx", "pdf"],
        key="ingestion_page_uploader",
        help="PDFs: chapter headings, sections, and page numbers are preserved. Large books are chunked intelligently.",
    )

    if uploaded is not None:
        file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.info(f"📄 **{uploaded.name}** — {file_size_mb:.1f} MB")

        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("▶ Run Ingestion", key="ingestion_page_run", type="primary", use_container_width=True)
        with col2:
            st.caption("Existing chunks are preserved. Unchanged files are skipped automatically.")

        if run_btn:
            progress_bar = st.progress(0, text="Starting ingestion...")

            import time as _time
            t0 = _time.perf_counter()

            progress_bar.progress(10, text="Reading file...")
            result = _ingest_uploaded_file(pipeline, uploaded, rerun=False)
            elapsed = (_time.perf_counter() - t0) * 1000

            progress_bar.progress(100, text="Complete!")

            if result is not None:
                st.session_state.force_reload = True
                if result.get("skipped"):
                    st.warning(f"⏭ **{uploaded.name}** is already indexed — no changes made.")
                else:
                    st.success(f"✅ Ingestion complete in {elapsed:.0f} ms")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Chunks Added", result.get("chunks_added", "—"))
                    r2.metric("Total Indexed", result.get("collection_count", "—"))
                    r3.metric("File Size", f"{file_size_mb:.1f} MB")
                    r4.metric("Ingestion Time", f"{elapsed:.0f} ms")

                    if result.get("chunks_created"):
                        st.caption(f"📊 {result['chunks_created']} chunks created from this document")
                    if result.get("pages_processed"):
                        st.caption(f"📄 {result['pages_processed']} pages processed")

    st.divider()
    st.markdown("### 🗂 Batch Ingest All Files in Knowledge Base")
    st.caption(f"Knowledge base directory: `{KNOWLEDGE_BASE_DIR}`")

    kb_files = list_knowledge_files(Path(KNOWLEDGE_BASE_DIR))
    if kb_files:
        st.markdown(f"**{len(kb_files)} files** found in knowledge base:")
        for f in kb_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            st.caption(f"• {f.name} ({size_mb:.1f} MB)")

        if st.button("▶ Ingest All Files", key="batch_ingest_btn", type="secondary"):
            with st.spinner("Ingesting all knowledge base files..."):
                import time as _t
                t0 = _t.perf_counter()
                try:
                    result = pipeline.ingest_documents()
                    elapsed = (_t.perf_counter() - t0) * 1000
                    st.success(f"✅ Batch ingestion complete in {elapsed:.0f} ms")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Files Processed", result.get("files_processed", "—"))
                    b2.metric("Chunks Indexed", result.get("chunks_indexed", "—"))
                    b3.metric("Total in Index", result.get("collection_count", "—"))
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
    else:
        st.info("No files found in knowledge base. Upload a file above or place files in the knowledge_base/ directory.")

    # The KB management section is always shown, regardless of whether an
    # ingestion has been run in this session.
    _render_manage_knowledge_base(pipeline)

    result = st.session_state.get("last_ingestion")

    if not result:
        st.info("No ingestion has been run in this session yet.")
        return

    _render_ingestion_result(result)

    render_html(render_section_title("Stage replay"))
    render_html(render_simulation_tag("Pipeline visualization"))
    st.caption(
        "A replay of the stages drawn after ingestion finished. This is "
        "not live internal telemetry."
    )

    if st.button("Replay stages", key="ingestion_replay"):
        _replay_ingestion_stages(result)


def _render_manage_knowledge_base(pipeline: RAGPipeline) -> None:
    """Render the 'Manage Knowledge Base' section inside the real-ingestion tab.

    Lists all documents currently in the knowledge base and lets the user
    select and remove one. Calls pipeline.delete_document() which removes
    chunks from both Chroma and BM25 and (optionally) deletes the file from
    disk. Shows counts of deleted chunks after removal and triggers a rerun.
    """
    render_html(render_section_title("Manage Knowledge Base"))

    kb_path = Path(pipeline.knowledge_base_path)
    try:
        kb_files = list_knowledge_files(kb_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read knowledge base directory: {exc}")
        return

    if not kb_files:
        st.info("Knowledge base is empty.")
        return

    file_names = sorted(f.name for f in kb_files)

    selected_file = st.selectbox(
        "Select a document to remove",
        options=file_names,
        key="remove_doc_select",
    )

    if st.button("Remove Document", key="remove_doc_btn", type="primary"):
        try:
            result = pipeline.delete_document(
                selected_file,
                delete_file_from_disk=True,
            )
            chroma_del = result.get("deleted_from_chroma", 0)
            bm25_del = result.get("deleted_from_bm25", 0)
            st.success(
                f"Removed **{selected_file}**  "
                f"{chroma_del} chunk(s) deleted from Chroma, "
                f"{bm25_del} chunk(s) removed from BM25."
            )
            st.session_state.force_reload = True
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to remove document: {exc}")


# ==========================================================================
# RETRIEVAL LAB  SECTIONS
# ==========================================================================

def _lab_section_query(query: dict) -> None:
    render_html(render_section_title("Section A  Query"))

    render_html(
        render_trace_query_card(
            "ORIGINAL QUERY",
            query.get("original_query") or "Not available in current trace.",
        )
    )

    render_html(
        render_kv_rows(
            [
                ("Prepared query", _na(query.get("prepared_query"))),
                ("Normalization / aliases", _na(query.get("normalization"))),
                ("BM25 token count", _na(query.get("bm25_token_count"))),
                (
                    "Multi-intent detected",
                    "Yes" if query.get("is_multi_intent") else "No",
                ),
                (
                    "Greeting detected",
                    "Yes" if query.get("is_greeting") else "No",
                ),
            ]
        )
    )

    tokens = query.get("bm25_tokens") or []

    if tokens:
        with st.expander(f"BM25 tokens ({len(tokens)})"):
            st.code(" ".join(str(token) for token in tokens), language="text")
    else:
        st.caption("BM25 tokens: not available in current trace.")


def _lab_section_mode(retrieval: dict) -> None:
    render_html(render_section_title("Section B  Retrieval mode"))

    render_html(render_mode_chip(retrieval.get("mode_label", "Unknown")))

    render_html(
        render_kv_rows(
            [
                (
                    "Dense retrieval",
                    "Enabled"
                    if retrieval.get("dense_enabled")
                    else "Disabled",
                ),
                (
                    "BM25 retrieval",
                    "Enabled" if retrieval.get("bm25_enabled") else "Disabled",
                ),
                (
                    "RRF fusion",
                    "Enabled" if retrieval.get("rrf_enabled") else "Disabled",
                ),
                ("Candidate count", _na(retrieval.get("candidate_count"))),
                ("Fused count", _na(retrieval.get("fused_count"))),
                ("Dense results", _na(retrieval.get("dense_count"))),
                ("BM25 results", _na(retrieval.get("bm25_count"))),
                ("Vector top-k", _na(retrieval.get("vector_top_k"))),
                ("BM25 top-k", _na(retrieval.get("bm25_top_k"))),
                ("Hybrid top-k", _na(retrieval.get("hybrid_top_k"))),
                (
                    "Query embedding dimension",
                    _na(retrieval.get("query_embedding_dim")),
                ),
                (
                    "Reranking",
                    "Enabled"
                    if retrieval.get("reranker_enabled")
                    else "Disabled (by design)",
                ),
            ]
        )
    )


def _lab_section_dense(retrieval: dict) -> None:
    render_html(render_section_title("Vector retrieval  dense results"))

    rows = retrieval.get("raw_dense") or []

    if not rows:
        st.info("No dense results were recorded in this trace.")
        return

    _dataframe([
            {
                "Rank": row.get("rank"),
                **_extract_metadata(row),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "Distance": row.get("distance"),
                "Score": row.get("score"),
            }
            for row in rows
        ])

    if any(row.get("distance") is not None for row in rows):
        st.caption(
            "Distance is the raw value returned by the vector store. "
            "Lower distance indicates greater vector similarity."
        )


def _lab_section_bm25(retrieval: dict) -> None:
    render_html(render_section_title("BM25 retrieval  lexical results"))

    rows = retrieval.get("raw_bm25") or []

    if not rows:
        st.info("No BM25 results were recorded in this trace.")
        return

    _dataframe([
            {
                "Rank": row.get("rank"),
                **_extract_metadata(row),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "BM25 score": row.get("score"),
            }
            for row in rows
        ])

    st.caption(
        "BM25 scores are unbounded term-weighting scores, comparable "
        "within this result set only."
    )


def _lab_section_rrf(retrieval: dict) -> None:
    render_html(render_section_title("Reciprocal Rank Fusion"))

    k = retrieval.get("rrf_k", 60)

    render_html(
        render_formula_card(
            "RRF formula",
            "RRF(d) =  1 / (k + rank)",
            f"Configured k = {k}",
        )
    )

    rows = retrieval.get("rrf_calculation") or []

    if not rows:
        st.info("No RRF calculations were recorded in this trace.")
        return

    _dataframe([
            {
                "Final rank": row.get("final_rank"),
                **_extract_metadata(row),
                "Source": row.get("filename"),
                "Chunk": row.get("chunk_id"),
                "Dense rank": row.get("dense_rank"),
                "BM25 rank": row.get("bm25_rank"),
                "Dense contribution": row.get("dense_contribution"),
                "BM25 contribution": row.get("bm25_contribution"),
                "Final RRF": row.get("rrf_score"),
                "In both": "Yes" if row.get("agreement") else "No",
            }
            for row in rows
        ])

    st.caption(
        "A contribution of 0 means that retriever did not return the "
        "document at all."
    )

    with st.expander("Per-document arithmetic"):
        for row in rows:
            st.markdown(
                f"**{row.get('filename', 'unknown')} / "
                f"{row.get('chunk_id', 'unknown')}**"
            )

            lines = []

            if row.get("dense_formula"):
                lines.append(f"Dense: {row['dense_formula']}")
            else:
                lines.append("Dense: not retrieved (contribution 0)")

            if row.get("bm25_formula"):
                lines.append(f"BM25:  {row['bm25_formula']}")
            else:
                lines.append("BM25:  not retrieved (contribution 0)")

            if row.get("formula"):
                lines.append(f"RRF:   {row['formula']}")

            st.code("\n".join(lines), language="text")

    st.info(
        "RRF does not measure probability or confidence. It combines "
        "rankings from different retrieval systems so documents ranked "
        "highly by either system can be prioritized."
    )


def _lab_section_evidence(evidence: dict) -> None:
    render_html(render_section_title("Evidence evaluation"))

    render_html(
        render_evidence_badge(
            describe_evidence_level(evidence.get("level")),
            str(evidence.get("level") or "none"),
        )
    )

    render_html(
        render_kv_rows(
            [
                ("Evidence level", _na(evidence.get("level"))),
                (
                    "Should answer",
                    "Yes" if evidence.get("should_answer") else "No",
                ),
                ("Score source", _na(evidence.get("score_source"))),
                ("Supporting chunks", _na(evidence.get("supporting_chunks"))),
                (
                    "Retrieval agreement",
                    "Yes" if evidence.get("agreement") else "No",
                ),
                ("Top score", _na(evidence.get("top_score"))),
                ("Top gap", _na(evidence.get("top_gap"))),
                ("Top percentile", _na(evidence.get("top_percentile"))),
                ("Retrieval mode", _na(evidence.get("retrieval_mode"))),
                ("Reason", _na(evidence.get("reason"))),
            ]
        )
    )

    st.caption(
        "Confidence comes from this evidence layer, not from the RRF "
        "score. An RRF score is a rank-fusion weight and says nothing "
        "about how well the evidence supports an answer."
    )


def _lab_section_retrieval_breakdown(response: dict) -> None:
    """Display retrieved evidence summary by book and chapter."""
    observation = response.get("observation", {})
    breakdown = observation.get("retrieval_breakdown", {})
    
    if not breakdown or not breakdown.get("breakdown"):
        return
    
    render_html(render_section_title("Retrieved Evidence Summary"))
    
    # Display summary stats
    col1, col2 = st.columns(2)
    col1.metric("Books Retrieved", breakdown.get("total_books", 0))
    col2.metric("Chapters Retrieved", breakdown.get("total_chapters", 0))
    
    # Display breakdown table
    breakdown_items = breakdown.get("breakdown", [])
    if breakdown_items:
        breakdown_rows = []
        for item in breakdown_items:
            ch_num = item.get("chapter")
            try:
                ch_int = int(ch_num) if ch_num is not None else None
                chapter_display = f"Ch. {ch_num}" if ch_int and ch_int > 0 else ("Intro" if ch_int == 0 else "-")
            except (ValueError, TypeError):
                chapter_display = "-"
            
            breakdown_rows.append({
                "Book": item.get("book", "Unknown"),
                "Chapter": chapter_display,
                "Chapter Title": item.get("chapter_title", ""),
                "# Chunks": item.get("chunk_count", 0),
            })
        st.dataframe(breakdown_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No retrieval breakdown available.")


def _lab_section_context(trace: dict, context: dict) -> None:
    render_html(render_section_title("Context sent to the model"))

    chunks = get_context_chunks(trace)

    col1, col2, col3 = st.columns(3)

    selected = context.get("selected_chunks")
    col1.metric("Chunks selected", selected if selected is not None else "N/A")

    chars = context.get("context_chars")
    col2.metric("Context characters", chars if chars is not None else "N/A")

    sources = context.get("unique_sources")

    if isinstance(sources, list):
        source_display = len(sources)
    elif isinstance(sources, int):
        source_display = sources
    else:
        source_display = "N/A"

    col3.metric("Sources represented", source_display)

    render_html(
        render_kv_rows(
            [
                ("Input chunks", _na(context.get("input_chunks"))),
                ("Removed chunks", _na(context.get("removed_chunks"))),
                ("Input characters", _na(context.get("input_chars"))),
                ("Selected characters", _na(context.get("selected_chars"))),
                ("Max chunks", _na(context.get("max_chunks"))),
                ("Max characters", _na(context.get("max_chars"))),
            ]
        )
    )

    if isinstance(sources, list) and sources:
        st.markdown(
            "**Sources represented:** "
            + ", ".join(str(item) for item in sources)
        )

    if not chunks:
        st.caption("Final context chunks are not available in this trace.")
        return

    with st.expander(f"Final context chunks ({len(chunks)})"):
        for chunk in chunks:
            st.markdown(
                f"**{chunk.get('position')}. "
                f"{chunk.get('filename') or 'unknown'}**  "
                f"`{chunk.get('chunk_id') or 'unknown'}`  "
                f"{chunk.get('chars')} characters"
            )
            st.code(chunk.get("text", ""), language="text")


def _lab_section_generation(generation: dict) -> None:
    render_html(render_section_title("Generation"))

    # The endpoint URL is deliberately not displayed: it identifies the
    # provider and is environment-specific.
    render_html(
        render_kv_rows(
            [
                ("Status", _na(generation.get("status"))),
                ("LLM model", _na(generation.get("model"))),
                (
                    "Endpoint",
                    "Configured" if generation.get("base_url") else None,
                ),
                ("Context characters", _na(generation.get("context_chars"))),
                ("Messages sent", _na(generation.get("message_count"))),
                ("Temperature", _na(generation.get("temperature"))),
                ("Response characters", _na(generation.get("response_chars"))),
                (
                    "Procedural answer",
                    "Yes" if generation.get("procedural") else "No",
                ),
            ]
        )
    )

    if generation.get("explanation"):
        st.caption(str(generation["explanation"]))

    if generation.get("response_preview"):
        with st.expander("Response preview"):
            st.code(str(generation["response_preview"]), language="text")

    if generation.get("error"):
        st.error(f"Generation error: {generation['error']}")




# ==========================================================================
# PAGE  CHUNK MONITOR
# ==========================================================================

def _render_chunk_monitor(pipeline: RAGPipeline) -> None:
    render_html(
        render_page_header(
            "STORAGE & INDEX INSPECTOR",
            "Index & Chunk Inspector",
            "Live inspection of what is stored in Chroma Vector DB, BM25 Lexical Index, and source documents.",
        )
    )

    from src.ui.pages import render_chunk_explorer

    tab_hierarchy, tab_chroma, tab_bm25, tab_chunks = st.tabs([
        "📖 Book Hierarchy & Explorer",
        "🔍 Vector Chroma DB",
        "⚡ BM25 Lexical Index",
        "📁 Document Files",
    ])

    with tab_hierarchy:
        chunks = _load_chunks(pipeline)
        render_chunk_explorer(chunks, pipeline=pipeline)

    # ================================================================
    # TAB 1: VECTOR CHROMA DB
    # ================================================================
    with tab_chroma:
        render_html(render_section_title("Chroma Vector Store  Live Vectors & Chunks"))
        try:
            col_count = pipeline.vector_store.get_collection_count()
            dim = pipeline.vector_store.get_collection_embedding_dimension()
            col_name = pipeline.vector_store.collection_name
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not connect to Chroma: {exc}")
            col_count, dim, col_name = 0, None, "knowledge_base"

        c1, c2, c3 = st.columns(3)
        c1.metric("Indexed Chunks in Chroma", col_count)
        c2.metric("Embedding Dimension", dim or "N/A")
        c3.metric("Collection Name", col_name)

        if col_count == 0:
            st.info("Chroma vector store is currently empty. Ingest documents to populate.")
        else:
            try:
                chroma_data = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
                ids = chroma_data.get("ids") or []
                docs = chroma_data.get("documents") or []
                metas = chroma_data.get("metadatas") or []

                chroma_rows = []
                for i, cid in enumerate(ids):
                    meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                    doc = docs[i] if i < len(docs) and docs[i] else ""
                    p_start = meta.get("page_start") or meta.get("page_number")
                    p_end = meta.get("page_end") or p_start
                    p_str = f"{p_start}–{p_end}" if p_end and p_end != p_start else (str(p_start) if p_start else "-")
                    ch_num = meta.get("chapter_num") if meta.get("chapter_num") is not None else meta.get("chapter")
                    ch_str = f"Ch. {ch_num}" if ch_num and int(ch_num) > 0 else ("Intro/Front" if ch_num == 0 else "-")
                    
                    fn = meta.get("filename") or meta.get("book_name") or ""
                    raw_b = meta.get("book_title") or meta.get("book") or (Path(fn).stem if fn else "Unknown")
                    clean_b = str(raw_b).replace(" (1)", "").replace(".pdf", "").replace(".PDF", "").strip() or "Unknown"

                    chroma_rows.append({
                        "Chunk ID": meta.get("chunk_id") or cid,
                        "Book": clean_b,
                        "Chapter": ch_str,
                        "Chapter Num": int(ch_num) if ch_num is not None else 0,
                        "Chapter Title": meta.get("chapter_title") or "",
                        "Section": meta.get("section_heading") or meta.get("section_title") or "-",
                        "Pages": p_str,
                        "Tokens": meta.get("token_count", 0),
                        "Filename": fn,
                        "File Type": meta.get("file_type", "pdf"),
                        "Characters": len(doc),
                        "Preview": (doc[:130] + "...") if len(doc) > 130 else doc,
                        "_full_text": doc,
                        "_meta": meta,
                    })

                filter_c1, filter_c2, filter_c3 = st.columns([4, 3, 3])
                with filter_c1:
                    search_chroma = st.text_input("Filter Chroma entries", placeholder="Filter by text, chunk ID, section...", key="chroma_filter")
                with filter_c2:
                    chroma_books = sorted(list({r["Book"] for r in chroma_rows if r["Book"]}))
                    book_filter = st.selectbox("Filter Book", ["All Books"] + chroma_books, key="chroma_book_filter")
                with filter_c3:
                    if book_filter != "All Books":
                        avail_chs = sorted(list({r["Chapter"] for r in chroma_rows if r["Book"] == book_filter and r["Chapter"] != "-"}))
                    else:
                        avail_chs = sorted(list({r["Chapter"] for r in chroma_rows if r["Chapter"] != "-"}))
                    chapter_filter = st.selectbox("Filter Chapter", ["All Chapters"] + avail_chs, key="chroma_chapter_filter")

                filtered_chroma = chroma_rows
                if book_filter != "All Books":
                    filtered_chroma = [r for r in filtered_chroma if r["Book"] == book_filter]
                if chapter_filter != "All Chapters":
                    filtered_chroma = [r for r in filtered_chroma if r["Chapter"] == chapter_filter]
                if search_chroma:
                    s_low = search_chroma.strip().lower()
                    filtered_chroma = [
                        r for r in filtered_chroma
                        if s_low in r["Chunk ID"].lower()
                        or s_low in r["Filename"].lower()
                        or s_low in r["_full_text"].lower()
                        or s_low in str(r["Section"]).lower()
                        or s_low in str(r["Book"]).lower()
                    ]

                st.caption(f"Displaying **{len(filtered_chroma)}** of **{len(chroma_rows)}** vector chunks in Chroma across **{len(chroma_books)}** book(s)")
                _dataframe([
                    {
                        "Chunk ID": r["Chunk ID"],
                        "Book": r["Book"],
                        "Chapter": r["Chapter"],
                        "Chapter Title": r["Chapter Title"] or "-",
                        "Section": r["Section"],
                        "Pages": r["Pages"],
                        "Tokens": r["Tokens"],
                        "Preview": r["Preview"],
                    }
                    for r in filtered_chroma
                ])

                render_html(render_section_title("Chunk inspector"))
                inspect_ids = [r["Chunk ID"] for r in filtered_chroma]
                if inspect_ids:
                    inspect_id = st.selectbox(
                        "Select chunk",
                        inspect_ids,
                        format_func=lambda cid: next(
                            (
                                f"{cid} — {r['Book'][:28]} | {r['Section'][:24]} | pp. {r['Pages']}"
                                for r in filtered_chroma
                                if r["Chunk ID"] == cid
                            ),
                            cid,
                        ),
                        key="chroma_inspect_select",
                    )
                    selected_entry = next(
                        (r for r in filtered_chroma if r["Chunk ID"] == inspect_id),
                        None,
                    )
                    if selected_entry:
                        st.caption(
                            f"`{selected_entry['Chunk ID']}` · "
                            f"{selected_entry['Book']} · pp. {selected_entry['Pages']} · "
                            f"{selected_entry['Tokens']} tokens"
                        )
                        body = selected_entry["_full_text"] or ""
                        st.markdown("**Paragraph text**")
                        if body.strip():
                            st.text_area(
                                "Chroma chunk text",
                                value=body,
                                height=min(400, max(140, len(body) // 3)),
                                disabled=True,
                                label_visibility="collapsed",
                            )
                        else:
                            st.warning("This Chroma record has no stored document text.")
                        with st.expander("Stored metadata", expanded=False):
                            st.json(selected_entry["_meta"])
                else:
                    st.caption("No matching chunks to inspect.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error fetching Chroma chunks: {exc}")

    # ================================================================
    # TAB 2: BM25 LEXICAL INDEX
    # ================================================================
    with tab_bm25:
        render_html(render_section_title("BM25 Lexical Index  Vocabulary & Chunks"))
        bm25_instance = getattr(pipeline, "bm25", None)
        if bm25_instance is None or not hasattr(bm25_instance, "_chunks") or not bm25_instance._chunks:
            st.info("BM25 lexical index is currently not loaded or empty. Ingest documents or sync the knowledge base.")
        else:
            bm25_chunks = bm25_instance._chunks
            vocab = getattr(bm25_instance, "_vocab", {})
            avg_dl = getattr(bm25_instance, "_avg_doc_len", 0.0)

            b1, b2, b3 = st.columns(3)
            b1.metric("Indexed Chunks in BM25", len(bm25_chunks))
            b2.metric("Vocabulary (Unique Terms)", len(vocab))
            b3.metric("Avg Document Length", f"{avg_dl:.1f} tokens" if avg_dl else "N/A")

            bm25_rows = []
            for i, c in enumerate(bm25_chunks):
                cid = c.get("chunk_id", f"chunk_{i}")
                p_start = c.get("page_start") or c.get("page_number")
                p_end = c.get("page_end") or p_start
                p_str = f"{p_start}–{p_end}" if p_end and p_end != p_start else (str(p_start) if p_start else "-")
                ch_num = c.get("chapter_num") if c.get("chapter_num") is not None else c.get("chapter")
                ch_str = f"Ch. {ch_num}" if ch_num and int(ch_num) > 0 else ("Intro/Front" if ch_num == 0 else "-")
                txt = c.get("text", "")
                fn = c.get("filename") or c.get("book_name") or ""
                raw_b = c.get("book_title") or c.get("book") or (Path(fn).stem if fn else "Unknown")
                clean_b = str(raw_b).replace(" (1)", "").replace(".pdf", "").replace(".PDF", "").strip() or "Unknown"

                bm25_rows.append({
                    "Chunk ID": cid,
                    "Book": clean_b,
                    "Chapter": ch_str,
                    "Chapter Title": c.get("chapter_title") or "",
                    "Section": c.get("section_heading") or c.get("section_title") or "-",
                    "Pages": p_str,
                    "Tokens": c.get("token_count", 0),
                    "Filename": fn,
                    "Format": c.get("file_type", "pdf"),
                    "Characters": len(txt),
                    "Preview": (txt[:130] + "...") if len(txt) > 130 else txt,
                    "_full_text": txt,
                    "_raw": c,
                })

            bf1, bf2, bf3 = st.columns([4, 3, 3])
            with bf1:
                search_bm25 = st.text_input("Filter BM25 chunks", placeholder="Filter by term, heading, or filename...", key="bm25_filter")
            with bf2:
                bm25_books = sorted(list({r["Book"] for r in bm25_rows if r["Book"]}))
                b_book_filter = st.selectbox("Filter Book", ["All Books"] + bm25_books, key="bm25_book_filter")
            with bf3:
                if b_book_filter != "All Books":
                    b_avail_chs = sorted(list({r["Chapter"] for r in bm25_rows if r["Book"] == b_book_filter and r["Chapter"] != "-"}))
                else:
                    b_avail_chs = sorted(list({r["Chapter"] for r in bm25_rows if r["Chapter"] != "-"}))
                b_chapter_filter = st.selectbox("Filter Chapter", ["All Chapters"] + b_avail_chs, key="bm25_chapter_filter")

            filtered_bm25 = bm25_rows
            if b_book_filter != "All Books":
                filtered_bm25 = [r for r in filtered_bm25 if r["Book"] == b_book_filter]
            if b_chapter_filter != "All Chapters":
                filtered_bm25 = [r for r in filtered_bm25 if r["Chapter"] == b_chapter_filter]
            if search_bm25:
                s_low = search_bm25.strip().lower()
                filtered_bm25 = [
                    r for r in filtered_bm25
                    if s_low in r["Chunk ID"].lower()
                    or s_low in r["Filename"].lower()
                    or s_low in r["_full_text"].lower()
                    or s_low in str(r["Section"]).lower()
                    or s_low in str(r["Book"]).lower()
                ]

            st.caption(f"Displaying **{len(filtered_bm25)}** of **{len(bm25_rows)}** chunks in BM25 index across **{len(bm25_books)}** book(s)")
            _dataframe([
                {
                    "Chunk ID": r["Chunk ID"],
                    "Book": r["Book"],
                    "Chapter": r["Chapter"],
                    "Chapter Title": r["Chapter Title"] or "-",
                    "Section": r["Section"],
                    "Pages": r["Pages"],
                    "Tokens": r["Tokens"],
                    "Preview": r["Preview"],
                }
                for r in filtered_bm25
            ])

            render_html(render_section_title("Chunk inspector"))
            inspect_b_ids = [r["Chunk ID"] for r in filtered_bm25]
            if inspect_b_ids:
                inspect_b_id = st.selectbox(
                    "Select chunk",
                    inspect_b_ids,
                    format_func=lambda cid: next(
                        (
                            f"{cid} — {r['Book'][:28]} | {r['Section'][:24]} | pp. {r['Pages']}"
                            for r in filtered_bm25
                            if r["Chunk ID"] == cid
                        ),
                        cid,
                    ),
                    key="bm25_inspect_select",
                )
                sel_b = next((r for r in filtered_bm25 if r["Chunk ID"] == inspect_b_id), None)
                if sel_b:
                    st.caption(
                        f"`{sel_b['Chunk ID']}` · "
                        f"{sel_b['Book']} · pp. {sel_b['Pages']} · "
                        f"{sel_b['Tokens']} tokens"
                    )
                    body = sel_b["_full_text"] or ""
                    st.markdown("**Paragraph text**")
                    if body.strip():
                        st.text_area(
                            "BM25 chunk text",
                            value=body,
                            height=min(400, max(140, len(body) // 3)),
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.warning("This BM25 record has no chunk text.")
                    with st.expander("Stored metadata", expanded=False):
                        st.json({k: v for k, v in sel_b["_raw"].items() if k != "text"})
            else:
                st.caption("No matching BM25 chunks to inspect.")

    # ================================================================
    # TAB 3: DOCUMENT CHUNKS (SOURCE FILES)
    # ================================================================
    with tab_chunks:
        files = list_knowledge_files(Path(KNOWLEDGE_BASE_DIR))

        if not files:
            st.info("No supported files were found in the knowledge base.")
            return

        chunks = _load_chunks(pipeline)

        if not chunks:
            st.warning(
                "Chunks could not be computed for preview. The index itself is "
                "unaffected."
            )
            return

        size, overlap = _chunk_config(pipeline)

        # ----------------------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------------------
        render_html(render_section_title("Knowledge base overview"))

        _dataframe([
                {
                    "Document": path.name,
                    "Chunks": len(
                        [c for c in chunks if c.get("filename") == path.name]
                    ),
                    "Characters": _document_characters(path),
                }
                for path in files
            ])

        # ----------------------------------------------------------------
        # DOCUMENT SELECTOR
        # ----------------------------------------------------------------
        render_html(render_section_title("Document"))

        selected_name = st.selectbox(
            "Select a document",
            [path.name for path in files],
        )

        selected_path = next(
            (path for path in files if path.name == selected_name),
            None,
        )

        selected_chunks = [
            c for c in chunks if c.get("filename") == selected_name
        ]

        if not selected_chunks:
            st.info("This document produced no non-empty chunks.")
            return

        file_size = None

        if selected_path is not None:
            try:
                file_size = selected_path.stat().st_size
            except Exception:  # noqa: BLE001
                file_size = None

        characters = (
            _document_characters(selected_path)
            if selected_path is not None
            else None
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "File size",
            f"{file_size:,} B" if file_size is not None else "N/A",
        )
        col2.metric(
            "Characters",
            f"{characters:,}" if characters is not None else "N/A",
        )
        col3.metric("Chunks", len(selected_chunks))
        col4.metric(
            "Target / max tokens",
            f"{size} / {overlap}" if size is not None else "N/A",
        )

        # ----------------------------------------------------------------
        # CHUNK BROWSER
        # ----------------------------------------------------------------
        render_html(render_section_title("Chunk browser"))

        chunk_number = st.number_input(
            "Chunk",
            min_value=1,
            max_value=len(selected_chunks),
            value=1,
            step=1,
        )

        index = int(chunk_number) - 1
        chunk = selected_chunks[index]

        render_html(render_chunk_map(len(selected_chunks), index))

        st.caption(
            f"Each cell is one chunk of {selected_name}, in document order. "
            "The highlighted cell is the chunk shown below."
        )

        text = chunk.get("text", "")

        render_html(
            render_kv_rows(
                [
                    ("Chunk ID", _na(chunk.get("chunk_id"))),
                    ("Filename", _na(chunk.get("filename"))),
                    ("Characters", len(text)),
                    ("Section heading", _na(chunk.get("section_heading"))),
                    ("Chunk index", _na(chunk.get("chunk_index"))),
                    ("Chunks in section", _na(chunk.get("total_section_chunks"))),
                    ("Category", _na(chunk.get("category"))),
                ]
            )
        )

        st.code(text, language="text")

        with st.expander("Full chunk metadata"):
            st.json(
                {
                    "source_path": chunk.get("source_path", ""),
                    "section_heading": chunk.get("section_heading", ""),
                    "section_path": chunk.get("section_path", []),
                    "section_level": chunk.get("section_level", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_section_chunks": chunk.get("total_section_chunks", 0),
                    "category": chunk.get("category", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                }
            )

        # ----------------------------------------------------------------
        # OVERLAP EXPLANATION
        # ----------------------------------------------------------------
        render_html(render_section_title("How chunks are sized"))

        para_overlap = get_chunking_config().get("paragraph_overlap_tokens")
        st.write(
            f"Chunks are packed by **token count**, not fixed character width. "
            f"The chunker aims for about **{size}** tokens per chunk and hard-limits "
            f"each chunk to **{overlap}** tokens. Paragraphs are merged until the target "
            f"is reached; oversized paragraphs are split with up to "
            f"**{para_overlap}** tokens of carry-over between split pieces."
        )

        with st.expander(f"All chunks in {selected_name}"):
            for position, item in enumerate(selected_chunks):
                st.markdown(
                    f"**[ Chunk {position} ]**  "
                    f"`{item.get('chunk_id', '')}`  "
                    f"{len(item.get('text', ''))} characters"
                )
                st.code((item.get("text") or "")[:600], language="text")




# ============================================================================
# PAGE  SYSTEM ARCHITECTURE
# ============================================================================

def _render_architecture_page(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> None:
    """
    Static architecture visualization of the real system.

    This page performs no retrieval and calls no model. The technical
    labels on the storage and model nodes come from live configuration,
    so the diagram cannot drift away from what is actually deployed.
    """
    render_html(
        render_page_header(
            "SYSTEM ARCHITECTURE",
            "Astronomy RAG System",
            "How knowledge enters the system, and how a question becomes "
            "a grounded answer.",
        )
    )

    render_html(render_simulation_tag("Architecture diagram  illustrative"))

    st.caption(
        "This diagram represents the components that actually exist in "
        "this project. It is not a live telemetry view."
    )

    render_html(render_arch_legend())

    st.write("")

    embedding_model = status.get("embedding_model") or "configured model"
    llm_model = _llm_model(status) or "configured model"

    chunks_label = (
        f"{collection_count} chunks"
        if collection_count is not None
        else "chunk count N/A"
    )

    size, overlap = _chunk_config(pipeline)

    chunk_tech = (
        f"target {size} / max {overlap} tokens"
        if size is not None and overlap is not None
        else "token chunking"
    )

    # ----------------------------------------------------------------
    # KNOWLEDGE PLANE
    # ----------------------------------------------------------------
    render_html(
        render_arch_plane_open(
            "Knowledge Plane  Ingestion",
            "Documents are loaded, cleaned and chunked once, then indexed "
            "twice: as vectors and as terms.",
        )
    )

    row = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1])

    with row[0]:
        render_html(
            render_arch_node(
                "",
                "Knowledge Sources",
                f"{status.get('txt_files_found', 'N/A')} TXT / MD files",
                "storage",
            )
        )

    with row[1]:
        render_html(render_arch_arrow(""))

    with row[2]:
        render_html(
            render_arch_node(
                "", "Document Loader", "read + detect", "processing"
            )
        )

    with row[3]:
        render_html(render_arch_arrow(""))

    with row[4]:
        render_html(
            render_arch_node(
                "", "Text Processing", "normalize", "processing"
            )
        )

    with row[5]:
        render_html(render_arch_arrow(""))

    with row[6]:
        render_html(
            render_arch_node("", "Chunking", chunk_tech, "processing")
        )

    st.caption("Chunking fans out into two independent indexes:")

    split = st.columns(2)

    with split[0]:
        render_html(
            render_arch_node(
                "", "Embedding Model", embedding_model, "processing"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Chroma Vector Store", chunks_label, "storage"
            )
        )

    with split[1]:
        render_html(
            render_arch_node(
                "", "BM25", "term weighting", "retrieval"
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "",
                "BM25 Index",
                "persistent lexical index",
                "storage",
            )
        )

    # ----------------------------------------------------------------
    # CONVERGENCE
    # ----------------------------------------------------------------
    bridge = st.columns(3)

    with bridge[1]:
        render_html(render_arch_arrow(""))
        st.caption("Both indexes serve the query plane below")
        render_html(render_arch_arrow(""))

    # ----------------------------------------------------------------
    # QUERY PLANE
    # ----------------------------------------------------------------
    render_html(
        render_arch_plane_open(
            "Query Plane  Retrieval & Generation",
            "A question is prepared once, searched by one or both "
            "retrievers depending on the selected mode, then gated, "
            "trimmed and answered.",
        )
    )

    head = st.columns(3)

    with head[1]:
        render_html(
            render_arch_node("", "User Query", "Mission Control", "processing")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "",
                "Query Preparation",
                "normalize / aliases",
                "processing",
            )
        )

    st.caption(
        "Vector mode uses dense search only. BM25 mode uses lexical "
        "search only. Hybrid mode runs both and fuses them."
    )

    search = st.columns(2)

    with search[0]:
        render_html(
            render_arch_node(
                "", "Dense Search", "Chroma nearest vectors", "retrieval"
            )
        )

    with search[1]:
        render_html(
            render_arch_node(
                "", "BM25 Search", "lexical match", "retrieval"
            )
        )

    tail = st.columns(3)

    with tail[1]:
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "",
                "RRF Fusion",
                "Hybrid mode only",
                "retrieval",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "",
                "Evidence Gate",
                "relevance / support",
                "decision",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "",
                "Context Selection",
                "chunk + char budget",
                "processing",
            )
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node("", "Generation", llm_model, "generation")
        )
        render_html(render_arch_arrow())
        render_html(
            render_arch_node(
                "", "Answer", "response + sources", "generation"
            )
        )

    # ----------------------------------------------------------------
    # COMPONENT REFERENCE
    # ----------------------------------------------------------------
    render_html(render_section_title("Component reference"))

    left, right = st.columns(2)

    midpoint = (len(ARCH_DESCRIPTIONS) + 1) // 2

    for column, items in (
        (left, ARCH_DESCRIPTIONS[:midpoint]),
        (right, ARCH_DESCRIPTIONS[midpoint:]),
    ):
        with column:
            for name, description in items:
                with st.expander(name):
                    st.write(description)

    st.info(
        "Not present in this system: cross-encoder reranking is "
        "intentionally disabled, and there are no agents, external "
        "services or additional databases beyond those shown above."
    )

    with st.expander("Text version of the full flow"):
        st.code(INGESTION_DIAGRAM, language="text")
        st.code(RETRIEVAL_DIAGRAM, language="text")


# ============================================================================
# SIDEBAR
# ============================================================================

def _render_sidebar(
    pipeline: RAGPipeline,
    status: dict,
    collection_count,
) -> str:
    """
    Sidebar: brand, grouped navigation, system status and index controls.

    Navigation lives here and only here. The native Streamlit collapse
    control is left untouched so the sidebar can always be reopened.
    """
    with st.sidebar:
        render_html(
            render_panel_card(
                " ASTRONOMY OBSERVATORY",
                render_kv_rows([("Knowledge Core", "ONLINE")]),
            )
        )

        # ------------------------------------------------------------
        # NAVIGATION
        # ------------------------------------------------------------
        current = st.session_state.page

        if current not in PAGES:
            current = PAGE_MISSION

        for group_label, items in NAV_GROUPS:
            render_html(render_nav_group_label(group_label))

            for page_name, icon in items:
                active = page_name == current

                marker = "" if active else ""

                if st.button(
                    _sidebar_nav_label(marker, page_name, icon),
                    key=f"nav_{page_name}",
                    type="primary" if active else "secondary",
                    **WIDE,
                ):
                    if page_name != current:
                        st.session_state.page = page_name
                        st.rerun()

        render_html(render_sidebar_divider())

        # ------------------------------------------------------------
        # SYSTEM STATUS
        # ------------------------------------------------------------
        status_html = render_stat_grid(
            [
                (
                    str(collection_count)
                    if collection_count is not None
                    else "N/A",
                    "Indexed chunks",
                ),
                (str(status.get("txt_files_found", "N/A")), "Documents"),
            ]
        )

        status_html += render_status_row(
            "Vector Index", bool(status.get("index_available"))
        )
        status_html += render_status_row(
            "BM25 Index", bool(status.get("bm25_available"))
        )
        status_html += render_status_row(
            "Embeddings", bool(status.get("embedding_model"))
        )
        status_html += render_status_row(
            "LLM", bool(_llm_model(status))
        )

        render_html(render_panel_card("SYSTEM STATUS", status_html))

        meta_html = render_kv_rows(
            [
                ("Embedding Model", _na(status.get("embedding_model"))),
                ("LLM Model", _na(_llm_model(status))),
                ("Vector Store", _na(status.get("vectorstore_path"))),
            ]
        )

        render_html(render_panel_card("CONFIGURATION", meta_html))

        render_html(render_sidebar_divider())

        # ------------------------------------------------------------
        # INDEX CONTROL
        # ------------------------------------------------------------
        render_html(render_nav_group_label("Index Control"))

        if st.button("Refresh Status", key="ctl_refresh", **WIDE):
            st.session_state.force_reload = True
            st.session_state.pop("pipeline_status", None)
            st.session_state.pop("_status_ts", None)
            st.session_state.pop("_kb_snapshot", None)
            st.rerun()

        if st.button("Sync Knowledge Base", key="ctl_sync", **WIDE):
            with st.spinner("Embedding any missing chunks "):
                try:
                    result = pipeline.ingest_documents()

                    st.session_state.force_reload = True
                    st.session_state.pop("pipeline_status", None)
                    st.session_state.pop("_status_ts", None)
                    st.session_state.pop("_kb_snapshot", None)
                    st.session_state.ingestion_notice = {
                        "type": "success",
                        "message": (
                            f"Index updated: {result['chunks_indexed']} new "
                            f"chunks ({result.get('chunks_skipped', 0)} "
                            f"skipped, {result['collection_count']} total) "
                            f"from {result['files_processed']} files."
                        ),
                    }
                    st.rerun()

                except Exception as exc:  # noqa: BLE001
                    _service_error("INGESTION SERVICE UNAVAILABLE", exc)

        st.caption(
            "Incremental  existing embeddings are preserved, never "
            "rebuilt from scratch."
        )

        if st.button("Clear Chat", key="ctl_clear", **WIDE):
            st.session_state.pop("chat_history", None)
            st.session_state.last_response = None
            st.rerun()

    return st.session_state.page


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Astronomy Observatory  Mission Control",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_html(inject_global_css())

    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # SESSION DEFAULTS
    # ----------------------------------------------------------------
    st.session_state.setdefault("page", PAGE_MISSION)
    st.session_state.setdefault("retrieval_mode", "hybrid")
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("last_ingestion", None)

    pipeline = _pipeline()
    status = _pipeline_status(pipeline)
    collection_count = _collection_count(pipeline)

    page = _render_sidebar(pipeline, status, collection_count)

    _show_ingestion_notice()

    if st.session_state.get("status_error"):
        _service_error(
            "SYSTEM STATUS UNAVAILABLE",
            st.session_state["status_error"],
        )

    # Page dispatch. Only Mission Control can trigger retrieval or
    # generation; every other page is read-only.
    if page == PAGE_LAB:
        _render_retrieval_lab(st.session_state.last_response)
        return

    if page == PAGE_OBSERVATION:
        _render_observation_page(pipeline)
        return

    if page == PAGE_DASHBOARD:
        _render_dashboard_page(pipeline)
        return

    if page == PAGE_CHAT_HISTORY:
        _render_chat_history_page(pipeline)
        return

    if page == PAGE_COMPARISON:
        _render_comparison_page(pipeline)
        return

    if page == PAGE_INGEST:
        _render_ingestion_page(pipeline, status, collection_count)
        return

    if page == PAGE_CHUNKS:
        _render_chunk_monitor(pipeline)
        return

    if page == PAGE_ARCH:
        _render_architecture_page(pipeline, status, collection_count)
        return

    _render_mission_control(pipeline, status, collection_count)


def _short_topic(query: str, index: int) -> str:
    words = [w for w in query.replace("?", "").split() if w.lower() not in {"what", "is", "are", "how", "does", "the", "and"}]
    return f"Q{index:02d}  {' '.join(words[:3]) or 'query'}"


def _render_observation_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Observation & Evaluation", "Persistent system insight dashboard. Live scores are heuristics, not accuracy."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet. Run a query on the Ask page to record an observation.")
        return

    # ── Tab layout ──────────────────────────────────────────────────────────
    tab_overview, tab_deepeval, tab_citations, tab_langfuse, tab_auto, tab_drill = st.tabs([
        "📊 Overview",
        "🔬 DeepEval Metrics",
        "📑 Citation Analysis",
        "🔭 Langfuse Traces",
        "💡 Automatic Observations",
        "🔍 Query Drill-Down",
    ])

    # ── Helper: short label per row ─────────────────────────────────────────
    def _row_label(row: dict, index: int) -> str:
        return _short_topic(row.get("raw_query", ""), index)

    row_labels = [_row_label(r, i) for i, r in enumerate(rows, 1)]

    # ========================================================================
    # TAB 1  OVERVIEW
    # ========================================================================
    with tab_overview:
        latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
        median_lat = sorted(latencies)[len(latencies) // 2] if latencies else 0

        # Retrieval strength from evaluation criteria
        strengths = []
        for r in rows:
            s = (r.get("evaluation") or {}).get("criteria", {}).get("retrieval_strength", {}).get("score")
            if isinstance(s, (int, float)):
                strengths.append(s)
        avg_strength = sum(strengths) / len(strengths) if strengths else None

        # Langfuse enabled?
        has_langfuse = any(
            r.get("trace", {}).get("langfuse_trace_id")
            or (r.get("trace") or {}).get("observation", {}).get("langfuse_trace_id")
            for r in rows
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", len(rows))
        c2.metric("Median Latency", f"{median_lat:.0f} ms" if median_lat else "—")
        c3.metric("Avg Retrieval Strength", f"{avg_strength:.1f}" if avg_strength is not None else "—")
        c4.metric("Langfuse", "✅ Enabled" if has_langfuse else "⬜ Not traced")

        st.markdown("---")
        st.markdown("#### Latency per Query")
        latency_data = {label: r.get("latency_ms", 0) for label, r in zip(row_labels, rows)}
        st.bar_chart(latency_data)

        st.markdown("#### Retrieval Method Distribution")
        mode_counts: dict[str, int] = {}
        for r in rows:
            mode = r.get("retrieval_mode") or (r.get("trace") or {}).get("retrieval", {}).get("mode") or "unknown"
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if mode_counts:
            st.bar_chart(mode_counts)
        else:
            st.caption("No retrieval mode information recorded.")

        st.markdown("#### Top 5 Slowest Queries")
        slowest = sorted(rows, key=lambda r: r.get("latency_ms", 0), reverse=True)[:5]
        slow_table = [
            {
                "Query": r.get("raw_query", "")[:80],
                "Latency (ms)": r.get("latency_ms", 0),
                "Mode": r.get("retrieval_mode", "—"),
                "Timestamp": r.get("timestamp", "")[:19],
            }
            for r in slowest
        ]
        st.dataframe(slow_table, hide_index=True, use_container_width=True)

    # ========================================================================
    # TAB 2  DEEPEVAL METRICS
    # ========================================================================
    with tab_deepeval:
        # Metric thresholds and descriptions
        _DE_THRESHOLDS: dict[str, float] = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "contextual_relevancy": 0.6,
            "contextual_precision": 0.6,
            "contextual_recall": 0.6,
            "answer_correctness": 0.7,
            "answer_completeness": 0.7,
            "citation_correctness": 0.7,
            "citation_completeness": 0.7,
            "groundedness": 0.7,
            "noise_sensitivity": 0.7,
        }
        _DE_HOW: dict[str, str] = {
            "faithfulness": "LLM checks each claim in the answer against retrieved context (top 3). Score: 0=contradicts context, 1=fully grounded.",
            "answer_relevancy": "LLM checks whether the answer directly responds to the query. Score: 0=completely irrelevant, 1=fully responsive.",
            "contextual_relevancy": "LLM inspects each retrieved passage against the query topic and rates the proportion that are on-point. Score: 0=all off-topic, 1=all relevant.",
            "contextual_precision": "LLM determines which chunks are relevant then checks if they appear before irrelevant ones (ranking quality). Score: 0=relevant chunks at bottom, 1=perfectly ranked.",
            "contextual_recall": "LLM checks each factual statement in the answer against retrieved passages to estimate coverage. Score: 0=not supported, 1=fully covered.",
            "answer_correctness": "LLM cross-checks specific claims in the answer against retrieved passages for factual accuracy. Score: 0=factually wrong, 1=fully correct.",
            "answer_completeness": "LLM decomposes the query into components and checks if each part is addressed. Score: 0=major parts unanswered, 1=fully addressed.",
            "citation_correctness": "LLM cross-references each [N] citation against chunk N to verify support. Score: 0=all citations misleading, 1=every citation accurate.",
            "citation_completeness": "LLM counts major factual claims and checks what proportion carry a bracketed citation. Score: 0=no claims cited, 1=every claim cited.",
            "groundedness": "LLM reads the answer and flags statements not found or inferable from context (hallucination check). Score: 0=full of hallucinations, 1=fully grounded.",
            "noise_sensitivity": "LLM evaluates whether irrelevant retrieved context negatively affected the answer. Score: 0=degraded answer, 1=unaffected by noise.",
        }

        # Categories for grouping in the UI
        _DE_CATEGORIES = {
            "Retrieval": ["contextual_relevancy", "contextual_precision", "contextual_recall", "noise_sensitivity"],
            "Answer Quality": ["faithfulness", "answer_relevancy", "answer_correctness", "answer_completeness", "groundedness"],
            "Citation": ["citation_correctness", "citation_completeness"],
        }


        # Collect deepeval data from observations
        # deepeval is stored as response["deepeval"]["by_retrieval_mode"][mode] = {status, scores}
        # In observation records it lands in evaluation or directly in the trace
        de_rows = []
        for row in rows:
            # Try multiple paths where deepeval scores might have been persisted
            deepeval_data = (
                (row.get("trace") or {}).get("deepeval")
                or row.get("deepeval")
                or {}
            )
            by_mode = deepeval_data.get("by_retrieval_mode") or {}
            # Flatten: pick first mode with scores
            scores_dict: dict[str, dict] = {}
            mode_used = row.get("retrieval_mode", "hybrid")
            for mode_key in [mode_used, *by_mode.keys()]:
                entry = by_mode.get(mode_key, {})
                if entry.get("scores"):
                    scores_dict = entry["scores"]
                    break
            if scores_dict:
                de_rows.append({
                    "label": _row_label(row, rows.index(row) + 1),
                    "query": row.get("raw_query", ""),
                    "scores": scores_dict,
                })

        if not de_rows:
            st.info(
                "No DeepEval results found in recorded observations.\n\n"
                "To enable DeepEval evaluation, set `ENABLE_DEEPEVAL=true` in your `.env` file "
                "and restart the app. Scores will appear here after your next query."
            )
        else:
            # Aggregate pass/fail
            total_pass = total_fail = 0
            for entry in de_rows:
                for metric, threshold in _DE_THRESHOLDS.items():
                    score_data = entry["scores"].get(metric)
                    if score_data is not None:
                        score = score_data.get("score", 0) if isinstance(score_data, dict) else float(score_data)
                        if score >= threshold:
                            total_pass += 1
                        else:
                            total_fail += 1
            pc1, pc2 = st.columns(2)
            pc1.metric("✅ Total Passes", total_pass)
            pc2.metric("❌ Total Fails", total_fail)
            st.markdown("---")

            # Per-observation tables
            query_choice = st.selectbox(
                "Select query to inspect",
                options=[e["label"] for e in de_rows],
                key="deepeval_query_select",
            )
            selected_de = next((e for e in de_rows if e["label"] == query_choice), de_rows[0])
            st.caption(f"Query: *{selected_de['query'][:120]}*")

            # Add Performance explicitly to the UI
            st.markdown("### Performance")
            perf_rows = []
            # Performance isn't an LLM metric, but we want to show it here.
            # Get the query's base observation row.
            base_row = next((r for r in rows if _row_label(r, rows.index(r)+1) == query_choice), None)
            if base_row:
                ret_ms = base_row.get("latency_ms") or 0
                llm_ms = (base_row.get("trace") or {}).get("llm_ms") or base_row.get("llm_ms") or 0
                perf_rows.append({"Metric": "Retrieval Latency", "Value": f"{ret_ms} ms", "Description": "Time taken to retrieve chunks."})
                perf_rows.append({"Metric": "End-to-End Latency", "Value": f"{ret_ms + llm_ms} ms", "Description": "Total query-response time."})
                st.dataframe(perf_rows, hide_index=True, use_container_width=True)

            for category, metric_list in _DE_CATEGORIES.items():
                st.markdown(f"### {category}")
                table_rows = []
                for metric_name in metric_list:
                    threshold = _DE_THRESHOLDS.get(metric_name, 0.5)
                    score_data = selected_de["scores"].get(metric_name)
                    if score_data is None:
                        continue
                    if isinstance(score_data, dict):
                        score = score_data.get("score", 0)
                        reason = score_data.get("reason", "")
                    else:
                        score = float(score_data)
                        reason = ""
                    passed = "✅ Pass" if score >= threshold else "❌ Fail"
                    table_rows.append({
                        "Metric": metric_name.replace("_", " ").title(),
                        "Score": round(score, 3),
                        "Threshold": threshold,
                        "Pass/Fail": passed,
                        "Reason": reason[:120],
                        "How Calculated": _DE_HOW.get(metric_name, ""),
                    })
                if table_rows:
                    st.dataframe(table_rows, hide_index=True, use_container_width=True)
                else:
                    st.caption(f"No metrics recorded for {category}.")

    # ========================================================================
    # TAB 3  CITATION ANALYSIS
    # ========================================================================
    with tab_citations:
        st.markdown("#### Citation Metrics per Query")

        cit_labels: list[str] = []
        cit_coverage_scores: list[float] = []

        for i, row in enumerate(rows, 1):
            ae = row.get("answer_evaluation") or {}
            label = _row_label(row, i)
            cov = ae.get("citation_coverage") or {}
            grounding = ae.get("evidence_claim_grounding") or {}
            correctness = ae.get("citation_correctness") or {}

            cov_score = cov.get("score") if isinstance(cov, dict) else cov
            grounding_score = grounding.get("score") if isinstance(grounding, dict) else grounding
            correct_score = correctness.get("score") if isinstance(correctness, dict) else correctness

            if cov_score is not None:
                cit_labels.append(label)
                cit_coverage_scores.append(float(cov_score))

            with st.expander(f"{label}", expanded=(i == 1)):
                st.caption(f"*{row.get('raw_query', '')[:100]}*")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric(
                    "Citation Coverage",
                    f"{float(cov_score):.2f}" if cov_score is not None else "—",
                )
                cc2.metric(
                    "Evidence Grounding",
                    f"{float(grounding_score):.2f}" if grounding_score is not None else "—",
                )
                cc3.metric(
                    "Citation Correctness",
                    f"{float(correct_score):.2f}" if correct_score is not None else "—",
                )
                if isinstance(cov, dict) and cov.get("reason"):
                    st.caption(f"Coverage reason: {cov['reason']}")
                if isinstance(grounding, dict) and grounding.get("reason"):
                    st.caption(f"Grounding reason: {grounding['reason']}")
                if isinstance(correctness, dict) and correctness.get("reason"):
                    st.caption(f"Correctness reason: {correctness['reason']}")

                # Supported vs unsupported claims
                citation_map = row.get("citation_map") or []
                if citation_map:
                    supported = [c for c in citation_map if c.get("status") == "supported" or c.get("direct_answer_support")]
                    unsupported = [c for c in citation_map if c.get("status") not in ("supported",) and not c.get("direct_answer_support")]
                    st.caption(f"Claims: {len(supported)} supported, {len(unsupported)} unsupported (from recorded citation map)")

        if cit_coverage_scores:
            st.markdown("---")
            st.markdown("#### Citation Coverage Scores Across Queries")
            st.bar_chart(dict(zip(cit_labels, cit_coverage_scores)))
        elif not rows:
            st.info("No observations recorded yet.")
        else:
            st.info("No citation analysis data found in recorded observations.")

    # ========================================================================
    # TAB 4  LANGFUSE TRACES
    # ========================================================================
    with tab_langfuse:
        # Check if any trace IDs are present
        trace_rows = []
        for i, row in enumerate(rows, 1):
            tid = (
                (row.get("trace") or {}).get("langfuse_trace_id")
                or (row.get("trace") or {}).get("observation", {}).get("langfuse_trace_id")
            )
            if tid:
                trace_rows.append({"row": row, "trace_id": tid, "label": _row_label(row, i)})

        if not trace_rows:
            st.info(
                "No Langfuse trace IDs found in recorded observations.\n\n"
                "To enable Langfuse tracing, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` "
                "in your `.env` file and restart the app."
            )
        else:
            st.markdown("#### Langfuse Trace Links")
            for entry in trace_rows:
                st.markdown(f"- {entry['label']}: [View trace](https://cloud.langfuse.com/project/cmuccsmd900y8ad0cu6kxobc9/traces/{entry['trace_id']})")

        st.markdown("---")
        st.markdown("#### Latency Breakdown per Query")
        latency_rows = []
        for i, row in enumerate(rows, 1):
            retrieval = (row.get("trace") or {}).get("retrieval") or {}
            timings = retrieval.get("timings_ms") or {}
            ret_ms = retrieval.get("retrieval_ms") or retrieval.get("dense_ms") or timings.get("dense_ms")
            llm_ms = (row.get("trace") or {}).get("llm_ms") or timings.get("llm_ms")
            latency_rows.append({
                "Query": _row_label(row, i),
                "Retrieval (ms)": round(float(ret_ms), 1) if isinstance(ret_ms, (int, float)) else None,
                "LLM (ms)": round(float(llm_ms), 1) if isinstance(llm_ms, (int, float)) else None,
                "Total (ms)": row.get("latency_ms"),
            })
        st.dataframe(latency_rows, hide_index=True, use_container_width=True)

        # Stacked bar chart: retrieval_ms vs llm_ms over time
        ret_series = {_row_label(r, i): (r.get("trace") or {}).get("retrieval", {}).get("retrieval_ms") or 0
                      for i, r in enumerate(rows, 1)}
        llm_series = {_row_label(r, i): (r.get("trace") or {}).get("llm_ms") or 0
                      for i, r in enumerate(rows, 1)}
        has_data = any(v > 0 for v in list(ret_series.values()) + list(llm_series.values()))
        if has_data:
            st.markdown("#### Retrieval vs LLM Latency Over Time")
            import pandas as _pd
            chart_df = _pd.DataFrame({
                "Retrieval ms": list(ret_series.values()),
                "LLM ms": list(llm_series.values()),
            }, index=list(ret_series.keys()))
            st.bar_chart(chart_df)
        else:
            st.caption("Stage-level timing not available — only end-to-end latency is recorded.")

    # ========================================================================
    # TAB 5  AUTOMATIC OBSERVATIONS
    # ========================================================================
    with tab_auto:
        st.markdown("#### Automatically Generated Observations")
        st.caption("Derived from recorded observations — no values are hardcoded or estimated.")

        latencies_all = [r.get("latency_ms", 0) for r in rows]
        median_all = sorted(latencies_all)[len(latencies_all) // 2] if latencies_all else 0

        # Best retrieval method
        mode_strengths: dict[str, list[float]] = {}
        for r in rows:
            mode = r.get("retrieval_mode") or "unknown"
            s = (r.get("evaluation") or {}).get("criteria", {}).get("retrieval_strength", {}).get("score")
            if isinstance(s, (int, float)):
                mode_strengths.setdefault(mode, []).append(s)
        if mode_strengths:
            best_mode = max(mode_strengths, key=lambda m: sum(mode_strengths[m]) / len(mode_strengths[m]))
            best_avg = sum(mode_strengths[best_mode]) / len(mode_strengths[best_mode])
            with st.expander(f"🏆 Best Retrieval Method: **{best_mode}** (avg strength {best_avg:.1f})", expanded=True):
                for mode, vals in mode_strengths.items():
                    st.caption(f"• {mode}: avg {sum(vals)/len(vals):.1f} over {len(vals)} queries")

        # Weak queries
        weak = [
            r for r in rows
            if r.get("latency_ms", 0) > 5000
            or (r.get("evaluation") or {}).get("criteria", {}).get("retrieval_strength", {}).get("score", 100) < 40
        ]
        with st.expander(f"⚠️ Weak Queries ({len(weak)} found)", expanded=bool(weak)):
            if weak:
                for r in weak:
                    latency = r.get("latency_ms", 0)
                    strength = (r.get("evaluation") or {}).get("criteria", {}).get("retrieval_strength", {}).get("score")
                    reason = []
                    if latency > 5000:
                        reason.append(f"latency {latency:.0f} ms")
                    if isinstance(strength, (int, float)) and strength < 40:
                        reason.append(f"retrieval strength {strength:.1f}")
                    st.caption(f"• *{r.get('raw_query', '')[:80]}* — {', '.join(reason)}")
            else:
                st.caption("No weak queries detected.")

        # Citation issues
        citation_issues = []
        for r in rows:
            ae = r.get("answer_evaluation") or {}
            cov = ae.get("citation_coverage") or {}
            score = cov.get("score") if isinstance(cov, dict) else cov
            if isinstance(score, (int, float)) and score < 0.5:
                citation_issues.append((r, score))
        with st.expander(f"📋 Citation Issues ({len(citation_issues)} queries with coverage < 0.5)", expanded=bool(citation_issues)):
            if citation_issues:
                for r, score in citation_issues:
                    st.caption(f"• *{r.get('raw_query', '')[:80]}* — citation coverage {score:.2f}")
            else:
                st.caption("No citation coverage issues detected.")

        # Hallucination risks (from deepeval faithfulness)
        hallucination_risks = []
        for i, row in enumerate(rows, 1):
            deepeval_data = (row.get("trace") or {}).get("deepeval") or row.get("deepeval") or {}
            by_mode = deepeval_data.get("by_retrieval_mode") or {}
            for entry in by_mode.values():
                faith_data = (entry.get("scores") or {}).get("faithfulness") or {}
                faith_score = faith_data.get("score") if isinstance(faith_data, dict) else None
                if isinstance(faith_score, (int, float)) and faith_score < 0.6:
                    hallucination_risks.append((row, faith_score))
                    break
        with st.expander(f"🚨 Hallucination Risks ({len(hallucination_risks)} queries with faithfulness < 0.6)", expanded=bool(hallucination_risks)):
            if hallucination_risks:
                for r, score in hallucination_risks:
                    st.caption(f"• *{r.get('raw_query', '')[:80]}* — faithfulness {score:.2f}")
            else:
                st.caption("No hallucination risks detected (or DeepEval not yet enabled).")

        # Latency outliers (> 2× median)
        if median_all > 0:
            outliers = [(r, r.get("latency_ms", 0)) for r in rows if r.get("latency_ms", 0) > 2 * median_all]
        else:
            outliers = []
        with st.expander(f"⏱ Latency Outliers ({len(outliers)} queries > 2× median {median_all:.0f} ms)", expanded=bool(outliers)):
            if outliers:
                for r, lat in outliers:
                    st.caption(f"• *{r.get('raw_query', '')[:80]}* — {lat:.0f} ms")
            else:
                st.caption("No latency outliers detected.")

    # ========================================================================
    # TAB 6  QUERY DRILL-DOWN
    # ========================================================================
    with tab_drill:
        choices = {_row_label(r, i): r for i, r in enumerate(rows, 1)}
        selected_label = st.selectbox("Select a recorded query", list(choices.keys()), key="drill_query_select")
        selected = choices[selected_label]

        st.caption(f"**Query:** {selected.get('raw_query', '')}")
        st.caption(f"**Timestamp:** {selected.get('timestamp', '')[:19]}  |  **Latency:** {selected.get('latency_ms', '—')} ms  |  **Mode:** {selected.get('retrieval_mode', '—')}")

        st.markdown("---")
        st.markdown("#### Evaluation Scores")
        eval_data = {**(selected.get("evaluation") or {}).get("criteria", {}), **(selected.get("answer_evaluation") or {})}
        if eval_data:
            score_rows = []
            for name, metric in eval_data.items():
                if isinstance(metric, dict):
                    score_rows.append({
                        "Metric": name.replace("_", " ").title(),
                        "Score": metric.get("score", "—"),
                        "Reason": str(metric.get("reason", ""))[:120],
                    })
            if score_rows:
                st.dataframe(score_rows, hide_index=True, use_container_width=True)
        else:
            st.caption("No evaluation scores recorded for this query.")

        st.markdown("#### Context Chunks Sent to LLM")
        ctx = (selected.get("trace") or {}).get("context") or {}
        final_chunks = ctx.get("final_chunks") or []
        if final_chunks:
            _render_chunk_list(final_chunks if isinstance(final_chunks, list) else [])
        else:
            context_summary = selected.get("context") or {}
            if context_summary:
                st.caption(
                    f"Context summary: {context_summary.get('chunks', '—')} chunks, "
                    f"{context_summary.get('chars', '—')} chars, "
                    f"sources: {', '.join(context_summary.get('sources', []))}"
                )
            else:
                st.caption("No context chunk detail recorded for this query.")

        with st.expander("Full Trace JSON", expanded=False):
            st.json({
                "trace": selected.get("trace", {}),
                "evaluation": selected.get("evaluation", {}),
                "answer_evaluation": selected.get("answer_evaluation", {}),
            })


def _render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Evaluation Dashboard", "Calculation-visible live heuristic evaluation, separated from labelled benchmarks."))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet."); return
    data = {**rows[0].get("evaluation", {}).get("criteria", {}), **rows[0].get("answer_evaluation", {})}
    formulas = {
        "retrieval_strength": "60 + min(35, relative_gap75), plus 5 when top candidate is in both retrievers; clamped 10100.",
        "direct_answer_support": "Maximum top-6 candidate support: definitional 50 + up to 30, or density path 30 + up to 20; +15 filename match; +15term coverage; clamped 5100.",
        "evidence_relevance": "(direct100 + broad55 + tangential10)/inspected, +10 when directly relevant evidence exists.",
        "retriever_agreement": "50% Jaccard(Dense,BM25) + 50% top-3 overlap ratio100.",
        "ranking_stability": "Average top-3 Dense/BM25 survival in RRF top-5 100; clamped 20100.",
        "evidence_sufficiency": "Simple: 95/75/40 based on direct support 70/45/else; complex: term coverage70 + min(20,chunks6).",
        "evidence_coherence": "Rule-based source concentration and source/query-term alignment.",
        "facet_coverage": "Evidence-covered detected live facets / detected facets 100.",
        "evidence_claim_grounding": "Supported citations / citations 100.",
        "citation_coverage": "Fully supported claims / extracted claims 100.",
        "citation_correctness": "Citations with direct answer support / citations 100.",
    }
    for name, metric in data.items():
        with st.expander(f"{metric.get('label', name.replace('_', ' ').title())}: {metric.get('score', 'Not instrumented')}"):
            st.write("**Definition:**", metric.get("definition", "Live answer/evidence check."))
            st.write("**Actual calculation:**", formulas.get(name, "Not instrumented"))
            st.write("**Data used:**", metric.get("metrics", metric.get("reason", "Recorded trace.")))
            st.write("**Interpretation:**", metric.get("reason", "Higher only reflects this project-specific heuristic."))
            st.caption("Standard vs heuristic: Project-specific heuristic, not accuracy.")
    st.subheader("Labelled benchmark evaluation")
    st.write("Precision@K = relevant retrieved / K; Recall@K = relevant retrieved / total relevant; MRR = 1 / rank of first relevant; nDCG@K = standard discounted gain. These require labels and are not live heuristic scores.")
    st.caption("Latency = end-to-end request duration. Median is available in Observation; P95 is Not instrumented until enough recorded observations exist.")


def _render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare system behavior using compact query identifiers."))
    rows = load_observations(pipeline.vectorstore_path)
    if len(rows) < 2:
        st.info("Run at least two queries to compare observations."); return
    labels = [_short_topic(r["raw_query"], i) for i, r in enumerate(rows, 1)]
    first = rows[labels.index(st.selectbox("First query", labels, key="compare_a"))]
    second = rows[labels.index(st.selectbox("Second query", labels, index=1, key="compare_b"))]
    def metrics(row):
        c, a = row.get("evaluation", {}).get("criteria", {}), row.get("answer_evaluation", {})
        return {"Latency ms": row.get("latency_ms"), "Retrieval strength": c.get("retrieval_strength", {}).get("score"), "Agreement": c.get("retriever_agreement", {}).get("score"), "Evidence support": c.get("direct_answer_support", {}).get("score"), "Grounding": a.get("evidence_claim_grounding", {}).get("score"), "Citation coverage": a.get("citation_coverage", {}).get("score")}
    left, right = metrics(first), metrics(second)
    st.dataframe([{"Metric": key, labels[rows.index(first)]: left[key], labels[rows.index(second)]: right[key]} for key in left], hide_index=True, use_container_width=True)
    changed = [f"{key}: {left[key]}  {right[key]}" for key in left if left[key] != right[key]]
    st.subheader("What changed?"); st.write("; ".join(changed) if changed else "No recorded metric difference.")


def _load_persistent_chat_history(vectorstore_path: Path | str | None = None) -> list[dict]:
    path = Path(vectorstore_path or VECTORSTORE_DIR) / "chat_history.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not read chat_history.json: %s", exc)
    return []


def _save_persistent_chat_turn(vectorstore_path: Path | str | None, user_entry: dict, assistant_entry: dict) -> None:
    path = Path(vectorstore_path or VECTORSTORE_DIR) / "chat_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    history = _load_persistent_chat_history(vectorstore_path)
    history.append(user_entry)
    history.append(assistant_entry)
    try:
        path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not write chat_history.json: %s", exc)


def _save_persistent_chat_history(vectorstore_path: Path | str | None, history: list) -> None:
    """Save entire chat history to file."""
    path = Path(vectorstore_path or VECTORSTORE_DIR) / "chat_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not write chat_history.json: %s", exc)


def _clear_persistent_chat_history(vectorstore_path: Path | str | None = None) -> None:
    path = Path(vectorstore_path or VECTORSTORE_DIR) / "chat_history.json"
    if path.exists():
        try:
            path.unlink()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not delete chat_history.json: %s", exc)


def _render_chat_history_page(pipeline: RAGPipeline) -> None:
    render_html(
        render_page_header(
            "CONVERSATION LOGS",
            "Chat & Query History",
            "Complete persistent log of all questions asked and answers generated.",
        )
    )

    history = _load_persistent_chat_history(pipeline.vectorstore_path)
    turns: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(history):
        item = history[i]
        if item.get("role") == "user":
            user_msg = item
            asst_msg = (
                history[i + 1]
                if (i + 1 < len(history) and history[i + 1].get("role") == "assistant")
                else None
            )
            turns.append((user_msg, asst_msg))
            i += 2 if asst_msg else 1
        else:
            i += 1

    top_col1, top_col2, top_col3 = st.columns([4, 5, 2.5])
    with top_col1:
        st.metric("Total Questions Asked", len(turns))
    with top_col2:
        search_filter = st.text_input(
            "Filter queries",
            placeholder="Search questions or responses...",
            key="chat_hist_filter",
            label_visibility="collapsed",
        )
    with top_col3:
        if turns and st.button(" Clear All History", key="clear_all_chat_hist", **WIDE):
            _clear_persistent_chat_history(pipeline.vectorstore_path)
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
            st.rerun()

    if not turns:
        st.info("No queries have been submitted yet. Go to Mission Control to ask questions about the universe!")
        return

    filtered_turns = turns
    if search_filter:
        q_low = search_filter.strip().lower()
        filtered_turns = [
            (u, a)
            for (u, a) in turns
            if q_low in (u.get("content") or "").lower()
            or (a and q_low in (a.get("content") or "").lower())
        ]
        st.caption(f"Showing {len(filtered_turns)} of {len(turns)} turns matching '{search_filter}'")

    for idx, (user_msg, asst_msg) in enumerate(reversed(filtered_turns), start=1):
        timestamp = user_msg.get("timestamp", "")
        q_text = user_msg.get("content", "")
        with st.expander(
            f"Q: {q_text[:85]}{'...' if len(q_text) > 85 else ''} {(' ' + timestamp) if timestamp else ''}",
            expanded=(idx == 1),
        ):
            st.markdown(f"**Question:**\n{q_text}")
            if asst_msg:
                st.markdown("---")
                st.markdown(f"**Answer:**\n{asst_msg.get('content', '')}")
                meta = asst_msg.get("meta") or {}
                if meta:
                    _render_answer_extras(meta)


if __name__ == "__main__":
    main()




