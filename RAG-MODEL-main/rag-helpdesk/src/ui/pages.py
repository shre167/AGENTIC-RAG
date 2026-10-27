from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import VECTORSTORE_DIR
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline.observation_store import load_observations
from src.ui.astronomy_theme import render_html, render_page_header, render_section_title


def render_observation_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Observation & Evaluation", "Dashboard"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    st.metric("Total Queries", len(rows))


def render_dashboard_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("EVALUATION", "Evaluation Dashboard", "Live metrics"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")
        return
    st.metric("Total Queries", len(rows))


def render_comparison_page(pipeline: RAGPipeline) -> None:
    render_html(render_page_header("RUNTIME DIAGNOSTICS", "Query Comparison", "Compare queries"))
    rows = load_observations(pipeline.vectorstore_path)
    if not rows:
        st.info("No observations recorded yet.")


def _normalize_chunk(c: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Ensure all required canonical metadata fields are present."""
    text = str(
        c.get("text")
        or c.get("document")
        or c.get("page_content")
        or ""
    ).strip()
    filename = str(c.get("filename") or c.get("book_name") or c.get("source_name") or "untitled_document.pdf")
    
    # Resolve clean human-readable book title
    raw_title = str(c.get("book_title") or c.get("book") or c.get("source_label") or "")
    if not raw_title or raw_title.lower().endswith(".pdf") or raw_title.lower() == "untitled document":
        raw_stem = Path(filename).stem
        clean_title = raw_stem.replace(" (1)", "").replace("_", " ").strip()
    else:
        clean_title = raw_title.replace(" (1)", "").strip()
        if clean_title.lower().endswith(".pdf"):
            clean_title = clean_title[:-4].strip()
            
    if not clean_title:
        clean_title = "Untitled document"

    ch_num = c.get("chapter_num") if c.get("chapter_num") is not None else c.get("chapter")
    try:
        ch_int = int(ch_num) if ch_num is not None and int(ch_num) > 0 else 0
    except (ValueError, TypeError):
        ch_int = 0

    p_start = c.get("page_start") or c.get("page_number") or 0
    p_end = c.get("page_end") or p_start
    try:
        p_start_int = int(p_start)
    except (ValueError, TypeError):
        p_start_int = 0
    try:
        p_end_int = int(p_end)
    except (ValueError, TypeError):
        p_end_int = p_start_int

    sec = str(c.get("section_heading") or c.get("section_title") or c.get("section") or "").strip()

    return {
        "chunk_id": str(c.get("chunk_id") or f"chunk_{index}"),
        "chunk_index": int(c.get("chunk_index") if c.get("chunk_index") is not None else index),
        "book": clean_title,
        "book_title": clean_title,
        "book_name": filename,
        "filename": filename,
        "book_id": str(c.get("book_id") or "untitled_document_1"),
        "chapter_num": ch_int,
        "chapter": ch_int if ch_int > 0 else None,
        "chapter_title": str(c.get("chapter_title") or "").strip(),
        "section_heading": sec,
        "section_title": sec,
        "section": sec or "General",
        "page_start": p_start_int,
        "page_end": p_end_int,
        "page_range": f"{p_start_int}–{p_end_int}" if p_end_int > p_start_int else str(p_start_int),
        "token_count": int(c.get("token_count") or len(text.split())),
        "character_count": len(text),
        "prev_chunk_id": str(c.get("prev_chunk_id") or ""),
        "next_chunk_id": str(c.get("next_chunk_id") or ""),
        "text": text,
        "raw": c,
    }


def render_chunk_explorer(chunks: list[dict[str, Any]], pipeline: Any = None) -> None:
    """
    Chunk Explorer organized around book and chapter hierarchy.
    Exposes canonical summary, chapter distribution, per-book breakdown,
    hierarchical Chapter -> Section -> Chunks accordion, and
    individual chunk inspection with full paragraph text.
    """
    # 1. Fallback to Chroma if chunks empty
    if not chunks and pipeline and hasattr(pipeline, "vector_store"):
        try:
            data = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
            if data and data.get("ids"):
                chunks = []
                for i, cid in enumerate(data["ids"]):
                    m = data["metadatas"][i] if i < len(data["metadatas"]) else {}
                    doc = data["documents"][i] if i < len(data["documents"]) else ""
                    chunks.append({**m, "chunk_id": cid, "text": doc})
        except Exception:
            chunks = []

    # 2. Fallback to cache
    if not chunks:
        cache_path = Path(VECTORSTORE_DIR) / "canonical_chunks_cache.json"
        if cache_path.exists():
            try:
                chunks = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                chunks = []

    if not chunks:
        st.warning("No canonical chunks are currently available in the database. Ingest documents to inspect.")
        return

    # Normalize all chunks
    normalized_chunks = [_normalize_chunk(c, i) for i, c in enumerate(chunks)]
    normalized_chunks.sort(key=lambda x: (x["book_title"], x["chapter_num"] or 0, x["chunk_index"]))
    chunk_by_id = {c["chunk_id"]: c for c in normalized_chunks}

    all_books = sorted(list({c["book_title"] for c in normalized_chunks if c["book_title"]}))
    book_options = ["All books"] + all_books if len(all_books) > 1 else (all_books or ["All books"])

    # Synchronize selected book in session state
    if "book_hierarchy_selected_book" not in st.session_state or st.session_state.book_hierarchy_selected_book not in book_options:
        st.session_state.book_hierarchy_selected_book = book_options[0]

    # Handle dropdown change cleanly
    def _on_book_selector_change():
        new_val = st.session_state.get("book_hierarchy_book_selector")
        if new_val:
            st.session_state.book_hierarchy_selected_book = new_val
            st.session_state.selected_chunk_id = None
            st.session_state.pop("direct_chunk_selector", None)

    cur_book_idx = book_options.index(st.session_state.book_hierarchy_selected_book) if st.session_state.book_hierarchy_selected_book in book_options else 0
    selected_book = st.selectbox(
        "📚 Select Book to Inspect",
        options=book_options,
        index=cur_book_idx,
        key="book_hierarchy_book_selector",
        on_change=_on_book_selector_change,
        help="Switch between viewing all ingested books or drilling down into a specific book's chapter-wise chunks.",
    )
    st.session_state.book_hierarchy_selected_book = selected_book

    filtered_chunks = [
        c for c in normalized_chunks
        if selected_book == "All books" or c["book_title"] == selected_book
    ]
    if not filtered_chunks:
        st.warning("No chunks available for the selected book.")
        return

    # Ensure selected chunk ID exists in filtered set
    valid_ids = [c["chunk_id"] for c in filtered_chunks]
    if not st.session_state.get("selected_chunk_id") or st.session_state.selected_chunk_id not in valid_ids:
        st.session_state.selected_chunk_id = valid_ids[0]

    # =========================================================================
    # 1. SUMMARY METRICS
    # =========================================================================
    scope_title = f"Summary: {selected_book}" if selected_book != "All books" else "Canonical Chunks Summary (All Books)"
    render_html(render_section_title(scope_title))

    total_chunks = len(filtered_chunks)
    chapters = sorted(list({c["chapter_num"] for c in filtered_chunks if c["chapter_num"] > 0}))
    has_unnumbered = any(c["chapter_num"] == 0 for c in filtered_chunks)
    sections = {c["section"] for c in filtered_chunks if c["section"] and c["section"] != "General"}
    tokens = [c["token_count"] for c in filtered_chunks]
    avg_tokens = round(sum(tokens) / len(tokens)) if tokens else 0
    min_tokens = min(tokens) if tokens else 0
    max_tokens = max(tokens) if tokens else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Chunks", total_chunks)
    c2.metric("Chapters", f"{len(chapters)}" + (" (+Intro)" if has_unnumbered else ""))
    c3.metric("Sections", len(sections))
    c4.metric("Avg Tokens", avg_tokens)
    c5.metric("Min Tokens", min_tokens)
    c6.metric("Max Tokens", max_tokens)

    # If "All books" is selected, display an Ingested Books Overview Table
    if selected_book == "All books" and len(all_books) > 1:
        st.markdown("#### 📚 Ingested Books Overview")
        book_rows = []
        for b in all_books:
            b_chunks = [c for c in normalized_chunks if c["book_title"] == b]
            b_chs = sorted(list({c["chapter_num"] for c in b_chunks if c["chapter_num"] > 0}))
            b_has_intro = any(c["chapter_num"] == 0 for c in b_chunks)
            b_toks = [c["token_count"] for c in b_chunks]
            b_avg = round(sum(b_toks) / len(b_toks)) if b_toks else 0
            b_fn = b_chunks[0]["filename"] if b_chunks else ""
            book_rows.append({
                "Book Title": b,
                "Filename": b_fn,
                "Total Chunks": len(b_chunks),
                "Chapters": f"{len(b_chs)}" + (" (+Intro)" if b_has_intro else ""),
                "Chapter Range": f"Ch. {min(b_chs)}–{max(b_chs)}" if b_chs else "Unnumbered",
                "Avg Tokens/Chunk": b_avg,
            })
        st.dataframe(pd.DataFrame(book_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # 2. CHAPTER CHUNK DISTRIBUTION
    # =========================================================================
    dist_heading = f"Chapter Distribution: {selected_book}" if selected_book != "All books" else "Chapter Chunk Distribution (Per Book & Chapter)"
    render_html(render_section_title(dist_heading))

    if selected_book != "All books":
        # Specific book chapter distribution
        chapter_dist = {}
        if has_unnumbered:
            intro_count = sum(1 for c in filtered_chunks if c["chapter_num"] == 0)
            chapter_dist["Intro / Front Matter"] = intro_count
        for ch in chapters:
            ch_title = next((c["chapter_title"] for c in filtered_chunks if c["chapter_num"] == ch and c["chapter_title"]), "")
            label = f"Ch. {ch} ({ch_title[:20]})" if ch_title else f"Chapter {ch}"
            chapter_dist[label] = sum(1 for c in filtered_chunks if c["chapter_num"] == ch)

        col_chart, col_stats = st.columns([7, 3])
        with col_chart:
            chart_df = pd.DataFrame(
                [{"Chapter": k, "Chunks": v} for k, v in chapter_dist.items()]
            ).set_index("Chapter")
            st.bar_chart(chart_df, use_container_width=True)

        with col_stats:
            st.markdown(f"**Chunks Per Chapter ({selected_book})**")
            for k, v in chapter_dist.items():
                pct = round((v / total_chunks) * 100, 1) if total_chunks else 0
                st.markdown(f"• **{k}**: {v} chunks `({pct}%)`")
    else:
        # All books: breakdown per book and chapter
        col_b_chart, col_b_stats = st.columns([7, 3])
        with col_b_chart:
            # Bar chart of chunks per book
            book_dist = {b: sum(1 for c in normalized_chunks if c["book_title"] == b) for b in all_books}
            b_chart_df = pd.DataFrame([{"Book": k, "Chunks": v} for k, v in book_dist.items()]).set_index("Book")
            st.bar_chart(b_chart_df, use_container_width=True)

        with col_b_stats:
            st.markdown("**Per-Book Breakdown**")
            for b in all_books:
                b_c = sum(1 for c in normalized_chunks if c["book_title"] == b)
                pct = round((b_c / len(normalized_chunks)) * 100, 1) if normalized_chunks else 0
                st.markdown(f"• **{b}**: {b_c} chunks `({pct}%)`")

        with st.expander("📊 Detailed Chapter Counts for Each Book", expanded=True):
            for b in all_books:
                b_chunks = [c for c in normalized_chunks if c["book_title"] == b]
                b_chapters = sorted(list({c["chapter_num"] for c in b_chunks if c["chapter_num"] > 0}))
                b_has_intro = any(c["chapter_num"] == 0 for c in b_chunks)
                ch_pills = []
                if b_has_intro:
                    ch_pills.append(f"**Intro:** {sum(1 for c in b_chunks if c['chapter_num'] == 0)} chunks")
                for ch in b_chapters:
                    ch_count = sum(1 for c in b_chunks if c["chapter_num"] == ch)
                    ch_title = next((c["chapter_title"] for c in b_chunks if c["chapter_num"] == ch and c["chapter_title"]), "")
                    title_hint = f" ({ch_title})" if ch_title else ""
                    ch_pills.append(f"**Ch. {ch}{title_hint}:** {ch_count} chunks")
                st.markdown(f"**📖 {b}** ({len(b_chunks)} total chunks): " + " | ".join(ch_pills))

    # =========================================================================
    # 3. CHUNK CONSISTENCY DASHBOARD
    # =========================================================================
    render_html(render_section_title(f"Chunk Consistency: {selected_book}"))

    total = len(filtered_chunks)
    empty = sum(1 for c in filtered_chunks if not c["text"].strip())
    dup_ids = total - len({c["chunk_id"] for c in filtered_chunks})
    dup_text = total - len({c["text"][:200] for c in filtered_chunks if c["text"]})
    missing_chapter = sum(1 for c in filtered_chunks if not c["chapter_num"])
    missing_section = sum(1 for c in filtered_chunks if not c["section_heading"])
    missing_page = sum(1 for c in filtered_chunks if not c["page_start"])
    below_min = sum(1 for c in filtered_chunks if c["token_count"] < 80)
    above_max = sum(1 for c in filtered_chunks if c["token_count"] > 450)

    # Broken prev/next links
    

    boundary_violations = 0
    for i in range(1, len(filtered_chunks)):
        prev_ch = filtered_chunks[i-1]["chapter_num"]
        curr_ch = filtered_chunks[i]["chapter_num"]
        prev_next = filtered_chunks[i-1]["next_chunk_id"]
        curr_prev = filtered_chunks[i]["prev_chunk_id"]
        if prev_ch != curr_ch and prev_next and curr_prev:
            boundary_violations += 1

    issues = [
        ("Empty chunks", empty, empty > 0),
        ("Duplicate IDs", dup_ids, dup_ids > 0),
        ("Duplicate text", dup_text, dup_text > 0),
        ("Missing chapter", missing_chapter, False),
        ("Missing section", missing_section, False),
        ("Missing page", missing_page, False),
        ("Below min tokens", below_min, below_min > 0),
        ("Above max tokens", above_max, above_max > 0),
        ("Chapter boundary links", boundary_violations, boundary_violations > 0),
    ]

    warn_cols = st.columns(4)
    for idx, (label, count, is_issue) in enumerate(issues):
        col = warn_cols[idx % 4]
        if is_issue:
            col.metric(label, count, delta="⚠️ issue", delta_color="inverse")
        else:
            col.metric(label, count, delta="✅ OK", delta_color="off")

    # =========================================================================
    # 4. HIERARCHICAL STRUCTURE (Book -> Chapter -> Section -> Chunks)
    # =========================================================================
    render_html(render_section_title("Book Hierarchy (Chapter → Section → Chunks)"))
    st.caption("Expand any chapter or section below to explore its constituent chunks.")

    # Function to render chapter chunks accordion
    def _render_chapter_group(ch_chunks: list[dict[str, Any]], ch_label: str, is_default_open: bool = False):
        with st.expander(f"{ch_label}  ({len(ch_chunks)} chunks)", expanded=is_default_open):
            section_dict: dict[str, list[dict[str, Any]]] = {}
            for c in ch_chunks:
                sec_name = c["section"] or "General / Unnamed Section"
                section_dict.setdefault(sec_name, []).append(c)

            for sec_name, sec_chunks in section_dict.items():
                st.markdown(f"##### 📑 Section: {sec_name} `({len(sec_chunks)} chunks)`")
                chunk_cols = st.columns(min(4, max(1, len(sec_chunks))))
                for idx, sc in enumerate(sec_chunks):
                    col = chunk_cols[idx % len(chunk_cols)]
                    is_active = (sc["chunk_id"] == st.session_state.selected_chunk_id)
                    btn_label = f"#{sc['chunk_index']:03d} (pp. {sc['page_range']})"
                    btn_type = "primary" if is_active else "secondary"
                    if col.button(btn_label, key=f"btn_chunk_{sc['chunk_id']}", type=btn_type, use_container_width=True):
                        st.session_state.selected_chunk_id = sc["chunk_id"]
                        st.rerun()

    books_to_show = all_books if selected_book == "All books" else [selected_book]
    for b in books_to_show:
        b_chunks = [c for c in normalized_chunks if c["book_title"] == b]
        b_chapters = sorted(list({c["chapter_num"] for c in b_chunks if c["chapter_num"] > 0}))
        b_has_intro = any(c["chapter_num"] == 0 for c in b_chunks)

        if selected_book == "All books":
            st.markdown(f"### 📖 Book: **{b}** `({len(b_chunks)} chunks)`")

        # Introduction / unnumbered chunks if present
        if b_has_intro:
            intro_chunks = [c for c in b_chunks if c["chapter_num"] == 0]
            _render_chapter_group(intro_chunks, "📖 Front Matter / Introduction / Unnumbered", is_default_open=False)

        for ch in b_chapters:
            ch_chunks = [c for c in b_chunks if c["chapter_num"] == ch]
            ch_title = next((c["chapter_title"] for c in ch_chunks if c["chapter_title"]), "")
            title_str = f" — {ch_title}" if ch_title else ""
            _render_chapter_group(ch_chunks, f"📖 Chapter {ch}{title_str}", is_default_open=(ch == 1 and selected_book != "All books"))

    # =========================================================================
    # 5. CHUNK INSPECTION & DETAILS PANEL
    # =========================================================================
    render_html(render_section_title("Chunk Details & Context Inspector"))

    # Direct chunk selector dropdown
    chunk_options = {
        f"[{c['book_title'][:22]}] Chunk #{c['chunk_index']:03d} | Ch. {c['chapter_num']} | pp. {c['page_range']} | {c['section'][:25]}": c["chunk_id"]
        for c in filtered_chunks
    }
    opt_labels = list(chunk_options.keys())
    opt_ids = list(chunk_options.values())

    cur_idx = opt_ids.index(st.session_state.selected_chunk_id) if st.session_state.selected_chunk_id in opt_ids else 0

    selected_label = st.selectbox(
        "Direct Chunk Selector",
        options=opt_labels,
        index=cur_idx,
        key="direct_chunk_selector",
    )
    st.session_state.selected_chunk_id = chunk_options[selected_label]
    selected_chunk = chunk_by_id.get(st.session_state.selected_chunk_id, filtered_chunks[0])

    # Display clear Chunk Details Panel
    det_c1, det_c2, det_c3, det_c4 = st.columns(4)
    det_c1.metric("Book", selected_chunk["book_title"])
    det_c2.metric("Chapter", f"Chapter {selected_chunk['chapter_num']}" if selected_chunk['chapter_num'] else "None/Intro")
    det_c3.metric("Section", selected_chunk["section"][:25] + ("..." if len(selected_chunk["section"]) > 25 else ""))
    det_c4.metric("Page Range", f"Pages {selected_chunk['page_range']}")

    det_c5, det_c6, det_c7, det_c8 = st.columns(4)
    det_c5.metric("Chunk Index", f"#{selected_chunk['chunk_index']}")
    det_c6.metric("Token Count", f"{selected_chunk['token_count']} tokens")
    det_c7.metric("Character Count", f"{selected_chunk['character_count']} chars")
    det_c8.metric("PDF Filename", selected_chunk["filename"][:20] + ("..." if len(selected_chunk["filename"]) > 20 else ""))

    st.caption(f"Chunk ID: `{selected_chunk['chunk_id']}`")
    if selected_chunk["chapter_title"]:
        st.markdown(f"**Chapter Title:** {selected_chunk['chapter_title']}")
    if selected_chunk["section_heading"]:
        st.markdown(f"**Section Heading:** {selected_chunk['section_heading']}")

    chunk_body = selected_chunk["text"]
    st.markdown("**Chunk text**")
    if chunk_body.strip():
        st.text_area(
            "Paragraph content",
            value=chunk_body,
            height=min(400, max(120, len(chunk_body) // 3)),
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.warning("No paragraph text is stored for this chunk.")

    # Expandable Raw JSON View
    with st.expander("Raw Metadata / JSON View", expanded=False):
        st.json(selected_chunk["raw"])
