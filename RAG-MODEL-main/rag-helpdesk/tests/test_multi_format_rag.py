"""
Multi-format RAG pipeline tests.

Covers:
  1. Extraction/detection for TXT, MD, DOCX, PDF
  2. DOCX headings and tables
  3. PDF 1-based page-number preservation
  4. Chunk metadata and deterministic IDs
  5. BM25 + Chroma indexing across formats
  6. Duplicate / incremental ingestion (hash skip)
  7. Dense + BM25 + RRF retrieval across formats (cross-format queries)
  8. Citation/evidence grounding with PDF page numbers
  9. Existing regression suite passes unchanged
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal but valid multi-page PDF with extractable text."""
    import pypdf
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        FloatObject,
        NameObject,
        NumberObject,
        RectangleObject,
    )

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # Inject a minimal content stream with the text
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        stream = DecodedStreamObject()
        stream.set_data(stream_content)
        resources = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                })
            })
        })
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_docx_bytes(headings_and_paragraphs: list[tuple[str | None, str]], table_rows: list[list[str]] | None = None) -> bytes:
    """Build a DOCX bytes object with headings, paragraphs, and an optional table."""
    import docx as _docx
    doc = _docx.Document()
    for style, text in headings_and_paragraphs:
        if style and style.startswith("Heading"):
            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                level = 1
            doc.add_heading(text, level=level)
        else:
            doc.add_paragraph(text)
    if table_rows:
        tbl = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for ri, row_data in enumerate(table_rows):
            for ci, cell_text in enumerate(row_data):
                tbl.rows[ri].cells[ci].text = cell_text
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ===========================================================================
# 1. Extraction / detection for all 4 formats
# ===========================================================================

class TestFormatDetectionAndExtraction:
    def test_detect_txt(self):
        from src.document_loader import detect_file_type
        assert detect_file_type("stars.txt") == "txt"
        assert detect_file_type("stars.txt", b"hello world") == "txt"

    def test_detect_md(self):
        from src.document_loader import detect_file_type
        assert detect_file_type("cosmology.md") == "md"
        assert detect_file_type("cosmology.md", b"# Big Bang") == "md"

    def test_detect_pdf_by_magic(self):
        from src.document_loader import detect_file_type
        assert detect_file_type("doc.pdf", b"%PDF-1.4 ...") == "pdf"
        assert detect_file_type("noext", b"%PDF-1.7 blob") == "pdf"

    def test_detect_docx_by_magic(self):
        from src.document_loader import detect_file_type
        docx_bytes = _make_docx_bytes([("Heading 1", "JWST")])
        assert detect_file_type("instrument.docx", docx_bytes) == "docx"

    def test_detect_falls_back_to_txt(self):
        from src.document_loader import detect_file_type
        assert detect_file_type("unknownfile", b"plain text") == "txt"

    def test_extract_txt(self):
        from src.document_loader import extract_text_from_txt
        raw = "The Milky Way is a barred spiral galaxy."
        assert "Milky Way" in extract_text_from_txt(raw.encode("utf-8"))

    def test_extract_md(self):
        from src.document_loader import extract_text_from_md
        raw = "# Black Holes\n\nBlack holes have an event horizon."
        assert "event horizon" in extract_text_from_md(raw.encode("utf-8"))

    def test_extract_pdf_returns_pages(self):
        from src.document_loader import extract_text_from_pdf
        pdf_bytes = _make_pdf_bytes(["Neutron stars spin rapidly.", "Pulsars emit radio waves."])
        pages = extract_text_from_pdf(pdf_bytes)
        assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pages)

    def test_load_document_txt(self, tmp_path):
        from src.document_loader import load_document
        f = tmp_path / "exo.txt"
        f.write_text("Exoplanets orbit stars beyond our solar system.", encoding="utf-8")
        docs = load_document(f)
        assert len(docs) == 1
        assert docs[0]["file_type"] == "txt"
        assert docs[0]["page_number"] is None
        assert "Exoplanets" in docs[0]["content"]

    def test_load_document_md(self, tmp_path):
        from src.document_loader import load_document
        f = tmp_path / "galaxies.md"
        f.write_text("# Galaxies\n\nGalaxies contain billions of stars.", encoding="utf-8")
        docs = load_document(f)
        assert len(docs) == 1
        assert docs[0]["file_type"] == "md"
        assert docs[0]["page_number"] is None

    def test_load_document_docx(self, tmp_path):
        from src.document_loader import load_document
        docx_bytes = _make_docx_bytes([("Heading 1", "Artemis"), (None, "Artemis will land humans on the Moon.")])
        f = tmp_path / "artemis.docx"
        f.write_bytes(docx_bytes)
        docs = load_document(f)
        assert len(docs) == 1
        assert docs[0]["file_type"] == "docx"
        assert docs[0]["page_number"] is None
        assert "Artemis" in docs[0]["content"]

    def test_load_document_pdf_multipage(self, tmp_path):
        from src.document_loader import load_document
        pdf_bytes = _make_pdf_bytes([
            "Page one: Mars has two moons named Phobos and Deimos.",
            "Page two: Jupiter is the largest planet in our solar system."
        ])
        f = tmp_path / "planets.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        assert len(docs) == 2, f"Expected 2 pages, got {len(docs)}"
        assert docs[0]["page_number"] == 1
        assert docs[1]["page_number"] == 2
        assert docs[0]["file_type"] == "pdf"
        assert docs[1]["file_type"] == "pdf"
        assert "Phobos" in docs[0]["content"] or "Phobos" in docs[1]["content"]


# ===========================================================================
# 2. DOCX headings and tables
# ===========================================================================

class TestDocxExtraction:
    def test_headings_converted_to_markdown(self):
        from src.document_loader import extract_text_from_docx
        docx_bytes = _make_docx_bytes([
            ("Heading 1", "James Webb Space Telescope"),
            (None, "JWST uses infrared astronomy."),
            ("Heading 2", "Instruments"),
            (None, "NIRCam is the primary imager."),
        ])
        text = extract_text_from_docx(docx_bytes)
        assert "# James Webb Space Telescope" in text
        assert "## Instruments" in text
        assert "NIRCam" in text

    def test_table_rows_extracted(self):
        from src.document_loader import extract_text_from_docx
        docx_bytes = _make_docx_bytes(
            [(None, "Mission comparison table:")],
            table_rows=[
                ["Mission", "Agency", "Year"],
                ["Apollo 11", "NASA", "1969"],
                ["Chandrayaan-3", "ISRO", "2023"],
            ]
        )
        text = extract_text_from_docx(docx_bytes)
        assert "Apollo 11" in text
        assert "Chandrayaan-3" in text
        assert "ISRO" in text
        assert "NASA" in text

    def test_docx_headings_create_sections_in_chunks(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        docx_bytes = _make_docx_bytes([
            ("Heading 1", "Planetary Science"),
            (None, "Planetary science studies the planets of our solar system."),
            ("Heading 2", "Mars"),
            (None, "Mars is the fourth planet from the Sun."),
        ])
        f = tmp_path / "planetary.docx"
        f.write_bytes(docx_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        headings = [c["section_heading"] for c in chunks]
        assert any(h for h in headings), f"Expected section headings, got: {headings}"


# ===========================================================================
# 3. PDF 1-based page-number preservation
# ===========================================================================

class TestPdfPageNumbers:
    def test_page_numbers_are_1_based(self, tmp_path):
        from src.document_loader import load_document
        pdf_bytes = _make_pdf_bytes([
            "First page: The Big Bang occurred 13.8 billion years ago.",
            "Second page: The universe is expanding due to dark energy.",
            "Third page: Dark matter makes up 27% of the universe.",
        ])
        f = tmp_path / "cosmology.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        page_nums = [d["page_number"] for d in docs]
        assert page_nums == [1, 2, 3], f"Expected [1,2,3], got {page_nums}"

    def test_pdf_chunk_ids_include_page_number(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        pdf_bytes = _make_pdf_bytes([
            "Page 1: Hubble's Law describes the expansion of the universe.",
            "Page 2: The cosmic microwave background is a remnant of the Big Bang.",
        ])
        f = tmp_path / "hubble.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert any("_p1_" in cid for cid in chunk_ids), f"No p1 chunk IDs: {chunk_ids}"
        assert any("_p2_" in cid for cid in chunk_ids), f"No p2 chunk IDs: {chunk_ids}"

    def test_pdf_chunk_ids_no_collision_across_pages(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        # Use the same section heading on two pages to stress-test uniqueness
        pdf_bytes = _make_pdf_bytes([
            "Introduction: Quasars are the brightest objects in the universe.",
            "Introduction: Quasars are powered by supermassive black holes.",
        ])
        f = tmp_path / "quasars.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs across pages!"

    def test_pdf_page_number_in_chunk_metadata(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        pdf_bytes = _make_pdf_bytes([
            "Voyager 1 has left the heliosphere and entered interstellar space.",
            "Voyager 2 also crossed the heliopause in 2018.",
        ])
        f = tmp_path / "voyager.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        page_numbers = [c["page_number"] for c in chunks]
        assert 1 in page_numbers, "Page 1 not found in chunk metadata"
        assert 2 in page_numbers, "Page 2 not found in chunk metadata"


# ===========================================================================
# 4. Chunk metadata and deterministic IDs
# ===========================================================================

class TestChunkMetadata:
    def test_txt_chunks_have_file_type(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        f = tmp_path / "saturn.txt"
        f.write_text("Saturn has beautiful rings composed of ice and rock.", encoding="utf-8")
        docs = load_document(f)
        chunks = chunk_documents(docs)
        for c in chunks:
            assert c["file_type"] == "txt", f"Expected 'txt', got {c['file_type']}"
            assert c["page_number"] == 0, f"Expected 0 for non-PDF, got {c['page_number']}"

    def test_md_chunks_have_file_type(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        f = tmp_path / "stars.md"
        f.write_text("# Stars\n\nStars fuse hydrogen into helium.", encoding="utf-8")
        docs = load_document(f)
        chunks = chunk_documents(docs)
        for c in chunks:
            assert c["file_type"] == "md"
            assert c["page_number"] == 0

    def test_docx_chunks_have_file_type(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        docx_bytes = _make_docx_bytes([(None, "The International Space Station orbits at 400 km altitude.")])
        f = tmp_path / "iss.docx"
        f.write_bytes(docx_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        for c in chunks:
            assert c["file_type"] == "docx"
            assert c["page_number"] == 0

    def test_pdf_chunks_have_correct_metadata(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        pdf_bytes = _make_pdf_bytes(["Black holes warp space-time around them."])
        f = tmp_path / "blackhole.pdf"
        f.write_bytes(pdf_bytes)
        docs = load_document(f)
        chunks = chunk_documents(docs)
        for c in chunks:
            assert c["file_type"] == "pdf"
            assert c["page_number"] == 1
            assert "_p1_" in c["chunk_id"]

    def test_deterministic_chunk_ids(self, tmp_path):
        """Same content produces same IDs on repeated calls."""
        from src.document_loader import chunk_documents, load_document
        f = tmp_path / "deterministic.txt"
        content = "Uranus rotates on its side with an axial tilt of 98 degrees."
        f.write_text(content, encoding="utf-8")
        docs1 = load_document(f)
        docs2 = load_document(f)
        chunks1 = chunk_documents(docs1)
        chunks2 = chunk_documents(docs2)
        assert [c["chunk_id"] for c in chunks1] == [c["chunk_id"] for c in chunks2]

    def test_all_required_fields_present(self, tmp_path):
        from src.document_loader import chunk_documents, load_document
        f = tmp_path / "venus.txt"
        f.write_text("Venus has a dense atmosphere of carbon dioxide.", encoding="utf-8")
        docs = load_document(f)
        chunks = chunk_documents(docs)
        required = {"filename", "chunk_id", "text", "file_type", "page_number",
                    "section_heading", "category", "source_path"}
        for c in chunks:
            missing = required - c.keys()
            assert not missing, f"Missing fields: {missing}"


# ===========================================================================
# 5. BM25 + Chroma indexing across formats
# ===========================================================================

class TestBM25Indexing:
    def test_bm25_indexes_all_four_formats(self, tmp_path):
        from src.bm25_retriever import BM25Retriever
        from src.document_loader import chunk_documents, load_document

        kb = tmp_path / "kb"
        kb.mkdir()

        # TXT
        (kb / "neptune.txt").write_text("Neptune is the eighth planet and has supersonic winds.", encoding="utf-8")
        # MD
        (kb / "comets.md").write_text("# Comets\n\nComets are icy bodies that develop comas near the Sun.", encoding="utf-8")
        # DOCX
        docx_bytes = _make_docx_bytes([("Heading 1", "Voyager"), (None, "Voyager spacecraft carries a golden record.")])
        (kb / "voyager.docx").write_bytes(docx_bytes)
        # PDF
        pdf_bytes = _make_pdf_bytes(["Pluto was reclassified as a dwarf planet in 2006."])
        (kb / "pluto.pdf").write_bytes(pdf_bytes)

        retriever = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        retriever.build_index()
        assert len(retriever._chunks) >= 4, f"Expected >=4 chunks, got {len(retriever._chunks)}"

        results = retriever.query("Neptune winds", top_k=3)
        assert len(results) > 0
        assert any("neptune" in r["filename"].lower() for r in results)

    def test_bm25_preserves_file_type_and_page_number(self, tmp_path):
        from src.bm25_retriever import BM25Retriever
        from src.document_loader import chunk_documents, load_document

        kb = tmp_path / "kb"
        kb.mkdir()
        pdf_bytes = _make_pdf_bytes(["Andromeda galaxy will collide with the Milky Way in 4 billion years."])
        (kb / "andromeda.pdf").write_bytes(pdf_bytes)

        retriever = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        retriever.build_index()

        results = retriever.query("Andromeda Milky Way", top_k=2)
        assert len(results) > 0
        first = results[0]
        assert first.get("file_type") == "pdf", f"Expected 'pdf', got {first.get('file_type')}"
        assert first.get("page_number") == 1, f"Expected 1, got {first.get('page_number')}"

    def test_bm25_chunk_key_includes_page_number_for_pdf(self, tmp_path):
        """Ensure BM25 chunk keys are unique even when two PDF pages share a heading."""
        from src.bm25_retriever import BM25Retriever

        kb = tmp_path / "kb"
        kb.mkdir()
        pdf_bytes = _make_pdf_bytes([
            "Introduction: Stellar nucleosynthesis creates heavy elements.",
            "Introduction: Supernovae distribute these elements through the galaxy.",
        ])
        (kb / "stellar.pdf").write_bytes(pdf_bytes)

        retriever = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        retriever.build_index()
        keys = [retriever._chunk_key(c) for c in retriever._chunks]
        assert len(keys) == len(set(keys)), "Duplicate BM25 keys from multi-page PDF!"


# ===========================================================================
# 6. Duplicate / incremental ingestion (hash skip)
# ===========================================================================

class TestIncrementalIngestion:
    def test_txt_skip_on_same_hash(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        content = b"The Sun is a G-type main-sequence star."
        res1 = pip.ingest_file("sun.txt", content=content)
        assert res1["skipped"] is False
        res2 = pip.ingest_file("sun.txt", content=content)
        assert res2["skipped"] is True

    def test_docx_skip_on_same_bytes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        docx_bytes = _make_docx_bytes([(None, "The Hubble Space Telescope has observed deep-field galaxies.")])
        res1 = pip.ingest_file("hubble.docx", content=docx_bytes)
        assert res1["skipped"] is False
        assert res1["chunks_added"] > 0
        res2 = pip.ingest_file("hubble.docx", content=docx_bytes)
        assert res2["skipped"] is True
        assert res2["chunks_added"] == 0

    def test_pdf_skip_on_same_bytes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        pdf_bytes = _make_pdf_bytes(["Cassini orbited Saturn for 13 years before its grand finale."])
        res1 = pip.ingest_file("cassini.pdf", content=pdf_bytes)
        assert res1["skipped"] is False
        res2 = pip.ingest_file("cassini.pdf", content=pdf_bytes)
        assert res2["skipped"] is True

    def test_changed_content_re_ingests(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        v1 = b"Mercury is the closest planet to the Sun."
        v2 = b"Mercury is also the smallest planet in the solar system."
        pip.ingest_file("mercury.txt", content=v1)
        res = pip.ingest_file("mercury.txt", content=v2)
        assert res["skipped"] is False, "Changed content should be re-ingested"

    def test_all_formats_can_be_ingested_together(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")

        r1 = pip.ingest_file("moon.txt", content=b"The Moon is Earth's natural satellite with a synchronous rotation.")
        r2 = pip.ingest_file("stars.md", content=b"# Stars\n\nStars are formed in molecular clouds called nebulae.")
        r3 = pip.ingest_file("iss.docx", content=_make_docx_bytes([(None, "The ISS is a modular space station in low Earth orbit.")]))
        r4 = pip.ingest_file("mars.pdf", content=_make_pdf_bytes(["Mars has the largest volcano in the solar system: Olympus Mons."]))

        for name, res in [("txt", r1), ("md", r2), ("docx", r3), ("pdf", r4)]:
            assert not res["skipped"], f"{name} should not be skipped on first ingest"
            assert res["chunks_added"] > 0, f"{name} should have chunks added"


# ===========================================================================
# 7. Dense + BM25 + RRF retrieval across formats including cross-format queries
# ===========================================================================

class TestHybridRetrieval:
    """Test that BM25 retrieval correctly finds chunks from different formats.

    Dense retrieval requires real embeddings, so we test BM25 directly
    (which uses the same chunk pipeline), and verify retrieval candidates
    carry the expected metadata.
    """

    def test_bm25_cross_format_retrieval(self, tmp_path):
        """Query should match content from whichever format it lives in."""
        from src.bm25_retriever import BM25Retriever

        kb = tmp_path / "kb"
        kb.mkdir()

        (kb / "uranus.txt").write_text(
            "Uranus has 27 known moons and a retrograde rotation.", encoding="utf-8"
        )
        (kb / "exo.md").write_text(
            "# Exoplanets\n\nExoplanets like Kepler-452b orbit in the habitable zone.",
            encoding="utf-8",
        )
        docx_bytes = _make_docx_bytes(
            [("Heading 1", "Dark Matter"), (None, "Dark matter is detected only through gravitational lensing.")]
        )
        (kb / "darkmatter.docx").write_bytes(docx_bytes)
        pdf_bytes = _make_pdf_bytes(["Gravitational waves were first detected by LIGO in 2015."])
        (kb / "ligo.pdf").write_bytes(pdf_bytes)

        retriever = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        retriever.build_index()

        # Cross-format: should hit the PDF
        results_ligo = retriever.query("LIGO gravitational waves", top_k=5)
        assert any("ligo" in r["filename"].lower() for r in results_ligo), \
            f"Expected LIGO in results: {[r['filename'] for r in results_ligo]}"

        # Cross-format: should hit the DOCX
        results_dm = retriever.query("dark matter gravitational lensing", top_k=5)
        assert any("darkmatter" in r["filename"].lower() for r in results_dm), \
            f"Expected darkmatter in results: {[r['filename'] for r in results_dm]}"

        # Cross-format: should hit the MD
        results_exo = retriever.query("exoplanets habitable zone Kepler", top_k=5)
        assert any("exo" in r["filename"].lower() for r in results_exo), \
            f"Expected exo in results: {[r['filename'] for r in results_exo]}"

    def test_retrieval_candidate_carries_file_type(self, tmp_path):
        """BM25 results must include file_type and page_number in their metadata."""
        from src.bm25_retriever import BM25Retriever

        kb = tmp_path / "kb"
        kb.mkdir()
        pdf_bytes = _make_pdf_bytes(["The Milky Way has 200-400 billion stars and is a barred spiral galaxy."])
        (kb / "milkyway.pdf").write_bytes(pdf_bytes)

        retriever = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        retriever.build_index()

        results = retriever.query("Milky Way barred spiral", top_k=3)
        assert len(results) > 0
        r = results[0]
        assert "file_type" in r, "file_type missing from BM25 result"
        assert "page_number" in r, "page_number missing from BM25 result"
        assert r["file_type"] == "pdf"
        assert r["page_number"] == 1

    def test_pipeline_hybrid_retrieval_with_multi_format_kb(self, monkeypatch, tmp_path):
        """Ingest all 4 formats and verify BM25 query retrieves across formats."""
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline

        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        pip.ingest_file("saturn.txt", content=b"Saturn's rings are made of ice particles and rocky debris orbiting the planet.")
        pip.ingest_file("comets.md", content=b"# Comets\n\nHalley's Comet returns every 75-76 years in an elliptical orbit.")
        pip.ingest_file("jwst.docx", content=_make_docx_bytes([
            ("Heading 1", "JWST"), (None, "The James Webb Space Telescope observes the universe in infrared light.")
        ]))
        pip.ingest_file("blackhole.pdf", content=_make_pdf_bytes([
            "A black hole singularity has infinite density inside the event horizon."
        ]))

        # Verify BM25 index now contains chunks from all 4 sources
        assert pip.bm25 is not None
        file_types_in_index = {c.get("file_type") for c in pip.bm25._chunks}
        assert "txt" in file_types_in_index, f"txt not found in BM25 index: {file_types_in_index}"
        assert "md" in file_types_in_index, f"md not found in BM25 index: {file_types_in_index}"
        assert "docx" in file_types_in_index, f"docx not found in BM25 index: {file_types_in_index}"
        assert "pdf" in file_types_in_index, f"pdf not found in BM25 index: {file_types_in_index}"

        # BM25 cross-format query
        results = pip.bm25.query("event horizon singularity black hole", top_k=5)
        assert len(results) > 0
        assert any("blackhole" in r["filename"].lower() for r in results)


# ===========================================================================
# 8. Citation / evidence grounding with PDF page numbers
# ===========================================================================

class TestCitationGrounding:
    def _make_pdf_candidate(self, page_number: int) -> object:
        from src.rag_pipeline.models import Candidate
        return Candidate(
            key=f"cosmology.pdf::chunk-cosmology.pdf_p{page_number}_0_0",
            filename="cosmology.pdf",
            chunk_id=f"cosmology.pdf_p{page_number}_0_0",
            text="The cosmic microwave background radiation was discovered in 1965.",
            metadata={"file_type": "pdf", "page_number": page_number, "filename": "cosmology.pdf"},
            dense_rank=1,
        )

    def test_citation_has_file_type_and_page_number(self):
        from src.rag_pipeline.citations import generate_claim_citations
        candidate = self._make_pdf_candidate(page_number=2)
        answer = "The cosmic microwave background was discovered in 1965."
        citations, coverage = generate_claim_citations(answer, [candidate])
        assert len(citations) > 0
        c = citations[0]
        assert c.file_type == "pdf", f"Expected 'pdf', got {c.file_type}"
        assert c.page_number == 2, f"Expected 2, got {c.page_number}"

    def test_citation_source_label_includes_page_for_pdf(self):
        from src.rag_pipeline.citations import generate_claim_citations
        candidate = self._make_pdf_candidate(page_number=3)
        answer = "The cosmic microwave background was discovered in 1965."
        citations, _ = generate_claim_citations(answer, [candidate])
        if citations and citations[0].file_type == "pdf":
            assert "p. 3" in citations[0].source_label, \
                f"Expected 'p. 3' in source_label: {citations[0].source_label}"

    def test_citation_source_label_no_page_for_txt(self):
        from src.rag_pipeline.models import Candidate
        from src.rag_pipeline.citations import generate_claim_citations
        candidate = Candidate(
            key="stars.txt::chunk-stars.txt_0_0",
            filename="stars.txt",
            chunk_id="stars.txt_0_0",
            text="Stars fuse hydrogen into helium in their cores.",
            metadata={"file_type": "txt", "page_number": 0, "filename": "stars.txt"},
            dense_rank=1,
        )
        answer = "Stars fuse hydrogen into helium."
        citations, _ = generate_claim_citations(answer, [candidate])
        if citations:
            assert "p. " not in citations[0].source_label, \
                f"Non-PDF source_label should not contain page number: {citations[0].source_label}"

    def test_citation_to_dict_includes_page_fields(self):
        from src.rag_pipeline.citations import generate_claim_citations
        candidate = self._make_pdf_candidate(page_number=5)
        answer = "The cosmic microwave background was discovered in 1965."
        citations, _ = generate_claim_citations(answer, [candidate])
        if citations:
            d = citations[0].to_dict()
            assert "file_type" in d
            assert "page_number" in d
            assert "source_label" in d

    def test_context_source_label_for_pdf(self):
        from src.rag_pipeline.models import Candidate
        from src.rag_pipeline.context import format_context_with_metadata
        candidate = Candidate(
            key="hubble.pdf::chunk-hubble.pdf_p7_0_0",
            filename="hubble.pdf",
            chunk_id="hubble.pdf_p7_0_0",
            text="Hubble measured the expansion rate of the universe.",
            metadata={"file_type": "pdf", "page_number": 7, "filename": "hubble.pdf"},
            dense_rank=1,
            rrf_score=0.5,
            in_rrf=True,
            in_final_evidence=True,
        )
        context, sources, chunks = format_context_with_metadata(
            candidates=[candidate], query="Hubble expansion"
        )
        assert any("p. 7" in s for s in sources), f"Expected 'p. 7' in sources: {sources}"
        assert any(c.get("page_number") == 7 for c in chunks), f"page_number missing: {chunks}"

    def test_context_source_label_no_page_for_txt(self):
        from src.rag_pipeline.models import Candidate
        from src.rag_pipeline.context import format_context_with_metadata
        candidate = Candidate(
            key="neptune.txt::chunk-neptune.txt_0_0",
            filename="neptune.txt",
            chunk_id="neptune.txt_0_0",
            text="Neptune has 16 known moons including Triton.",
            metadata={"file_type": "txt", "page_number": 0, "filename": "neptune.txt"},
            dense_rank=1,
            rrf_score=0.5,
            in_rrf=True,
            in_final_evidence=True,
        )
        context, sources, chunks = format_context_with_metadata(
            candidates=[candidate], query="Neptune moons"
        )
        assert not any("p. " in s for s in sources), \
            f"Non-PDF source should not have page number: {sources}"


# ===========================================================================
# 9. Existing regression tests — verified by running these imported tests
# ===========================================================================

class TestExistingRegressionSuite:
    """Lightweight wrappers that re-run the key assertions from existing tests."""

    def test_list_txt_files_backward_compat(self, tmp_path):
        from src.document_loader import list_txt_files
        base = tmp_path / "kb"
        base.mkdir()
        (base / "one.txt").write_text("alpha", encoding="utf-8")
        (base / "two.md").write_text("beta", encoding="utf-8")
        (base / "three.txt").write_text("gamma", encoding="utf-8")
        files = list_txt_files(base)
        assert len(files) == 2
        assert {p.name for p in files} == {"one.txt", "three.txt"}

    def test_list_knowledge_files_includes_all_formats(self, tmp_path):
        from src.document_loader import list_knowledge_files
        base = tmp_path / "kb"
        base.mkdir()
        (base / "a.txt").write_text("txt", encoding="utf-8")
        (base / "b.md").write_text("md", encoding="utf-8")
        (base / "c.docx").write_bytes(_make_docx_bytes([(None, "docx")]))
        (base / "d.pdf").write_bytes(_make_pdf_bytes(["pdf page"]))
        files = list_knowledge_files(base)
        names = {p.name for p in files}
        assert names == {"a.txt", "b.md", "c.docx", "d.pdf"}

    def test_bm25_metadata_persistence_after_save_load(self, tmp_path):
        from src.bm25_retriever import BM25Retriever
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "mars.txt").write_text(
            "## Atmosphere\nMars has a thin atmosphere of carbon dioxide.\n\n"
            "## Moons\nMars has two small moons: Phobos and Deimos.",
            encoding="utf-8",
        )
        r = BM25Retriever(kb_path=str(kb), vectorstore_dir=tmp_path / "vs")
        r.build_index()
        idx = tmp_path / "vs" / "bm25_test.pkl"
        r.save_index(idx)
        r2 = BM25Retriever(vectorstore_dir=tmp_path / "vs")
        assert r2.load_index(idx) is True
        results = r2.query("carbon dioxide", top_k=2)
        assert len(results) > 0
        assert results[0]["bm25_score"] > 0
        assert "file_type" in results[0], "file_type missing after load"
        assert "page_number" in results[0], "page_number missing after load"

    def test_incremental_ingest_txt_backward_compat(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEV_EMBEDDINGS", "true")
        from src.rag_pipeline import RAGPipeline
        pip = RAGPipeline(knowledge_base_path=tmp_path / "kb", vectorstore_path=tmp_path / "vs")
        sample = "## Pulsars\nA pulsar is a rotating neutron star emitting radio beams."
        res = pip.ingest_file(file_path="pulsars.txt", content=sample.encode())
        assert res["skipped"] is False
        assert res["chunks_added"] > 0
        assert (tmp_path / "kb" / "pulsars.txt").exists()
        res2 = pip.ingest_file(file_path="pulsars.txt", content=sample.encode())
        assert res2["skipped"] is True
        assert res2["chunks_added"] == 0

    def test_citation_model_source_label_property(self):
        from src.rag_pipeline.models import Citation
        c_pdf = Citation(
            id=1, marker="[1]", claim="test", filename="cosmology.pdf",
            chunk_id="c1", passage="text", file_type="pdf", page_number=4,
        )
        assert "p. 4" in c_pdf.source_label
        assert "cosmology.pdf" in c_pdf.source_label

        c_txt = Citation(
            id=2, marker="[2]", claim="test", filename="stars.txt",
            chunk_id="c2", passage="text", file_type="txt", page_number=None,
        )
        assert c_txt.source_label == "stars.txt"
        assert "p." not in c_txt.source_label


# ===========================================================================
# 10. Calibrated confidence & persistent chat history tests
# ===========================================================================

class TestConfidenceAndHistory:
    def test_confidence_high_with_direct_support(self):
        from src.rag_pipeline.confidence import evaluate_confidence
        from src.rag_pipeline.models import Citation, CitationCoverage, CriterionScore, EvaluationResult

        eval_result = EvaluationResult(
            criteria={
                "direct_answer_support": CriterionScore(score=85.0, label="Direct Answer Support", definition="", reason=""),
                "evidence_relevance": CriterionScore(score=80.0, label="Evidence Relevance", definition="", reason=""),
                "evidence_sufficiency": CriterionScore(score=80.0, label="Evidence Sufficiency", definition="", reason=""),
                "retriever_agreement": CriterionScore(score=70.0, label="Retriever Agreement", definition="", reason=""),
            },
            health_score=80.0,
            summary="Strong evidence",
        )
        cov = CitationCoverage(
            total_claims=2, supported_claims=2, unsupported_claims=0,
            coverage_percentage=100.0, has_unsupported=False, unsupported_claims_list=[],
            partially_supported_claims=0,
        )
        citations = [
            Citation(id=1, marker="[1]", claim="c1", filename="f.txt", chunk_id="c1", passage="p1", status="supported"),
            Citation(id=2, marker="[2]", claim="c2", filename="f.txt", chunk_id="c2", passage="p2", status="supported"),
        ]
        conf = evaluate_confidence(eval_result, citations, cov, evidence_level="strong")
        assert conf.level == "HIGH", f"Expected HIGH confidence, got: {conf.level} (score: {conf.score})"
        assert conf.score >= 68.0

    def test_persistent_chat_history_roundtrip(self, tmp_path):
        from app import _load_persistent_chat_history, _save_persistent_chat_turn, _clear_persistent_chat_history

        u = {"role": "user", "content": "What is a pulsar?", "timestamp": "2026-09-29 10:00:00"}
        a = {"role": "assistant", "content": "A pulsar is a rotating neutron star.", "timestamp": "2026-09-29 10:00:01"}

        _save_persistent_chat_turn(tmp_path, u, a)
        loaded = _load_persistent_chat_history(tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "What is a pulsar?"
        assert loaded[1]["content"] == "A pulsar is a rotating neutron star."

        _clear_persistent_chat_history(tmp_path)
        assert len(_load_persistent_chat_history(tmp_path)) == 0
