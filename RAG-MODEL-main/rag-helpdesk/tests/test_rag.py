from pathlib import Path
import os

from src.document_loader import list_txt_files, list_knowledge_files
from src.rag_pipeline import RAGPipeline, ABSTAIN_MESSAGE


def test_list_txt_files_finds_txt_files(tmp_path):
    base = tmp_path / "knowledge_base"
    base.mkdir()
    (base / "one.txt").write_text("alpha", encoding="utf-8")
    (base / "two.md").write_text("beta", encoding="utf-8")
    (base / "three.txt").write_text("gamma", encoding="utf-8")

    files = list_txt_files(base)

    assert len(files) == 2
    assert {p.name for p in files} == {"one.txt", "three.txt"}

    # list_knowledge_files finds both .txt and .md
    kb_files = list_knowledge_files(base)
    assert len(kb_files) == 3


def test_rag_pipeline_handles_missing_information_without_hallucinating(monkeypatch, tmp_path):
    # Use synthetic embeddings to ensure test runs deterministically offline
    monkeypatch.setenv("DEV_EMBEDDINGS", "true")

    pip = RAGPipeline(
        knowledge_base_path=Path("knowledge_base"),
        vectorstore_path=tmp_path / "test_vs",
        top_k=3,
        score_threshold=0.1,
    )

    result = pip.answer_question("What is the procedure for an unrelated topic that does not exist?")

    # The canonical abstain message indicates insufficient information
    answer_lower = result["answer"].lower()
    assert (
        "couldn't find" in answer_lower
        or "not found" in answer_lower
        or "insufficient" in answer_lower
        or "relevant information" in answer_lower
    )
    assert result["sources"] == []


def test_query_normalizer_expanded_query():
    from src.query_normalizer import canonicalize_query

    # Backwards-compatibility test for legacy queries
    q = "How do I submit my timesheet in Replicon when on sick leave?"
    res = canonicalize_query(q)
    assert res["canonical"] == "replicon"
    assert "submit" in res["expanded_query"]
    assert "timesheet" in res["expanded_query"]
    assert "sick leave" in res["expanded_query"]
    assert res["expanded_query"].endswith("replicon")

    # Generic query with no alias
    res2 = canonicalize_query("What are the office opening hours?")
    assert res2["canonical"] is None
    assert res2["expanded_query"] == "what are the office opening hours"


def test_astronomy_query_normalization():
    from src.query_normalizer import canonicalize_query

    # Test astronomy aliases
    res_jwst = canonicalize_query("What instruments are on the James Webb Space Telescope?")
    assert res_jwst["canonical"] == "jwst"
    assert "jwst" in res_jwst["expanded_query"]

    res_chandrayaan = canonicalize_query("Where did Pragyan rover land on the Moon?")
    assert res_chandrayaan["canonical"] == "chandrayaan"
    assert "chandrayaan" in res_chandrayaan["expanded_query"]

    res_blackhole = canonicalize_query("What happens at the event horizon of a singularity?")
    assert res_blackhole["canonical"] == "black hole"
    assert "black hole" in res_blackhole["expanded_query"]


def test_bm25_metadata_and_persistence(tmp_path):
    from src.bm25_retriever import BM25Retriever

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "mars.txt").write_text(
        "## Atmosphere\nMars has a thin atmosphere consisting mostly of carbon dioxide.\n\n"
        "## Moons\nMars has two small moons, Phobos and Deimos.",
        encoding="utf-8",
    )

    retriever = BM25Retriever(
        kb_path=str(kb_dir),
        chunk_size=400,
        chunk_overlap=50,
        vectorstore_dir=tmp_path / "vs",
    )
    retriever.build_index()

    results = retriever.query("carbon dioxide atmosphere", top_k=2)
    assert len(results) > 0
    assert "mars.txt" in results[0]["filename"]
    assert "section_heading" in results[0]
    assert results[0]["bm25_score"] > 0

    # Test persistence
    index_file = tmp_path / "vs" / "bm25_test.pkl"
    retriever.save_index(index_file)
    assert index_file.exists()

    new_retriever = BM25Retriever(vectorstore_dir=tmp_path / "vs")
    loaded = new_retriever.load_index(index_file)
    assert loaded is True
    assert len(new_retriever.query("carbon dioxide", top_k=1)) > 0


def test_incremental_ingest_file_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("DEV_EMBEDDINGS", "true")

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    vs_dir = tmp_path / "vs"
    vs_dir.mkdir()

    pip = RAGPipeline(
        knowledge_base_path=kb_dir,
        vectorstore_path=vs_dir,
    )

    # Ingest a new file
    sample_content = (
        "## Pulsars\n"
        "A pulsar is a highly magnetized rotating neutron star that emits beams of "
        "electromagnetic radiation out of its magnetic poles."
    )
    res = pip.ingest_file(file_path="pulsars.txt", content=sample_content)
    assert res["skipped"] is False
    assert res["chunks_added"] > 0
    assert (kb_dir / "pulsars.txt").exists()

    # Ingest the same file again without changes -> should be skipped
    res_repeat = pip.ingest_file(file_path="pulsars.txt", content=sample_content)
    assert res_repeat["skipped"] is True
    assert res_repeat["chunks_added"] == 0


def test_vector_store_query_pass_through(tmp_path):
    from unittest.mock import MagicMock
    from src.vector_store import ChromaVectorStore

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["text1", "text2"]],
        "metadatas": [[{"filename": "doc1.txt"}, {"filename": "doc2.txt"}]],
        "distances": [[0.1, 0.4]],
    }

    store = ChromaVectorStore(persist_directory=str(tmp_path / "test_store"))
    store.collection = mock_collection

    res = store.query([0.1, 0.2, 0.3], n_results=5)
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=5,
    )
    assert len(res["documents"][0]) == 2
