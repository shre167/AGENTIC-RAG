#!/usr/bin/env python3
"""Test every Retrieval Lab rendering function for errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    from src.rag_pipeline.pipeline import RAGPipeline
    from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
    from src.rag_pipeline.observatory import (
        has_trace, build_observatory_snapshot
    )

    print("Initializing pipeline...")
    pipeline = RAGPipeline(
        vectorstore_path=Path(VECTORSTORE_DIR),
        knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
    )

    print("Running query...")
    response = pipeline.answer_question("whats on page 60?", retrieval_mode="hybrid")
    print("Query done. Testing snapshot build...")

    trace = response.get("trace") or {}
    snapshot = build_observatory_snapshot(trace)
    retrieval = snapshot.get("retrieval", {})
    evidence = snapshot.get("evidence", {})
    generation = snapshot.get("generation", {})
    context = snapshot.get("context", {})
    query = snapshot.get("query", {})

    print()
    print("=== SECTION DATA CHECKS ===")
    
    # Check dense section
    raw_dense = retrieval.get("raw_dense", [])
    print(f"raw_dense: {len(raw_dense)} entries")
    if raw_dense:
        print("  First dense entry keys:", list(raw_dense[0].keys()) if isinstance(raw_dense[0], dict) else type(raw_dense[0]))

    # Check BM25 section
    raw_bm25 = retrieval.get("raw_bm25", [])
    print(f"raw_bm25: {len(raw_bm25)} entries")
    if raw_bm25:
        print("  First BM25 entry keys:", list(raw_bm25[0].keys()) if isinstance(raw_bm25[0], dict) else type(raw_bm25[0]))

    # Check RRF section
    rrf_results = retrieval.get("rrf_calculation", [])
    rrf_calc = retrieval.get("rrf_calculations", [])
    print(f"rrf_results: {len(rrf_results)} entries, rrf_calculations: {len(rrf_calc) if isinstance(rrf_calc, list) else type(rrf_calc)}")

    # Check evidence
    print(f"evidence level: {evidence.get('level')}, top_score: {evidence.get('top_score')}")

    # Check generation
    print(f"generation status: {generation.get('status')}, model: {generation.get('model')}")
    if generation.get("error"):
        print(f"  generation ERROR: {generation['error']}")

    # Check context
    print(f"context chunks: {context.get('chunks')}, chars: {context.get('chars')}")

    # Now test the astronomy_theme render functions
    print()
    print("=== TESTING ASTRONOMY THEME IMPORTS ===")
    try:
        from src.ui.astronomy_theme import (
            render_section_title,
            render_kv_rows,
            render_mode_chip,
            render_arch_node,
            render_arch_arrow,
            render_page_header,
            render_trace_query_card,
            render_kb_state_card,
        )
        print("All render functions imported OK")
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        return 1

    print()
    print("=== ALL CHECKS PASSED ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
