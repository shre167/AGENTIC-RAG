#!/usr/bin/env python3
"""Test Retrieval Lab trace pipeline."""
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

    print()
    print("=== TRACE CHECK ===")
    print("has_trace:", has_trace(response))
    print("trace keys:", list((response.get("trace") or {}).keys()))

    if not has_trace(response):
        print("ERROR: No trace found in response")
        if response.get("error"):
            print("Error:", response["error"])
        return 1

    print()
    print("=== BUILDING SNAPSHOT ===")
    try:
        snapshot = build_observatory_snapshot(response["trace"])
        print("Snapshot keys:", list(snapshot.keys()))
        print()

        q = snapshot.get("query", {})
        print("Query section keys:", list(q.keys()))
        print("  original_query:", q.get("original_query", "MISSING"))
        print("  prepared_query:", q.get("prepared_query", "MISSING"))

        r = snapshot.get("retrieval", {})
        print("Retrieval section keys:", list(r.keys()))
        print("  mode_label:", r.get("mode_label", "MISSING"))
        print("  dense_enabled:", r.get("dense_enabled", "MISSING"))
        print("  bm25_enabled:", r.get("bm25_enabled", "MISSING"))
        print("  candidate_count:", r.get("candidate_count", "MISSING"))

        e = snapshot.get("evidence", {})
        print("Evidence section keys:", list(e.keys()))

        g = snapshot.get("generation", {})
        print("Generation section keys:", list(g.keys()))
        print("  status:", g.get("status", "MISSING"))

        print()
        print("SUCCESS: Retrieval Lab trace is complete")
        return 0

    except Exception as ex:
        print(f"SNAPSHOT BUILD ERROR: {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
