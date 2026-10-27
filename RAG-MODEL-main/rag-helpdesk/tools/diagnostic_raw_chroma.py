from __future__ import annotations

import os
import sys
from pathlib import Path

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config, EMBEDDING_MODEL, VECTORSTORE_DIR
from src.vector_store import ChromaVectorStore
from src.embeddings import EmbeddingService
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline.evidence import evaluate_evidence
from src.rag_pipeline.retrieval import dense_retrieve

def run_diagnostic():
    print("=" * 80)
    print("RAW CHROMA RETRIEVAL & EVIDENCE GATE DIAGNOSTIC")
    print("=" * 80)

    store = ChromaVectorStore(
        collection_name="knowledge_base",
        persist_directory=VECTORSTORE_DIR,
    )
    col_count = store.get_collection_count()
    stored_dim = store.get_collection_embedding_dimension()

    emb_svc = EmbeddingService(
        api_key=Config.GE_API_KEY,
        model=EMBEDDING_MODEL,
        base_url=Config.LLM_BASE_URL,
    )

    print(f"Chroma Path: {VECTORSTORE_DIR.resolve()}")
    print(f"Collection Name: knowledge_base")
    print(f"Collection Count: {col_count}")
    print(f"Stored Embedding Dimension: {stored_dim}")
    print(f"Configured Embedding Model: {EMBEDDING_MODEL}")
    print()

    queries = [
        "Whats the cosmos and how are stars formed",
        "Are aliens real",
        "How are stars formed",
        "What is the universe",
        "What are galaxies",
        "What is Mars",
        "What is Chandrayaan-3",
    ]

    import time
    for q in queries:
        time.sleep(2)
        print("-" * 80)
        print(f"QUERY: {q}")
        print("-" * 80)

        q_vec = emb_svc.create_embedding(q)
        q_dim = len(q_vec) if q_vec else 0
        print(f"Embedding model actually used: {emb_svc.model}")
        print(f"Embedding dimension: {q_dim}")
        print(f"Chroma path actually used: {VECTORSTORE_DIR.resolve()}")
        print(f"Collection actually used: knowledge_base")
        print(f"Collection count: {col_count}")
        print()

        if not q_vec:
            print("ERROR: Failed to generate query embedding!")
            continue

        raw_results = store.query(q_vec, n_results=10)
        docs = (raw_results.get("documents") or [[]])[0]
        metas = (raw_results.get("metadatas") or [[]])[0]
        dists = (raw_results.get("distances") or [[]])[0]
        ids = (raw_results.get("ids") or [[]])[0]

        print(f"RAW RESULTS: {len(docs)}")
        for idx in range(len(docs)):
            m = metas[idx] if idx < len(metas) else {}
            src = m.get("filename") or m.get("source") or "unknown"
            cid = ids[idx] if idx < len(ids) else "unknown"
            d = dists[idx] if idx < len(dists) else -1.0
            doc_text = docs[idx] if idx < len(docs) else ""
            clean_text = doc_text.replace("\n", " ").strip()[:200]
            print(f"{idx + 1}.")
            print(f"Source: {src}")
            print(f"Chunk ID: {cid}")
            print(f"Distance: {d:.4f}")
            print(f"Text: {clean_text}...")
            print()

        # Now test how Dense Retrieval and Evidence Gate evaluate these candidates
        candidates_dict = dense_retrieve(store, q_vec, top_k=10)
        ranked_candidates = sorted(
            candidates_dict.values(),
            key=lambda c: (c.dense_rank if c.dense_rank is not None else 9999),
        )

        final_cands, evidence = evaluate_evidence(
            ranked_candidates,
            reranked=False,
            search_text=q,
            retrieval_mode="vector",
        )

        print(f"EVIDENCE GATE EVALUATION (mode='vector'):")
        print(f"  Level: {evidence.level}")
        print(f"  Should Answer: {evidence.should_answer}")
        print(f"  Top Score (distance): {evidence.top_score}")
        print(f"  Supporting Chunks: {evidence.supporting_chunks}")
        print(f"  Reason: {evidence.reason}")
        print(f"  Selected Candidates Count: {len(final_cands)}")
        if final_cands:
            for fc in final_cands:
                print(f"    - {fc.filename} (dense_rank={fc.dense_rank}, dist={fc.dense_distance:.4f})")
        print()

if __name__ == "__main__":
    run_diagnostic()
