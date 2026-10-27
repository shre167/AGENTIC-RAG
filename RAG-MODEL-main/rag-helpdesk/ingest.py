from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
from src.document_loader import list_knowledge_files
from src.rag_pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest knowledge base into vector store"
    )
    args = parser.parse_args()

    base_dir = Path(KNOWLEDGE_BASE_DIR)
    vector_dir = Path(VECTORSTORE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    files = list_knowledge_files(base_dir)
    if not files:
        print(
            "No supported knowledge files (.txt, .md, .docx, .pdf) were found in knowledge_base/. "
            "Add files there and run this command again."
        )
        return 0

    pipeline = RAGPipeline(
        knowledge_base_path=base_dir,
        vectorstore_path=vector_dir,
    )

    try:
        result = pipeline.ingest_documents()
    except Exception as exc:
        print(f"Ingestion error: {exc}")
        return 1

    skipped = result.get("chunks_skipped", 0)
    print(
        f"Indexed {result['chunks_indexed']} new chunks "
        f"({skipped} already present, "
        f"{result.get('chunks_total', result['chunks_indexed'] + skipped)} total) "
        f"from {result['files_processed']} files."
    )
    print(f"Collection count: {result['collection_count']}")
    print(f"Vector store saved to: {vector_dir}")
    print("To launch the assistant, run: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
