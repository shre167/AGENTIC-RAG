"""Safe vectorstore inspection — no API keys printed."""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EMBEDDING_MODEL, LLM_BASE_URL, VECTORSTORE_DIR
from src.vector_store import ChromaVectorStore


def _provider_label(base_url: str) -> str:
    if not base_url:
        return "unknown"
    host = urlparse(base_url).netloc or base_url
    return host


def main() -> int:
    dev = os.getenv("DEV_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
    print("=== Embedding configuration ===")
    print(f"provider_host: {_provider_label(LLM_BASE_URL)}")
    print(f"embedding_model: {EMBEDDING_MODEL or '(not set)'}")
    print(f"dev_embeddings: {dev}")

    vs = ChromaVectorStore(persist_directory=VECTORSTORE_DIR)
    count = vs.get_collection_count()
    existing_dim = vs.get_collection_embedding_dimension()
    print("\n=== Collection ===")
    print(f"collection_name: {vs.collection_name}")
    print(f"collection_count: {count}")
    print(f"existing_embedding_dimension: {existing_dim}")

    if count > 0:
        sample = vs.collection.get(limit=min(5, count), include=["metadatas"])
        ids = sample.get("ids") or []
        print(f"sample_ids ({len(ids)}): {ids[:3]}")

    print("\n=== Current embedding dimension (test) ===")
    current_dim: int | None = None
    if dev:
        from src.embeddings import SyntheticEmbeddingService

        svc = SyntheticEmbeddingService()
        test = svc.create_embedding("dimension probe")
        current_dim = len(test)
        print(f"current_source: synthetic (DEV_EMBEDDINGS)")
        print(f"current_embedding_dim: {current_dim}")
    else:
        from src.embeddings import EmbeddingService

        svc = EmbeddingService()
        try:
            test = svc.create_embedding("dimension probe")
            if test:
                current_dim = len(test)
                print("current_source: configured API")
                print(f"current_embedding_dim: {current_dim}")
            else:
                print("current_source: configured API (test embedding returned None)")
        except Exception as exc:
            print(f"current_source: configured API (test failed: {exc})")

    diagnostics = vs.get_diagnostics(
        configured_model=EMBEDDING_MODEL,
        configured_base_url=LLM_BASE_URL,
        dev_embeddings=dev,
        current_embedding_dim=current_dim,
    )
    print("\n=== Diagnostics summary ===")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    db_path = Path(VECTORSTORE_DIR) / "chroma.sqlite3"
    if db_path.exists():
        print(f"\n=== SQLite audit ({db_path.name}, {db_path.stat().st_size} bytes) ===")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for table in sorted(tables):
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                print(f"  {table}: {cur.fetchone()[0]} rows")
            except Exception as exc:
                print(f"  {table}: error ({exc})")
        for query in ("SELECT id, name FROM collections", "SELECT id, type, scope, collection FROM segments"):
            try:
                cur.execute(query)
                rows = cur.fetchall()
                print(f"  {query}: {rows}")
            except Exception as exc:
                print(f"  {query}: error ({exc})")
        conn.close()

    seg_dirs = [p for p in Path(VECTORSTORE_DIR).iterdir() if p.is_dir()]
    print(f"\n=== Segment directories ({len(seg_dirs)}) ===")
    for seg in seg_dirs:
        files = list(seg.glob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file())
        print(f"  {seg.name}: {len(files)} files, {total} bytes")

    kb_files = list(Path(VECTORSTORE_DIR).parent.joinpath("knowledge_base").glob("*"))
    kb_count = len([f for f in kb_files if f.suffix.lower() in (".txt", ".md")])
    print(f"\n=== Source documents ===")
    print(f"knowledge_base_files: {kb_count}")
    print(f"vector_chunks_in_chroma: {count}")
    if kb_count > count:
        print("NOTE: source documents outnumber stored vector chunks.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
