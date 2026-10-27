"""
Migrate data from Chroma to plain PostgreSQL WITHOUT pgvector.

Requirements:
    - PostgreSQL server running
    - Database exists
    - Valid DATABASE_URL in .env
    - USE_POSTGRES=true in .env
    - postgres_vector_store_simple.py available in src/
"""

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Load the project's .env file.
load_dotenv(BASE_DIR / ".env", override=False)

# Ensure project root is importable.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def migrate_chroma_to_postgres():
    """Migrate all Chroma chunks to PostgreSQL without pgvector."""

    print("=" * 60)
    print("CHROMA TO POSTGRESQL MIGRATION")
    print("Backend: Plain PostgreSQL - NO pgvector")
    print("=" * 60)

    # Validate configuration.
    use_postgres = os.getenv(
        "USE_POSTGRES", "false"
    ).strip().lower()

    if use_postgres not in ("true", "1", "yes"):
        print("ERROR: PostgreSQL is not enabled.")
        print("Set USE_POSTGRES=true in your .env file.")
        return False

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        print("ERROR: DATABASE_URL is missing from .env.")
        return False

    try:
        # -------------------------------------------------
        # 1. Connect to Chroma
        # -------------------------------------------------
        from src.vector_store import ChromaVectorStore
        from src.config import VECTORSTORE_DIR

        print("\n[1/4] Connecting to Chroma...")

        chroma = ChromaVectorStore(
            persist_directory=VECTORSTORE_DIR
        )

        chroma_count = chroma.get_collection_count()

        if chroma_count == 0:
            print("ERROR: No chunks found in Chroma.")
            return False

        print(f"Found {chroma_count} chunks in Chroma.")

        # -------------------------------------------------
        # 2. Fetch Chroma data
        # -------------------------------------------------
        print("\n[2/4] Fetching documents and embeddings...")

        data = chroma.collection.get(
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ]
        )

        ids = data.get("ids") or []
        documents = data.get("documents")
        metadatas = data.get("metadatas")
        embeddings = data.get("embeddings")

        if not ids:
            print("ERROR: Chroma returned no IDs.")
            return False

        if (
            documents is None
            or metadatas is None
            or embeddings is None
        ):
            print("ERROR: Chroma returned incomplete data.")
            return False

        total = len(ids)

        if not (
            len(documents) == total
            and len(metadatas) == total
            and len(embeddings) == total
        ):
            raise ValueError(
                "The number of IDs, documents, metadata entries "
                "and embeddings does not match."
            )

        # Validate embedding dimensions without changing them.
        dimensions = {
            len(embedding)
            for embedding in embeddings
            if embedding is not None
        }

        if len(dimensions) != 1:
            raise ValueError(
                f"Inconsistent embedding dimensions: {dimensions}"
            )

        embedding_dimension = next(iter(dimensions))

        print(f"Records retrieved: {total}")
        print(f"Embedding dimension: {embedding_dimension}")

        # -------------------------------------------------
        # 3. Connect to plain PostgreSQL
        # -------------------------------------------------
        print("\n[3/4] Connecting to plain PostgreSQL...")

        # IMPORTANT:
        # This imports the implementation that stores embeddings
        # as JSON and does not import pgvector.
        from src.postgres_vector_store_simple import (
            PostgresVectorStore,
        )

        postgres = PostgresVectorStore()

        print("PostgreSQL connection initialized.")
        print("Using JSON embeddings; pgvector is not required.")

        # Check for incompatible pre-existing schema/data.
        # The simple store expects an embedding_json column.
        from sqlalchemy import inspect

        inspector = inspect(postgres.engine) if hasattr(
            postgres, "engine"
        ) else None

        # The simple implementation may keep its engine at module
        # level rather than on the instance.
        if inspector is None:
            from src import postgres_vector_store_simple as simple_store

            inspector = inspect(simple_store.engine)

        if inspector.has_table("chunks"):
            columns = {
                column["name"]
                for column in inspector.get_columns("chunks")
            }

            if "embedding_json" not in columns:
                raise RuntimeError(
                    "The existing 'chunks' table does not contain "
                    "'embedding_json'. Its schema may belong to the "
                    "old pgvector implementation. Migration stopped "
                    "to protect existing data. Do not drop the table "
                    "without first backing it up and reviewing its schema."
                )

        # -------------------------------------------------
        # 4. Migrate in batches
        # -------------------------------------------------
        print("\n[4/4] Migrating records...")

        batch_size = 100
        migrated = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_ids = ids[start:end]
            batch_documents = documents[start:end]
            batch_metadatas = metadatas[start:end]
            batch_embeddings = embeddings[start:end]

            # Check each batch before writing.
            batch_lengths = {
                len(batch_ids),
                len(batch_documents),
                len(batch_metadatas),
                len(batch_embeddings),
            }

            if len(batch_lengths) != 1:
                raise ValueError(
                    f"Batch {start}:{end} contains mismatched data."
                )

            # Preserve original Chroma IDs, text, metadata and
            # embedding values. Do not regenerate embeddings.
            postgres.add_documents(
                texts=batch_documents,
                embeddings=[
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                    for embedding in batch_embeddings
                ],
                metadatas=[
                    metadata or {}
                    for metadata in batch_metadatas
                ],
                ids=batch_ids,
            )

            migrated += len(batch_ids)
            print(f"  Processed {migrated}/{total} chunks.")

        # -------------------------------------------------
        # Verify destination
        # -------------------------------------------------
        postgres_count = postgres.get_collection_count()

        print("\n" + "=" * 60)
        print("MIGRATION VERIFICATION")
        print("=" * 60)
        print(f"Chroma source count:      {chroma_count}")
        print(f"Records processed:        {migrated}")
        print(f"PostgreSQL destination:   {postgres_count}")
        print(f"Embedding dimension:      {embedding_dimension}")

        if postgres_count != chroma_count:
            print("\nWARNING: Source and destination counts differ.")
            print(
                "Check for pre-existing records or duplicate IDs "
                "before attempting another migration."
            )
            return False

        print("\nSUCCESS: Source and destination counts match.")
        print("Original Chroma data has not been deleted.")
        print(
            "Test PostgreSQL retrieval before switching your "
            "application to the new backend."
        )

        return True

    except Exception as exc:
        print(f"\nMIGRATION FAILED: {exc}")
        traceback.print_exc()
        return False


def main():
    print("Chroma -> PostgreSQL migration")
    print("Storage: JSON embeddings; no pgvector extension.")
    print()
    print("Prerequisites:")
    print("  1. PostgreSQL is running.")
    print("  2. Database 'rag_helpdesk' exists.")
    print("  3. DATABASE_URL contains valid credentials.")
    print("  4. USE_POSTGRES=true is set in .env.")
    print("  5. The destination table uses the simple-store schema.")
    print()
    print("WARNING:")
    print("  - Existing destination records may be overwritten by ID.")
    print("  - Chroma data will not be deleted by this script.")
    print("  - An incompatible existing schema will stop migration.")
    print()

    response = input(
        "Are you ready to migrate? (yes/no): "
    ).strip().lower()

    if response != "yes":
        print("Migration cancelled.")
        return

    success = migrate_chroma_to_postgres()

    if success:
        print("\nMigration completed successfully.")
    else:
        print("\nMigration did not complete successfully.")


if __name__ == "__main__":
    main()