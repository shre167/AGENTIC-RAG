"""
PostgreSQL Vector Store WITHOUT pgvector extension.
Works with plain PostgreSQL - no extension needed!

Uses JSON column for embeddings and calculates similarity in Python.
"""
from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from sqlalchemy import Column, Integer, String, Float, Text, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rag_helpdesk"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

_ID_LOOKUP_BATCH_SIZE = 500


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when new embeddings are incompatible with collection."""
    pass


# ============================================================================
# SQLAlchemy Models - NO pgvector extension needed!
# ============================================================================

class ChunkRecord(Base):
    """SQLAlchemy model for storing chunks with embeddings as JSON."""
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, index=True)
    document_text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)  # Store as JSON, not vector
    
    # Metadata fields (all indexed)
    book_title = Column(String(255), nullable=True, index=True)
    book_id = Column(String(100), nullable=True, index=True)
    chapter_num = Column(Integer, nullable=True, index=True)
    chapter_title = Column(String(255), nullable=True)
    section_heading = Column(String(255), nullable=True)
    section_title = Column(String(255), nullable=True)
    source = Column(String(255), nullable=True, index=True)
    filename = Column(String(255), nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    has_images = Column(Integer, default=0)
    
    # Flexible JSON storage for additional metadata
    metadata_json = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ChunkRecord id={self.id} book={self.book_title}>"
    
    @property
    def embedding(self) -> List[float]:
        """Parse embedding from JSON."""
        return json.loads(self.embedding_json) if self.embedding_json else []


# ============================================================================
# Helper Functions
# ============================================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)


# ============================================================================
# Context Manager
# ============================================================================

from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# PostgreSQL Vector Store Implementation (No pgvector!)
# ============================================================================

class PostgresVectorStore:
    """Vector database using plain PostgreSQL (no pgvector extension needed!)."""
    
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str | Path = None):
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory or "./vectorstore"))
        
        # Create tables
        self._initialize_db()
        self._validate_collection()
    
    def _initialize_db(self) -> None:
        """Create tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to initialize database: {exc}")
            raise
    
    def _validate_collection(self) -> None:
        """Validate collection exists."""
        try:
            with get_db_session() as session:
                count = session.query(ChunkRecord).count()
                logger.info(f"PostgreSQL vector store initialized with {count} existing chunks")
        except Exception as exc:
            logger.warning(f"Could not validate collection: {exc}")
    
    def get_collection_embedding_dimension(self) -> Optional[int]:
        """Return the dimension of a stored embedding, or None if no usable
        embedding exists yet. Never assume 1536: this store supports any
        consistent embedding dimension, including local 384-dimensional models.
        """
        with get_db_session() as session:
            record = session.query(ChunkRecord).filter(
                ChunkRecord.embedding_json.isnot(None),
                ChunkRecord.embedding_json != "",
            ).first()

            if record is None:
                return None

            embedding = record.embedding
            return len(embedding) if embedding else None
    
    def _validate_new_embeddings(self, embeddings: List[List[float]]) -> None:
        """Ensure new embeddings are compatible."""
        if not embeddings:
            return
        
        new_dims = {len(e) for e in embeddings}
        if len(new_dims) != 1:
            raise ValueError(f"Inconsistent embedding dimensions: {sorted(new_dims)}")
        
        new_dim = next(iter(new_dims))
        existing_dim = self.get_collection_embedding_dimension()
        
        if existing_dim and existing_dim != new_dim:
            raise EmbeddingDimensionMismatchError(
                f"Embedding dimension mismatch: existing={existing_dim}, new={new_dim}"
            )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents with embeddings to the vector store."""
        if not texts:
            return
        
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        
        self._validate_new_embeddings(embeddings)
        
        try:
            with get_db_session() as session:
                for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                    record = ChunkRecord(
                        id=doc_id,
                        document_text=text,
                        embedding_json=json.dumps(embedding),
                        book_title=metadata.get("book_title"),
                        book_id=metadata.get("book_id"),
                        chapter_num=metadata.get("chapter_num"),
                        chapter_title=metadata.get("chapter_title"),
                        section_heading=metadata.get("section_heading"),
                        section_title=metadata.get("section_title"),
                        source=metadata.get("source", metadata.get("filename")),
                        filename=metadata.get("filename"),
                        page_start=metadata.get("page_start"),
                        page_end=metadata.get("page_end"),
                        token_count=metadata.get("token_count"),
                        has_images=1 if metadata.get("has_images") else 0,
                        metadata_json=json.dumps(metadata),
                    )
                    session.merge(record)
                session.commit()
                logger.info(f"Added {len(texts)} documents to PostgreSQL")
        except Exception as exc:
            logger.error(f"Error adding documents: {exc}")
            raise
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query similar chunks using cosine similarity (calculated in Python)."""
        try:
            with get_db_session() as session:
                # Get all records and calculate similarity in Python
                all_records = session.query(ChunkRecord).all()
                
                if not all_records:
                    return {"documents": [], "metadatas": [], "distances": [], "ids": []}
                
                # Calculate similarity for each record
                similarities = []
                for record in all_records:
                    emb = record.embedding
                    if emb and len(emb) == len(query_embedding):
                        sim = _cosine_similarity(query_embedding, emb)
                        similarities.append((record, sim))
                
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Take top n results
                top_results = similarities[:n_results]
                
                # Format results
                documents = [r.document_text for r, _ in top_results]
                ids = [r.id for r, _ in top_results]
                # Convert similarity to distance (1 - similarity)
                distances = [1 - sim for _, sim in top_results]
                metadatas = []
                
                for record, _ in top_results:
                    meta = {
                        "id": record.id,
                        "book_title": record.book_title,
                        "book_id": record.book_id,
                        "chapter_num": record.chapter_num,
                        "chapter_title": record.chapter_title,
                        "section_heading": record.section_heading,
                        "section_title": record.section_title,
                        "source": record.source,
                        "filename": record.filename,
                        "page_start": record.page_start,
                        "page_end": record.page_end,
                        "token_count": record.token_count,
                    }
                    metadatas.append(meta)
                
                return {
                    "documents": [documents],
                    "metadatas": [metadatas],
                    "distances": [distances],
                    "ids": [ids],
                }
        except Exception as exc:
            logger.error(f"Query error: {exc}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def get_chunks_by_page(
        self,
        page_start: int,
        page_end: int | None = None,
        book_title: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all chunks covering the specified page or page range."""
        target_end = page_end if page_end is not None else page_start
        try:
            with get_db_session() as session:
                query = session.query(ChunkRecord).filter(
                    ChunkRecord.page_start <= target_end,
                    ChunkRecord.page_end >= page_start,
                )
                if book_title:
                    from sqlalchemy import func
                    query = query.filter(
                        func.lower(ChunkRecord.book_title).contains(book_title.lower())
                    )
                records = query.order_by(
                    ChunkRecord.book_title,
                    ChunkRecord.page_start,
                    ChunkRecord.id,
                ).all()
                results = []
                for r in records:
                    results.append({
                        "id": r.id,
                        "text": r.document_text,
                        "metadata": {
                            "id": r.id,
                            "chunk_id": r.id,
                            "book_title": r.book_title,
                            "book_id": r.book_id,
                            "chapter_num": r.chapter_num,
                            "chapter_title": r.chapter_title,
                            "section_heading": r.section_heading,
                            "section_title": r.section_title,
                            "source": r.source,
                            "filename": r.filename,
                            "page_start": r.page_start,
                            "page_end": r.page_end,
                            "token_count": r.token_count,
                        }
                    })
                return results
        except Exception as exc:
            logger.warning(f"Error querying postgres by page: {exc}")
            return []

    def delete_by_ids(self, ids: List[str]) -> int:
        """Delete chunks by ID from PostgreSQL."""
        if not ids:
            return 0
        try:
            with get_db_session() as session:
                deleted = session.query(ChunkRecord).filter(
                    ChunkRecord.id.in_(ids)
                ).delete(synchronize_session=False)
                session.commit()
                logger.info(f"PostgreSQL: deleted {deleted} chunks by ID")
                return deleted
        except Exception as exc:
            logger.error(f"Error deleting chunks by ID from PostgreSQL: {exc}")
            return 0

    def delete_file(self, filename: str) -> int:
        """Delete all chunks belonging to a filename or source from PostgreSQL."""
        if not filename:
            return 0
        fname = Path(filename).name
        try:
            with get_db_session() as session:
                deleted = session.query(ChunkRecord).filter(
                    (ChunkRecord.filename == fname) |
                    (ChunkRecord.source == fname) |
                    (ChunkRecord.book_id == fname) |
                    (func.lower(ChunkRecord.book_title) == fname.lower())
                ).delete(synchronize_session=False)
                session.commit()
                logger.info(f"PostgreSQL: deleted {deleted} chunks for file {fname}")
                return deleted
        except Exception as exc:
            logger.error(f"Error deleting file {fname} from PostgreSQL: {exc}")
            return 0

    def delete(self, ids: List[str] | None = None, where: Dict[str, Any] | None = None) -> int:
        """Chroma-compatible delete API on PostgreSQL store."""
        if ids:
            return self.delete_by_ids(ids)
        if where:
            filename = where.get("filename") or where.get("source")
            if filename:
                return self.delete_file(filename)
        return 0

    def delete_collection(self) -> None:
        """Delete all chunks from collection."""
        try:
            with get_db_session() as session:
                session.query(ChunkRecord).delete()
                session.commit()
                logger.info("Cleared all chunks from PostgreSQL")
        except Exception as exc:
            logger.error(f"Error clearing collection: {exc}")
            raise
    
    def clear_collection(self) -> None:
        """Remove every document without recreating the table."""
        self.delete_collection()
    
    def get_collection_count(self) -> int:
        """Return total number of chunyks in collection."""
        try:
            with get_db_session() as session:
                return session.query(ChunkRecord).count()
        except Exception:
            return 0
    
    def collection_exists(self) -> bool:
        """Return True if collection is accessible."""
        try:
            with get_db_session() as session:
                session.query(ChunkRecord).count()
            return True
        except Exception:
            return False
    
    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        """Return subset of ids that already exist."""
        if not ids:
            return set()
        
        existing: Set[str] = set()
        try:
            with get_db_session() as session:
                for start in range(0, len(ids), _ID_LOOKUP_BATCH_SIZE):
                    batch = ids[start:start + _ID_LOOKUP_BATCH_SIZE]
                    result = session.query(ChunkRecord.id).filter(
                        ChunkRecord.id.in_(batch)
                    ).all()
                    existing.update(r[0] for r in result)
        except Exception as exc:
            logger.warning(f"Error checking existing IDs: {exc}")
        
        return existing
    
    def get_diagnostics(self, **kwargs) -> Dict[str, Any]:
        """Return vector store diagnostics."""
        try:
            db_url = urlparse(DATABASE_URL)
            db_host = db_url.hostname or "unknown"
        except Exception:
            db_host = "unknown"
        
        existing_dim = self.get_collection_embedding_dimension()
        count = self.get_collection_count()
        
        return {
            "backend": "PostgreSQL (no pgvector)",
            "collection_name": self.collection_name,
            "collection_exists": self.collection_exists(),
            "collection_count": count,
            "existing_embedding_dimension": existing_dim,
            "database_host": db_host,
            "vector_similarity": "cosine (Python)",
        }
    
    @property
    def index(self):
        return self if self.get_collection_count() > 0 else None
    
    @property
    def collection(self):
        return self
