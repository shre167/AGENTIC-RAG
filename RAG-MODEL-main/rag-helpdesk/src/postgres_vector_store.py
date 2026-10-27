"""
PostgreSQL Vector Store using pgvector for RAG pipeline.

Drop-in replacement for ChromaVectorStore that uses PostgreSQL with pgvector extension.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from sqlalchemy import Column, Integer, String, Float, Text, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/rag_helpdesk"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

_ID_LOOKUP_BATCH_SIZE = 500
EMBEDDING_DIMENSION = 1536  # Default for most models (adjust as needed)


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when new embeddings are incompatible with collection."""
    pass


# ============================================================================
# SQLAlchemy Models
# ============================================================================

class ChunkRecord(Base):
    """SQLAlchemy model for storing chunks with embeddings."""
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, index=True)
    document_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False, index=True)
    
    # Metadata fields
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


# ============================================================================
# Context Manager for Database Sessions
# ============================================================================

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
# PostgreSQL Vector Store Implementation
# ============================================================================

class PostgresVectorStore:
    """Vector database manager using PostgreSQL with pgvector."""
    
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str | Path = None):
        """
        Initialize PostgreSQL vector store.
        
        Args:
            collection_name: Not used (PostgreSQL uses tables instead)
            persist_directory: Not used (PostgreSQL handles persistence)
        """
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory or "./vectorstore"))
        
        # Create tables
        self._initialize_db()
        
        # Validate embedding dimension
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
        """Validate collection exists and is accessible."""
        try:
            with get_db_session() as session:
                count = session.query(ChunkRecord).count()
                logger.info(f"PostgreSQL vector store initialized with {count} existing chunks")
        except Exception as exc:
            logger.warning(f"Could not validate collection: {exc}")
    
    def get_collection_embedding_dimension(self) -> Optional[int]:
        """Return the embedding dimension of stored vectors."""
        try:
            with get_db_session() as session:
                count = session.query(ChunkRecord).count()
                if count == 0:
                    return None
                record = session.query(ChunkRecord).first()
                if record and record.embedding:
                    return len(record.embedding)
        except Exception as exc:
            logger.warning(f"Could not read embedding dimension: {exc}")
        return EMBEDDING_DIMENSION
    
    def _validate_new_embeddings(self, embeddings: List[List[float]]) -> None:
        """Ensure new embeddings are compatible with existing collection."""
        if not embeddings:
            return
        
        new_dims = {len(e) for e in embeddings}
        if len(new_dims) != 1:
            raise ValueError(f"Inconsistent embedding dimensions: {sorted(new_dims)}")
        
        new_dim = next(iter(new_dims))
        existing_dim = self.get_collection_embedding_dimension()
        
        if existing_dim and existing_dim != new_dim:
            raise EmbeddingDimensionMismatchError(
                f"Embedding dimension mismatch: existing={existing_dim}, new={new_dim}. "
                "Ensure EMBEDDING_MODEL matches previous configuration."
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
                        embedding=embedding,
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
                        metadata_json=str(metadata),
                    )
                    # Use merge to handle upserts
                    session.merge(record)
                session.commit()
                logger.info(f"Added {len(texts)} documents to PostgreSQL")
        except Exception as exc:
            logger.error(f"Error adding documents: {exc}")
            raise
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query similar chunks using vector similarity."""
        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise EmbeddingDimensionMismatchError(
                f"Query embedding dimension {len(query_embedding)} != {EMBEDDING_DIMENSION}"
            )
        
        try:
            with get_db_session() as session:
                # Use cosine similarity for nearest neighbors
                results = session.query(ChunkRecord).order_by(
                    ChunkRecord.embedding.cosine_distance(query_embedding)
                ).limit(n_results).all()
                
                # Format results to match Chroma interface
                documents = [r.document_text for r in results]
                ids = [r.id for r in results]
                distances = []  # Cosine distance (0 = identical, 1 = opposite)
                metadatas = []
                
                for record in results:
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
        """Return total number of chunks in collection."""
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
        """Return subset of ids that already exist in collection."""
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
    
    def get_diagnostics(
        self,
        *,
        configured_model: Optional[str] = None,
        configured_base_url: Optional[str] = None,
        current_embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return vector store diagnostics."""
        try:
            db_url = urlparse(DATABASE_URL)
            db_host = db_url.hostname or "unknown"
        except Exception:
            db_host = "unknown"
        
        existing_dim = self.get_collection_embedding_dimension()
        count = self.get_collection_count()
        
        dimensions_compatible: Optional[bool] = None
        if existing_dim and current_embedding_dim:
            dimensions_compatible = existing_dim == current_embedding_dim
        
        return {
            "backend": "PostgreSQL + pgvector",
            "collection_name": self.collection_name,
            "collection_exists": self.collection_exists(),
            "collection_count": count,
            "existing_embedding_dimension": existing_dim,
            "current_embedding_dimension": current_embedding_dim,
            "dimensions_compatible": dimensions_compatible,
            "configured_embedding_model": configured_model,
            "database_host": db_host,
        }
    
    @property
    def index(self) -> Optional[Any]:
        """Compatibility property (not used in PostgreSQL)."""
        return self if self.get_collection_count() > 0 else None
    
    @property
    def collection(self) -> Any:
        """Compatibility property returning self."""
        return self
