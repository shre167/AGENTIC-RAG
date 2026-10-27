from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMADB = True
    _CHROMADB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency fallback
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    _HAS_CHROMADB = False
    _CHROMADB_IMPORT_ERROR = exc

from src.config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_ID_LOOKUP_BATCH_SIZE = 500


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when new embeddings are incompatible with an existing collection."""


def _embedding_dimension(embedding: List[float]) -> int:
    return len(embedding)


def _is_dimension_mismatch_error(exc: Exception) -> bool:
    return "dimension" in str(exc).lower()


class ChromaVectorStore:
    """Vector database manager using ChromaDB."""

    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str | Path = VECTORSTORE_DIR):
        if not _HAS_CHROMADB:
            raise ImportError(
                "ChromaDB could not be imported. Install project requirements with "
                "`pip install -r requirements.txt`. Original import error: "
                f"{_CHROMADB_IMPORT_ERROR}"
            ) from _CHROMADB_IMPORT_ERROR
        base = Path(persist_directory).resolve()
        # chromadb 1.x PersistentClient expects a plain directory and creates
        # chroma.sqlite3 inside it.  Use a dedicated subdirectory so it never
        # conflicts with other files (bm25 index, manifests, etc.) stored in
        # the parent vectorstore folder.
        chroma_dir = base / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.persist_directory = str(chroma_dir)
        if hasattr(chromadb, "PersistentClient"):
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False) if Settings else None,
            )
        else:
            self.client = chromadb.Client(Settings(persist_directory=self.persist_directory, anonymized_telemetry=False))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        # expose `index` for compatibility with earlier vector-store implementations
        # return the collection object when there are any items, else None
        self._index_cached: Optional[Any] = None

    def get_collection_embedding_dimension(self) -> Optional[int]:
        """Return the embedding dimension of one stored vector, if any."""
        if self.get_collection_count() == 0:
            return None
        try:
            sample = self.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                return None
            first = embeddings[0]
            if first is None:
                return None
            return _embedding_dimension(list(first))
        except Exception as exc:
            logger.warning("Could not read collection embedding dimension: %s", exc)
            return None

    def _validate_new_embeddings(self, embeddings: List[List[float]]) -> None:
        """Ensure new embeddings are compatible with the existing collection."""
        if not embeddings:
            return

        new_dims = {_embedding_dimension(embedding) for embedding in embeddings}
        if len(new_dims) != 1:
            raise ValueError(
                f"Inconsistent embedding dimensions within batch: {sorted(new_dims)}"
            )
        new_dim = next(iter(new_dims))

        existing_dim = self.get_collection_embedding_dimension()
        if existing_dim is not None and existing_dim != new_dim:
            raise EmbeddingDimensionMismatchError(
                "Embedding dimension mismatch: the existing Chroma collection "
                f"uses dimension {existing_dim}, but the new embeddings have "
                f"dimension {new_dim}. The collection was NOT modified. "
                "Your embedding configuration (EMBEDDING_MODEL and LLM_BASE_URL) "
                "must match the configuration used to build the existing index."
            )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        self._validate_new_embeddings(embeddings)

        try:
            if hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
            else:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
        except Exception as exc:
            if _is_dimension_mismatch_error(exc):
                existing_dim = self.get_collection_embedding_dimension()
                new_dim = _embedding_dimension(embeddings[0]) if embeddings else None
                raise EmbeddingDimensionMismatchError(
                    "Chroma rejected embeddings due to a dimension mismatch. "
                    f"Existing collection dimension: {existing_dim}. "
                    f"New embedding dimension: {new_dim}. "
                    "The collection was NOT deleted and was NOT modified. "
                    "Align EMBEDDING_MODEL and LLM_BASE_URL "
                    "with the configuration used for the existing vectors."
                ) from exc
            raise

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Query the collection and return raw Chroma results."""
        cnt = self.get_collection_count()
        if cnt == 0:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        query_dim = _embedding_dimension(query_embedding)
        existing_dim = self.get_collection_embedding_dimension()
        if existing_dim is not None and query_dim != existing_dim:
            raise EmbeddingDimensionMismatchError(
                f"Query embedding dimension {query_dim} does not match "
                f"collection dimension {existing_dim}."
            )
        effective_n = min(n_results, cnt)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=effective_n)
        return results

    def get_chunks_by_page(
        self,
        page_start: int,
        page_end: int | None = None,
        book_title: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all chunks that cover the specified page or page range."""
        target_end = page_end if page_end is not None else page_start
        cnt = self.get_collection_count()
        if cnt == 0:
            return []

        try:
            # Build Chroma native where filter for efficient range lookup.
            # Chroma supports $lte and $gte operators in the where clause.
            where_conditions: list[dict] = [
                {"page_start": {"$lte": target_end}},
                {"page_end": {"$gte": page_start}},
            ]

            try:
                if book_title:
                    # Exact book_title match as an additional filter.
                    # Fall back to Python-side filtering if Chroma rejects $and with 3 conditions.
                    where_clause: dict = {
                        "$and": [
                            {"page_start": {"$lte": target_end}},
                            {"page_end": {"$gte": page_start}},
                            {"book_title": {"$eq": book_title}},
                        ]
                    }
                else:
                    where_clause = {
                        "$and": [
                            {"page_start": {"$lte": target_end}},
                            {"page_end": {"$gte": page_start}},
                        ]
                    }
                data = self.collection.get(
                    where=where_clause,
                    include=["documents", "metadatas"],
                )
            except Exception as where_exc:
                # Chroma version may not support $and/$lte/$gte — fall back to full scan.
                logger.warning(
                    "Chroma where-filter failed (%s); falling back to full scan.",
                    where_exc,
                )
                data = self.collection.get(include=["documents", "metadatas"])

            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            ids = data.get("ids") or []

            results: List[Dict[str, Any]] = []
            for cid, doc, meta in zip(ids, docs, metas):
                if not meta or not doc:
                    continue
                p_s = meta.get("page_start") or meta.get("page_number")
                p_e = meta.get("page_end") or p_s
                if p_s is None:
                    continue
                try:
                    p_s = int(p_s)
                    p_e = int(p_e)
                except (ValueError, TypeError):
                    continue

                # Re-apply range check in case the fallback path ran.
                if p_s <= target_end and p_e >= page_start:
                    if book_title:
                        b_cand = str(
                            meta.get("book_title")
                            or meta.get("book_name")
                            or meta.get("filename")
                            or ""
                        ).lower()
                        b_target = book_title.lower()
                        if b_target not in b_cand and b_cand not in b_target:
                            continue

                    results.append({
                        "id": cid,
                        "text": str(doc),
                        "metadata": dict(meta),
                    })

            results.sort(
                key=lambda x: (
                    str(x["metadata"].get("book_title") or ""),
                    int(x["metadata"].get("page_start") or 0),
                    str(x["id"]),
                )
            )
            return results
        except Exception as exc:
            logger.warning("Could not get chunks by page from Chroma: %s", exc)
            return []

    def delete_collection(self) -> None:
        """Explicit collection deletion — never called automatically by this class."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def clear_collection(self) -> None:
        """Remove every document from the current collection without recreating the DB."""
        try:
            ids = self.collection.get(include=[]).get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:
            logger.warning("Could not clear collection %s by IDs: %s", self.collection_name, exc)
            self.delete_collection()

    def get_collection_count(self) -> int:
        return self.collection.count()

    def collection_exists(self) -> bool:
        """Return True if the named collection is registered in Chroma."""
        try:
            self.client.get_collection(name=self.collection_name)
            return True
        except Exception:
            return False

    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        """Return the subset of *ids* that already exist in the collection."""
        if not ids:
            return set()

        existing: Set[str] = set()
        for start in range(0, len(ids), _ID_LOOKUP_BATCH_SIZE):
            batch = ids[start : start + _ID_LOOKUP_BATCH_SIZE]
            try:
                result = self.collection.get(ids=batch, include=[])
            except Exception as exc:
                logger.warning("get_existing_ids batch lookup failed: %s", exc)
                continue
            if result and result.get("ids"):
                existing.update(result["ids"])
        return existing

    def get_diagnostics(
        self,
        *,
        configured_model: Optional[str] = None,
        configured_base_url: Optional[str] = None,
        current_embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return safe, non-secret vector-store diagnostics."""
        provider_host = "unknown"
        if configured_base_url:
            provider_host = urlparse(configured_base_url).netloc or configured_base_url

        existing_dim = self.get_collection_embedding_dimension()
        count = self.get_collection_count()
        dimensions_compatible: Optional[bool] = None
        if existing_dim is not None and current_embedding_dim is not None:
            dimensions_compatible = existing_dim == current_embedding_dim

        return {
            "collection_name": self.collection_name,
            "collection_exists": self.collection_exists(),
            "collection_count": count,
            "existing_embedding_dimension": existing_dim,
            "current_embedding_dimension": current_embedding_dim,
            "dimensions_compatible": dimensions_compatible,
            "configured_embedding_provider": provider_host,
            "configured_embedding_model": configured_model,
            "persist_directory": self.persist_directory,
        }

    @property
    def index(self) -> Optional[Any]:
        try:
            cnt = self.get_collection_count()
            return self.collection if cnt and cnt > 0 else None
        except Exception:
            return None
