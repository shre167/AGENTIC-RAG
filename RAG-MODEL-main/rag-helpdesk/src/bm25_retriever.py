from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rank_bm25 import BM25Plus as BM25Model
except ImportError:
    from rank_bm25 import BM25Okapi as BM25Model

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, VECTORSTORE_DIR
from src.document_loader import chunk_documents, load_documents

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "bm25_index.pkl"
_INDEX_VERSION = 2


class BM25Retriever:
    """Persistent BM25 index using the exact same canonical chunks as Chroma.

    The important invariant is: one chunk produced by document_loader is one
    record here, with the same filename + chunk_id + text + metadata.

    File replacement is atomic at the logical level: old chunks for a file are
    removed before its new chunks are inserted. This prevents stale + duplicate
    BM25 records when a document is edited or re-uploaded.
    """

    def __init__(
        self,
        kb_path: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        vectorstore_dir: str | Path | None = None,
    ) -> None:
        self.kb_path = kb_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_dir = Path(vectorstore_dir or VECTORSTORE_DIR)
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[Any] = None
        self._document_hashes: Dict[str, str] = {}

    def _index_path(self) -> Path:
        return self.vectorstore_dir / _INDEX_FILENAME

    @staticmethod
    def _file_hash(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_key(chunk: Dict[str, Any]) -> str:
        return f"{chunk.get('filename', '')}::chunk-{chunk.get('chunk_id', '')}"

    def _rebuild_model(self) -> None:
        self._bm25 = BM25Model(self._tokenized) if self._tokenized else None

    def build_index(
        self,
        kb_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Build BM25 from the canonical document_loader chunks and persist it."""
        path = kb_path or self.kb_path
        if not path:
            raise ValueError("BM25 knowledge-base path is not configured.")
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        docs = load_documents(path)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self._chunks = []
        self._tokenized = []
        file_contents: Dict[str, List[str]] = {}
        for doc in docs:
            fname = str(doc.get("filename", ""))
            file_contents.setdefault(fname, []).append(str(doc.get("content", "")))
        self._document_hashes = {
            fname: self._file_hash("\n\n".join(parts))
            for fname, parts in file_contents.items()
        }

        seen_keys: set[str] = set()
        for chunk in chunks:
            if not (chunk.get("text") or "").strip():
                continue
            key = self._chunk_key(chunk)
            if key in seen_keys:
                raise ValueError(f"Duplicate BM25 chunk ID generated: {key}")
            seen_keys.add(key)
            self._chunks.append(self._extract_chunk_meta(chunk))
            self._tokenized.append(self._tokenize(chunk.get("text") or ""))

        self._rebuild_model()
        self.save_index()
        logger.info(
            "BM25: rebuilt index over %d chunks from %d documents.",
            len(self._chunks), len(docs),
        )

    def replace_file_chunks(
        self,
        filename: str,
        new_chunks: List[Dict[str, Any]],
        content_hash: str | None = None,
    ) -> None:
        """Replace every BM25 chunk belonging to *filename* with new_chunks."""
        filename = Path(filename).name
        kept_chunks: List[Dict[str, Any]] = []
        kept_tokens: List[List[str]] = []

        for chunk, tokens in zip(self._chunks, self._tokenized):
            if str(chunk.get("filename", "")) != filename:
                kept_chunks.append(chunk)
                kept_tokens.append(tokens)

        seen: set[str] = {self._chunk_key(c) for c in kept_chunks}
        for chunk in new_chunks:
            if not (chunk.get("text") or "").strip():
                continue
            key = self._chunk_key(chunk)
            if key in seen:
                raise ValueError(f"Duplicate BM25 chunk ID: {key}")
            seen.add(key)
            kept_chunks.append(self._extract_chunk_meta(chunk))
            kept_tokens.append(self._tokenize(chunk.get("text") or ""))

        self._chunks = kept_chunks
        self._tokenized = kept_tokens
        if content_hash is not None:
            self._document_hashes[filename] = content_hash
        self._rebuild_model()
        logger.info(
            "BM25: replaced file %s; index now contains %d chunks.",
            filename, len(self._chunks),
        )

    def delete_file(self, filename: str) -> int:
        """Delete all BM25 chunks belonging to a source file."""
        filename = Path(filename).name
        before = len(self._chunks)
        kept = [
            (chunk, tokens)
            for chunk, tokens in zip(self._chunks, self._tokenized)
            if str(chunk.get("filename", "")) != filename
        ]
        self._chunks = [c for c, _ in kept]
        self._tokenized = [t for _, t in kept]
        self._document_hashes.pop(filename, None)
        self._rebuild_model()
        removed = before - len(self._chunks)
        if removed:
            logger.info("BM25: deleted %d chunks for %s.", removed, filename)
        return removed

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Backward-compatible append; prefer replace_file_chunks for updates."""
        if not new_chunks:
            return
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in new_chunks:
            by_file.setdefault(Path(str(chunk.get("filename", ""))).name, []).append(chunk)
        for filename, chunks in by_file.items():
            self.replace_file_chunks(filename, chunks)

    def save_index(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._index_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "chunks": self._chunks,
            "tokenized": self._tokenized,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "document_hashes": self._document_hashes,
        }
        with open(target, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25: index saved to %s.", target)

    def load_index(self, path: str | Path | None = None) -> bool:
        target = Path(path) if path else self._index_path()
        if not target.exists():
            return False
        try:
            with open(target, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != _INDEX_VERSION:
                logger.info("BM25: ignoring incompatible index version.")
                return False
            self._chunks = payload.get("chunks", [])
            self._tokenized = payload.get("tokenized", [])
            self.chunk_size = payload.get("chunk_size", self.chunk_size)
            self.chunk_overlap = payload.get("chunk_overlap", self.chunk_overlap)
            self._document_hashes = payload.get("document_hashes", {})
            if len(self._chunks) != len(self._tokenized):
                return False
            self._rebuild_model()
            logger.info("BM25: loaded index from %s (%d chunks).", target, len(self._chunks))
            return True
        except Exception as exc:
            logger.warning("BM25: failed to load index from %s: %s", target, exc)
            return False

    def is_in_sync_with_documents(self, kb_path: str | Path | None = None) -> bool:
        """Return True only when persisted BM25 hashes match the current KB files."""
        path = kb_path or self.kb_path
        if not path:
            return False
        docs = load_documents(path)
        file_contents: Dict[str, List[str]] = {}
        for doc in docs:
            fname = str(doc.get("filename", ""))
            file_contents.setdefault(fname, []).append(str(doc.get("content", "")))
        current = {
            fname: self._file_hash("\n\n".join(parts))
            for fname, parts in file_contents.items()
        }
        return current == self._document_hashes

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(0, top_k)]
        results: List[Dict[str, Any]] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                overlap = set(tokens) & set(self._tokenized[idx])
                if not overlap:
                    continue
                score = float(len(overlap))
            results.append({**self._chunks[idx], "bm25_score": score})
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    @staticmethod
    def _extract_chunk_meta(c: Dict[str, Any]) -> Dict[str, Any]:
        fname = (
            c.get("filename")
            or c.get("book_name")
            or c.get("source_name")
            or ""
        )
        suffix = Path(fname).suffix.lstrip(".").lower() if fname else ""
        p_start = c.get("page_start") or c.get("page_number") or 0
        p_end = c.get("page_end") or p_start
        try:
            p_start = int(p_start)
        except (ValueError, TypeError):
            p_start = 0
        try:
            p_end = int(p_end)
        except (ValueError, TypeError):
            p_end = p_start

        return {
            "filename": fname,
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", ""),
            "book_title": c.get("book_title") or c.get("book_name") or "",
            "book_name": c.get("book_name") or fname,
            "book_id": c.get("book_id", ""),
            "source_path": c.get("source_path", ""),
            "section_heading": c.get("section_heading", ""),
            "section_title": c.get("section_title", ""),
            "chapter_num": c.get("chapter_num"),
            "chapter_title": c.get("chapter_title", ""),
            "section_path": c.get("section_path", ""),
            "section_level": c.get("section_level", 0),
            "chunk_index": c.get("chunk_index", 0),
            "total_section_chunks": c.get("total_section_chunks", 0),
            "category": c.get("category", ""),
            "has_heading": c.get("has_heading", False),
            "file_type": c.get("file_type") or suffix or "txt",
            "page_number": p_start,
            "page_start": p_start,
            "page_end": p_end,
        }

    def get_chunks_by_page(
        self,
        page_start: int,
        page_end: int | None = None,
        book_title: str | None = None,
    ) -> List[Dict[str, Any]]:
        target_end = page_end if page_end is not None else page_start
        results: List[Dict[str, Any]] = []
        for chunk in self._chunks:
            p_s = chunk.get("page_start") or chunk.get("page_number") or 0
            p_e = chunk.get("page_end") or p_s
            try:
                p_s = int(p_s)
                p_e = int(p_e)
            except (ValueError, TypeError):
                continue
            if p_s <= target_end and p_e >= page_start:
                if book_title:
                    b_cand = str(
                        chunk.get("book_title")
                        or chunk.get("book_name")
                        or chunk.get("filename")
                        or ""
                    ).lower()
                    b_target = book_title.lower()
                    if b_target not in b_cand and b_cand not in b_target:
                        continue
                results.append(dict(chunk))
        return results

