from __future__ import annotations

import os
from typing import Any, List, Optional


class ReRanker:
    """
    CrossEncoder-based document reranker.

    Uses:
        cross-encoder/ms-marco-MiniLM-L-12-v2

    The rest of the RAG pipeline only needs to call:
        reranker.rerank(query, docs)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = (
            model_name
            or os.getenv(
                "RERANKER_MODEL",
                self.DEFAULT_MODEL,
            )
        ).strip()

        self._model: Any = None
        self.available = False

        # Load CrossEncoder only if sentence-transformers
        # is available in the current environment.
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.available = True

        except Exception:
            # Reranking is optional. The rest of the RAG
            # pipeline can continue without it.
            self._model = None
            self.available = False

    def rerank(
        self,
        query: str,
        docs: List[str],
    ) -> List[float]:
        """
        Return one relevance score for every document.

        Higher score = stronger relevance.

        Scores are returned in the SAME order as `docs`.
        """

        if not docs:
            return []

        if not query or not query.strip():
            return [0.0 for _ in docs]

        if not self.available or self._model is None:
            return [0.0 for _ in docs]

        try:
            pairs = [
                [query, document]
                for document in docs
            ]

            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
            )

            return [
                float(score)
                for score in scores
            ]

        except Exception:
            return [0.0 for _ in docs]