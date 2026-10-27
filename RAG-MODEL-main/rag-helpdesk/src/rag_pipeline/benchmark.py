"""Label-aware benchmark metrics; intentionally separate from live heuristics."""
from __future__ import annotations

import math
from typing import Any


def evaluate_benchmark_case(case: dict[str, Any], retrieved: list[dict[str, Any]], k: int = 5) -> dict[str, Any] | None:
    relevant = {
        (f"{item.get('filename', item.get('source', ''))}::{item.get('chunk_id', '')}" if isinstance(item, dict) else f"{item}::")
        for item in case.get("relevant_documents", case.get("relevant_chunks", []))
    }
    if not relevant:
        return None
    ranks = [
        f"{item.get('filename', '')}::{item.get('chunk_id', '')}" in relevant
        or f"{item.get('filename', '')}::" in relevant
        for item in retrieved[:k]
    ]
    hits = sum(ranks)
    first = next((index + 1 for index, hit in enumerate(ranks) if hit), None)
    dcg = sum((1 / math.log2(index + 2)) for index, hit in enumerate(ranks) if hit)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return {
        "precision_at_k": hits / k, "recall_at_k": hits / len(relevant),
        "mrr": 1 / first if first else 0.0, "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "k": k, "labelled": True,
    }
