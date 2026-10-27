from __future__ import annotations

import logging
from typing import Any

from src.rag_pipeline.models import (
    Citation,
    CitationCoverage,
    ConfidenceResult,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


def evaluate_confidence(
    evaluation: EvaluationResult,
    citations: list[Citation],
    citation_coverage: CitationCoverage,
    evidence_level: str = "moderate",
) -> ConfidenceResult:
    """
    Computes evidence/grounding-driven confidence (HIGH, MEDIUM, LOW).
    Does NOT use raw scores as probabilities. Grounded in direct answer support,
    sufficiency, and absence of unsupported claims.
    """
    crit = evaluation.criteria
    direct_score = crit.get("direct_answer_support", {}).score if "direct_answer_support" in crit else 50.0
    relevance_score = crit.get("evidence_relevance", {}).score if "evidence_relevance" in crit else 50.0
    sufficiency_score = crit.get("evidence_sufficiency", {}).score if "evidence_sufficiency" in crit else 50.0
    agreement_score = crit.get("retriever_agreement", {}).score if "retriever_agreement" in crit else 50.0
    coverage_pct = citation_coverage.coverage_percentage
    has_unsupported = citation_coverage.has_unsupported

    # Grounding-driven score. Retrieval agreement and raw retrieval scores do
    # not enter this calculation: a popular-but-wrong passage must not raise
    # confidence. Claim verification is the largest component.
    raw_confidence_score = (
        (direct_score * 0.30)
        + (coverage_pct * 0.40)
        + (sufficiency_score * 0.20)
        + (relevance_score * 0.10)
    )

    # Unsupported claim penalty.
    #
    # This is proportional to the SHARE of claims that are unsupported,
    # not the raw count. A flat per-claim penalty unfairly punishes
    # detailed answers: an answer with 11 extracted claims and 4
    # unsupported ones is not automatically worse than a 4-claim answer
    # with 4 unsupported ones, but a flat count-based penalty scored
    # them identically (both hit the floor). Scaling by ratio preserves
    # the ability to distinguish "mostly right" from "mostly wrong."
    if has_unsupported:
        total_claims = max(1, citation_coverage.total_claims)
        unsupported_ratio = citation_coverage.unsupported_claims / total_claims
        penalty = unsupported_ratio * 40.0
        raw_confidence_score = raw_confidence_score - penalty

    final_score = round(max(5.0, min(100.0, raw_confidence_score)), 1)

    # Determine confidence tier
    if final_score >= 78.0 and not has_unsupported and direct_score >= 65.0:
        level = "HIGH"
    elif final_score >= 50.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Build explainable evidence checklist
    reasons: list[str] = []

    # 1. Direct answer support
    if direct_score >= 70.0:
        reasons.append("✓ Directly supports the query with explicit explanatory statements")
    elif direct_score >= 45.0:
        reasons.append("ℹ Partial direct answer support: relevant context retrieved")
    else:
        reasons.append("⚠ Weak direct answer support: retrieved passages lack direct answer")

    # 2. Retriever agreement
    if agreement_score >= 60.0:
        reasons.append("✓ Confirmed by multiple independent retrieval mechanisms (Dense + BM25)")
    else:
        reasons.append("ℹ Identified primarily by a single retrieval mechanism")

    # 3. Ranking stability & RRF retention
    stability_score = crit.get("ranking_stability", {}).score if "ranking_stability" in crit else 50.0
    if stability_score >= 65.0:
        reasons.append("✓ Consistently retained at top rank through RRF fusion")
    else:
        reasons.append("ℹ Moderate rank movement during fusion")

    # 4. Evidence sufficiency
    if sufficiency_score >= 75.0:
        reasons.append("✓ Evidence is sufficient for the query complexity")
    else:
        reasons.append("⚠ Evidence sufficiency is limited for full query scope")

    # 5. Grounding & unsupported claims
    if not has_unsupported and coverage_pct >= 90.0:
        reasons.append(f"✓ {coverage_pct:.0f}% claim citation coverage (no unsupported claims detected)")
    elif has_unsupported:
        reasons.append(
            f"⚠ Unsupported claim detected ({citation_coverage.unsupported_claims} ungrounded statement(s))"
        )
    else:
        reasons.append(f"ℹ Partial citation coverage ({coverage_pct:.0f}%)")

    # Collect supporting citations
    supporting_evidence: list[dict[str, Any]] = [
        {
            "marker": c.marker,
            "filename": c.filename,
            "chunk_id": c.chunk_id,
            "status": c.status,
            "passage": c.passage[:180] + ("..." if len(c.passage) > 180 else ""),
        }
        for c in citations
        if c.status in ("supported", "partially_supported")
    ][:4]

    return ConfidenceResult(
        level=level,
        score=final_score,
        reasons=reasons,
        supporting_evidence=supporting_evidence,
    )