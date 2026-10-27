from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.rag_pipeline.models import (
    Candidate,
    CriterionScore,
    EvaluationResult,
)

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
}

_DEFINITION_PATTERNS = [
    re.compile(r"\b(?:is|are)\s+(?:a|an|the)?\s*(?:scientific\s+study|study|branch|field|process|phenomenon|region|celestial|optical|space|telescope|mission|theory|model)\b", re.IGNORECASE),
    re.compile(r"\b(?:refers\s+to|defined\s+as|known\s+as|describes|is\s+composed\s+of|consists\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+(?:primarily|commonly|generally|essentially|typically)\b", re.IGNORECASE),
]


def _extract_query_terms(query: str) -> list[str]:
    """Extract content terms from query in order, lowercase, minus stopwords."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return content if content else tokens


def _find_definitional_sentences(text: str, query_terms: list[str]) -> list[str]:
    """Find sentences that define or directly describe the query terms."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    matches = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        has_term = any(t in s_lower for t in query_terms)
        if not has_term:
            continue
        is_def = any(p.search(s_lower) for p in _DEFINITION_PATTERNS)
        # Also check simple 'X is ...' or 'X are ...' pattern
        for term in query_terms:
            if re.search(rf"\b{re.escape(term)}\s+(?:is|are|was|were)\b", s_lower):
                is_def = True
                break
        if is_def:
            matches.append(s_clean)
    return matches


def _term_density_in_text(text: str, query_terms: list[str]) -> float:
    """Calculate term occurrence frequency and prominence."""
    if not query_terms or not text:
        return 0.0
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text_lower)
    if not tokens:
        return 0.0
    match_count = sum(1 for t in tokens if t in query_terms)
    # Check if first sentence contains query terms (prominence)
    first_sent = text_lower.split(".")[0] if "." in text_lower else text_lower
    has_leading_focus = any(t in first_sent for t in query_terms)
    base_ratio = match_count / max(1, len(tokens))
    prominence_bonus = 0.05 if has_leading_focus else 0.0
    return min(1.0, (base_ratio * 10.0) + prominence_bonus)


# ============================================================================
# CRITERION A: RETRIEVAL STRENGTH
# ============================================================================

def evaluate_retrieval_strength(
    candidates: list[Candidate],
    retrieval_mode: str = "hybrid",
) -> CriterionScore:
    """
    Evaluates the relative strength and separation of the top candidates
    within the current query result set. Does NOT use fixed raw-score cutoffs.
    """
    if not candidates:
        return CriterionScore(
            score=0.0,
            label="Retrieval Strength",
            definition="Relative separation and prominence of top candidates within the query.",
            reason="No candidates were retrieved for this query.",
            supporting_chunks=[],
            metrics={"candidate_count": 0},
        )

    count = len(candidates)
    if count == 1:
        return CriterionScore(
            score=75.0,
            label="Retrieval Strength",
            definition="Relative separation and prominence of top candidates within the query.",
            reason="A single candidate was retrieved, showing isolated presence without competition.",
            supporting_chunks=[candidates[0].to_dict()],
            metrics={"candidate_count": 1},
        )

    # Use query-local relative gap between top candidate and next-best
    if retrieval_mode == "hybrid":
        scores = [c.rrf_score for c in candidates if c.rrf_score > 0]
        if not scores:
            scores = [1.0 / (60 + (c.dense_rank or 99)) for c in candidates]
        top = scores[0]
        second = scores[1] if len(scores) > 1 else top
        bottom = scores[-1]
        score_span = max(1e-6, top - bottom)
        top_gap = top - second
        relative_gap = top_gap / score_span
    elif retrieval_mode == "vector":
        dists = [c.dense_distance for c in candidates if c.dense_distance is not None]
        if dists:
            top_d = dists[0]
            sec_d = dists[1] if len(dists) > 1 else top_d
            max_d = dists[-1]
            span = max(1e-6, max_d - top_d)
            relative_gap = (sec_d - top_d) / span
        else:
            relative_gap = 0.5
    else:  # bm25
        scores = [c.bm25_score for c in candidates if c.bm25_score is not None]
        if scores and scores[0] > 0:
            top_s = scores[0]
            sec_s = scores[1] if len(scores) > 1 else top_s
            bot_s = scores[-1]
            span = max(1e-6, top_s - bot_s)
            relative_gap = (top_s - sec_s) / span
        else:
            relative_gap = 0.5

    # Map relative separation to 0-100 scale:
    # A moderate gap (0.1 - 0.3) indicates healthy ranking; very large (>0.4) indicates dominant top candidate
    base_score = 60.0 + min(35.0, relative_gap * 75.0)
    top_candidate = candidates[0]
    if top_candidate.in_both_retrievers:
        base_score = min(100.0, base_score + 5.0)

    score = round(max(10.0, min(100.0, base_score)), 1)
    if relative_gap >= 0.25:
        reason = (
            f"Top candidate shows clear relevance separation (relative lead: {relative_gap:.1%}) "
            f"over subsequent candidates in the pool."
        )
    elif relative_gap >= 0.10:
        reason = (
            f"Top candidate exhibits healthy margin over competing candidates "
            f"(relative lead: {relative_gap:.1%})."
        )
    else:
        reason = (
            f"Top candidates have clustered retrieval scores (relative lead: {relative_gap:.1%}), "
            f"indicating multiple closely competing passages."
        )

    return CriterionScore(
        score=score,
        label="Retrieval Strength",
        definition="Relative separation and prominence of top candidates within the query.",
        reason=reason,
        supporting_chunks=[candidates[0].to_dict()],
        metrics={"relative_gap": round(relative_gap, 4), "pool_size": count},
    )


# ============================================================================
# CRITERION B: DIRECT ANSWER SUPPORT (PRIORITIZED)
# ============================================================================

def evaluate_direct_answer_support(
    query: str,
    candidates: list[Candidate],
) -> CriterionScore:
    """
    Examines whether retrieved chunks actually contain information capable
    of directly answering the query, distinguishing core explanations from
    incidental mentions (e.g. JWST mentioning 'cosmology' vs cosmology.txt).
    """
    if not candidates or not query.strip():
        return CriterionScore(
            score=0.0,
            label="Direct Answer Support",
            definition="Whether retrieved evidence actually contains passages that answer the query.",
            reason="No candidate evidence available to answer the query.",
            supporting_chunks=[],
            metrics={"supporting_count": 0},
        )

    query_terms = _extract_query_terms(query)
    supporting_candidates: list[dict[str, Any]] = []
    best_support_score = 0.0

    for rank, cand in enumerate(candidates[:6], start=1):
        text = cand.text or ""
        text_lower = text.lower()
        definitional_sents = _find_definitional_sentences(text, query_terms)
        density = _term_density_in_text(text, query_terms)

        # Check title/filename match
        cand_fname = (cand.filename or "").lower()
        filename_match = any(t in cand_fname for t in query_terms)

        # Score this candidate's direct answer capability
        cand_support = 0.0
        if cand.metadata.get("is_page_match"):
            cand_support += 75.0
        elif definitional_sents:
            cand_support += 50.0 + min(30.0, len(definitional_sents) * 15.0)
        elif density > 0.1:
            cand_support += 30.0 + min(20.0, density * 50.0)

        if filename_match:
            cand_support += 15.0

        # Term coverage
        terms_in_text = sum(1 for t in query_terms if t in text_lower)
        coverage_ratio = terms_in_text / max(1, len(query_terms))
        cand_support += coverage_ratio * 15.0

        cand_support = min(100.0, cand_support)

        if cand_support > best_support_score:
            best_support_score = cand_support

        if cand_support >= 35.0:
            excerpt = definitional_sents[0] if definitional_sents else text[:250].strip() + "..."
            supporting_candidates.append({
                "rank": rank,
                "filename": cand.filename,
                "chunk_id": cand.chunk_id,
                "passage": excerpt,
                "definitional_match": bool(definitional_sents),
                "support_score": round(cand_support, 1),
            })

    # Overall score reflecting top supporting passage quality
    score = round(max(5.0, min(100.0, best_support_score)), 1)

    if supporting_candidates:
        top_sup = supporting_candidates[0]
        if top_sup["definitional_match"]:
            reason = (
                f"Candidate #{top_sup['rank']} ({top_sup['filename']} Chunk {top_sup['chunk_id']}) "
                f"contains direct definitional/explanatory statements answering the query."
            )
        else:
            reason = (
                f"Candidate #{top_sup['rank']} ({top_sup['filename']} Chunk {top_sup['chunk_id']}) "
                f"provides strong topical context directly addressing query terms."
            )
    else:
        reason = (
            "Retrieved passages contain term matches but lack direct explanatory statements "
            "answering the user's specific inquiry."
        )

    return CriterionScore(
        score=score,
        label="Direct Answer Support",
        definition="Whether retrieved evidence actually contains passages that answer the query.",
        reason=reason,
        supporting_chunks=supporting_candidates[:3],
        metrics={
            "best_support_score": round(best_support_score, 1),
            "supporting_chunks_count": len(supporting_candidates),
        },
    )


# ============================================================================
# CRITERION C: EVIDENCE RELEVANCE (PRIORITIZED)
# ============================================================================

def evaluate_evidence_relevance(
    query: str,
    candidates: list[Candidate],
) -> CriterionScore:
    """
    Distinguishes directly relevant content from broadly related astronomy content
    and irrelevant noise across the candidate set.
    """
    if not candidates:
        return CriterionScore(
            score=0.0,
            label="Evidence Relevance",
            definition="How closely retrieved evidence matches the user's actual information need.",
            reason="No candidates available to evaluate relevance.",
            supporting_chunks=[],
            metrics={},
        )

    query_terms = _extract_query_terms(query)
    directly_relevant = []
    broadly_related = []
    tangential = []

    for cand in candidates[:8]:
        text_lower = (cand.text or "").lower()
        fname_lower = (cand.filename or "").lower()
        term_matches = sum(1 for t in query_terms if t in text_lower or t in fname_lower)
        coverage = term_matches / max(1, len(query_terms))
        density = _term_density_in_text(cand.text or "", query_terms)

        cand_info = {
            "filename": cand.filename,
            "chunk_id": cand.chunk_id,
            "coverage": round(coverage, 2),
            "density": round(density, 2),
        }

        if coverage >= 0.75 or (coverage >= 0.5 and density >= 0.1):
            directly_relevant.append(cand_info)
        elif coverage >= 0.33 or density >= 0.05:
            broadly_related.append(cand_info)
        else:
            tangential.append(cand_info)

    total_inspected = len(directly_relevant) + len(broadly_related) + len(tangential)
    if total_inspected == 0:
        score = 20.0
        reason = "Unable to assess candidate relevance."
    else:
        score = (
            (len(directly_relevant) * 100.0)
            + (len(broadly_related) * 55.0)
            + (len(tangential) * 10.0)
        ) / total_inspected

        # Top candidate relevance bonus
        if directly_relevant:
            score = min(100.0, score + 10.0)

    score = round(max(5.0, min(100.0, score)), 1)
    reason = (
        f"Found {len(directly_relevant)} directly relevant chunk(s), "
        f"{len(broadly_related)} broadly related, and {len(tangential)} peripheral across top candidates."
    )

    return CriterionScore(
        score=score,
        label="Evidence Relevance",
        definition="How closely retrieved evidence matches the user's actual information need.",
        reason=reason,
        supporting_chunks=directly_relevant[:3],
        metrics={
            "directly_relevant_count": len(directly_relevant),
            "broadly_related_count": len(broadly_related),
            "tangential_count": len(tangential),
        },
    )


# ============================================================================
# CRITERION D: RETRIEVER AGREEMENT
# ============================================================================

def evaluate_retriever_agreement(
    dense_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
) -> CriterionScore:
    """
    Evaluates candidate overlap between Dense and BM25 independent retrieval.
    Does NOT conflate agreement with ground-truth correctness.
    """
    dense_keys = {
        f"{c.get('filename')}::{c.get('chunk_id')}"
        for c in dense_candidates
        if c.get("filename")
    }
    bm25_keys = {
        f"{c.get('filename')}::{c.get('chunk_id')}"
        for c in bm25_candidates
        if c.get("filename")
    }

    if not dense_keys or not bm25_keys:
        return CriterionScore(
            score=50.0 if (dense_keys or bm25_keys) else 0.0,
            label="Retriever Agreement",
            definition="Independent consensus between Dense and BM25 candidate pools.",
            reason="Only one retrieval mechanism was active or returned results for this query.",
            supporting_chunks=[],
            metrics={"dense_count": len(dense_keys), "bm25_count": len(bm25_keys), "overlap_count": 0},
        )

    intersection = dense_keys & bm25_keys
    union = dense_keys | bm25_keys
    jaccard = (len(intersection) / len(union)) * 100.0 if union else 0.0

    # Also calculate overlap in top-3 candidates specifically
    top_dense = {f"{c.get('filename')}::{c.get('chunk_id')}" for c in dense_candidates[:3]}
    top_bm25 = {f"{c.get('filename')}::{c.get('chunk_id')}" for c in bm25_candidates[:3]}
    top3_overlap = len(top_dense & top_bm25)

    # Score: 60% Jaccard pool + 40% top-3 agreement
    top3_ratio = (top3_overlap / min(len(top_dense), len(top_bm25))) if top_dense and top_bm25 else 0.0
    combined_score = (jaccard * 0.5) + (top3_ratio * 100.0 * 0.5)
    score = round(max(0.0, min(100.0, combined_score)), 1)

    overlapping_chunks = []
    for c in dense_candidates:
        key = f"{c.get('filename')}::{c.get('chunk_id')}"
        if key in intersection:
            bm_match = next((b for b in bm25_candidates if f"{b.get('filename')}::{b.get('chunk_id')}" == key), {})
            overlapping_chunks.append({
                "filename": c.get("filename"),
                "chunk_id": c.get("chunk_id"),
                "dense_rank": c.get("rank") or c.get("dense_rank"),
                "bm25_rank": bm_match.get("rank") or bm_match.get("bm25_rank"),
            })

    reason = (
        f"Dense and BM25 share {len(intersection)} chunk(s) (Jaccard: {jaccard:.1f}%), "
        f"with {top3_overlap} overlapping in the top-3. "
        "Retriever agreement reflects consensus between methods, not factual accuracy."
    )

    return CriterionScore(
        score=score,
        label="Retriever Agreement",
        definition="Independent consensus between Dense and BM25 candidate pools.",
        reason=reason,
        supporting_chunks=overlapping_chunks[:4],
        metrics={
            "jaccard_similarity": round(jaccard, 1),
            "overlapping_chunks_count": len(intersection),
            "top3_overlap_count": top3_overlap,
        },
    )


# ============================================================================
# CRITERION E: RANKING STABILITY
# ============================================================================

def evaluate_ranking_stability(
    fused_candidates: list[Candidate],
    dense_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
) -> CriterionScore:
    """
    Evaluates whether top candidates from Dense and BM25 survive into RRF
    fusion and final selected evidence without unreasonable demotion.
    """
    if not fused_candidates:
        return CriterionScore(
            score=0.0,
            label="Ranking Stability",
            definition="Retention and preservation of top candidates into fusion and selection.",
            reason="No candidates survived fusion.",
            supporting_chunks=[],
            metrics={},
        )

    # Check survival of top-3 dense and top-3 bm25
    top_dense_keys = [
        f"{c.get('filename')}::{c.get('chunk_id')}"
        for c in dense_candidates[:3]
        if c.get("filename")
    ]
    top_bm25_keys = [
        f"{c.get('filename')}::{c.get('chunk_id')}"
        for c in bm25_candidates[:3]
        if c.get("filename")
    ]

    fused_key_to_rank = {
        cand.key: rrf_rank
        for rrf_rank, cand in enumerate(fused_candidates, start=1)
    }

    survived_dense = sum(1 for k in top_dense_keys if k in fused_key_to_rank and fused_key_to_rank[k] <= 5)
    survived_bm25 = sum(1 for k in top_bm25_keys if k in fused_key_to_rank and fused_key_to_rank[k] <= 5)

    dense_possible = max(1, len(top_dense_keys))
    bm25_possible = max(1, len(top_bm25_keys))

    survival_rate = ((survived_dense / dense_possible) + (survived_bm25 / bm25_possible)) / 2.0
    score = round(max(20.0, min(100.0, survival_rate * 100.0)), 1)

    movements = []
    for k in (top_dense_keys[:2] + top_bm25_keys[:2]):
        if k in fused_key_to_rank:
            cand = next((c for c in fused_candidates if c.key == k), None)
            if cand:
                movements.append({
                    "filename": cand.filename,
                    "chunk_id": cand.chunk_id,
                    "dense_rank": cand.dense_rank,
                    "bm25_rank": cand.bm25_rank,
                    "final_rank": fused_key_to_rank[k],
                })

    reason = (
        f"Strong candidates preserved: {survived_dense}/{dense_possible} top Dense and "
        f"{survived_bm25}/{bm25_possible} top BM25 candidates maintained top-5 positions in RRF."
    )

    return CriterionScore(
        score=score,
        label="Ranking Stability",
        definition="Retention and preservation of top candidates into fusion and selection.",
        reason=reason,
        supporting_chunks=movements[:4],
        metrics={
            "survived_dense": survived_dense,
            "survived_bm25": survived_bm25,
            "survival_rate": round(survival_rate, 2),
        },
    )


# ============================================================================
# CRITERION F: EVIDENCE COHERENCE
# ============================================================================

def evaluate_evidence_coherence(
    query: str,
    selected_candidates: list[Candidate],
) -> CriterionScore:
    """
    Evaluates whether the final evidence chunks form a logically coherent set
    around the query topic, without punishing valid multi-source queries.
    """
    if not selected_candidates:
        return CriterionScore(
            score=0.0,
            label="Evidence Coherence",
            definition="Topical cohesion and logical relevance across selected evidence chunks.",
            reason="No candidates selected for final context.",
            supporting_chunks=[],
            metrics={},
        )

    sources = [c.filename for c in selected_candidates if c.filename]
    unique_sources = list(dict.fromkeys(sources))
    query_terms = _extract_query_terms(query)

    # Detect multi-concept query (e.g. 'space technology' AND 'isro')
    is_multi_concept = any(
        conj in query.lower() for conj in (" and ", " vs ", " versus ", " compared to ", " difference between ")
    ) or len(query_terms) >= 5

    # Check topical alignment of sources with query
    relevant_sources = [
        s for s in unique_sources if any(t in s.lower() for t in query_terms)
    ]

    if len(unique_sources) == 1:
        # High concentration on a single matching topic
        coherence_score = 95.0
        reason = f"All evidence concentrated coherently in primary source '{unique_sources[0]}'."
    elif is_multi_concept and len(relevant_sources) >= 2:
        # Multi-source expected and provided
        coherence_score = 92.0
        reason = (
            f"Multi-concept query matched across complementary sources: "
            f"{', '.join(unique_sources[:3])}."
        )
    elif len(relevant_sources) == len(unique_sources):
        # Multiple sources, but all explicitly relevant to query
        coherence_score = 88.0
        reason = (
            f"Selected evidence spans {len(unique_sources)} sources, all topically "
            f"relevant to the query."
        )
    else:
        # Diverse sources with potential topical dispersion
        ratio_relevant = len(relevant_sources) / max(1, len(unique_sources))
        coherence_score = 50.0 + (ratio_relevant * 40.0)
        unrelated = [s for s in unique_sources if s not in relevant_sources]
        reason = (
            f"Evidence contains {len(relevant_sources)} topically aligned source(s) and "
            f"{len(unrelated)} auxiliary source(s) ({', '.join(unrelated[:2])})."
        )

    score = round(max(20.0, min(100.0, coherence_score)), 1)
    return CriterionScore(
        score=score,
        label="Evidence Coherence",
        definition="Topical cohesion and logical relevance across selected evidence chunks.",
        reason=reason,
        supporting_chunks=[c.to_dict() for c in selected_candidates[:3]],
        metrics={
            "unique_sources": unique_sources,
            "source_count": len(unique_sources),
            "is_multi_concept": is_multi_concept,
        },
    )


# ============================================================================
# CRITERION G: EVIDENCE SUFFICIENCY (PRIORITIZED)
# ============================================================================

def evaluate_evidence_sufficiency(
    query: str,
    selected_candidates: list[Candidate],
    direct_support_score: float,
) -> CriterionScore:
    """
    Evaluates whether selected evidence is sufficient to answer the query,
    accounting for query complexity (simple definition vs multi-part comparison).
    """
    if not selected_candidates:
        return CriterionScore(
            score=0.0,
            label="Evidence Sufficiency",
            definition="Whether selected evidence provides sufficient coverage for the query complexity.",
            reason="No evidence chunks selected.",
            supporting_chunks=[],
            metrics={},
        )

    query_terms = _extract_query_terms(query)
    q_lower = query.lower().strip()

    is_simple_def = (
        q_lower.startswith(("what is ", "what are ", "who is ", "define "))
        or len(query_terms) <= 2
    )

    total_chars = sum(len(c.text or "") for c in selected_candidates)

    if is_simple_def:
        # For simple definitions, a single strong chunk with direct support is sufficient
        if direct_support_score >= 70.0:
            score = 95.0
            reason = "A single focused chunk provides complete definition and context for this straightforward query."
        elif direct_support_score >= 45.0:
            score = 75.0
            reason = "Evidence provides adequate definition, though additional detail could enhance depth."
        else:
            score = 40.0
            reason = "Retrieved chunks mention the subject but lack a complete definition."
    else:
        # Multi-part / comparative / complex question
        covered_terms = set()
        for c in selected_candidates:
            c_lower = (c.text or "").lower()
            for t in query_terms:
                if t in c_lower:
                    covered_terms.add(t)
        term_cov_ratio = len(covered_terms) / max(1, len(query_terms))
        base = term_cov_ratio * 70.0 + min(20.0, len(selected_candidates) * 6.0)
        score = min(100.0, base)
        reason = (
            f"Evidence covers {len(covered_terms)}/{len(query_terms)} query concepts "
            f"across {len(selected_candidates)} chunk(s) ({total_chars} chars)."
        )

    score = round(max(10.0, min(100.0, score)), 1)
    return CriterionScore(
        score=score,
        label="Evidence Sufficiency",
        definition="Whether selected evidence provides sufficient coverage for the query complexity.",
        reason=reason,
        supporting_chunks=[c.to_dict() for c in selected_candidates[:2]],
        metrics={
            "query_complexity": "simple_definition" if is_simple_def else "complex_multi_part",
            "evidence_chunk_count": len(selected_candidates),
            "evidence_char_count": total_chars,
        },
    )


# ============================================================================
# COMPOSITE EVALUATION: RETRIEVAL HEALTH SCORE
# ============================================================================

def evaluate_retrieval(
    query: str,
    candidates: list[Candidate],
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    fused_results: list[Candidate],
    selected_candidates: list[Candidate],
    retrieval_mode: str = "hybrid",
) -> EvaluationResult:
    """
    Runs the complete 7-criteria runtime evaluation.
    Prioritizes Direct Answer Support (30%), Evidence Relevance (25%),
    and Evidence Sufficiency (15%) over Agreement (5%).
    """
    strength = evaluate_retrieval_strength(candidates, retrieval_mode=retrieval_mode)
    direct = evaluate_direct_answer_support(query, candidates)
    relevance = evaluate_evidence_relevance(query, candidates)
    agreement = evaluate_retriever_agreement(dense_results, bm25_results)
    stability = evaluate_ranking_stability(fused_results, dense_results, bm25_results)
    coherence = evaluate_evidence_coherence(query, selected_candidates)
    sufficiency = evaluate_evidence_sufficiency(query, selected_candidates, direct.score)

    criteria = {
        "retrieval_strength": strength,
        "direct_answer_support": direct,
        "evidence_relevance": relevance,
        "retriever_agreement": agreement,
        "ranking_stability": stability,
        "evidence_coherence": coherence,
        "evidence_sufficiency": sufficiency,
    }

    # Weighted Composite Score (Health Score)
    # Direct Answer Support (30%) + Relevance (25%) + Sufficiency (15%) = 70% Core Evidence
    # Strength (10%) + Stability (10%) + Coherence (5%) + Agreement (5%) = 30% Architecture signals
    health_score = (
        (direct.score * 0.30)
        + (relevance.score * 0.25)
        + (sufficiency.score * 0.15)
        + (strength.score * 0.10)
        + (stability.score * 0.10)
        + (coherence.score * 0.05)
        + (agreement.score * 0.05)
    )
    health_score = round(max(0.0, min(100.0, health_score)), 1)

    if health_score >= 80.0:
        summary = (
            f"Robust retrieval health ({health_score}/100): evidence contains direct answer support, "
            f"high topical relevance, and sufficient depth for the query."
        )
    elif health_score >= 55.0:
        summary = (
            f"Moderate retrieval health ({health_score}/100): relevant evidence was retrieved, "
            f"with partial direct answer support."
        )
    else:
        summary = (
            f"Low retrieval health ({health_score}/100): retrieved candidate passages exhibit weak "
            f"direct answer support or tangential overlap."
        )

    # Collect distinct supporting chunks from direct answer support
    all_supporting = direct.supporting_chunks

    return EvaluationResult(
        retrieval_health_score=health_score,
        criteria=criteria,
        summary=summary,
        supporting_chunks=all_supporting,
    )


def detect_lexical_mismatch(
    query: str,
    candidates: list[Candidate],
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Detect whether retriever agreement is driven by incidental lexical overlap
    rather than primary topic match.

    Example: query "cosmology" → JWST.txt ranks highly in BM25 because the
    word "cosmology" appears once in a paragraph about JWST's research goals,
    while cosmology.txt (the primary source) may rank lower.

    Returns a diagnostic dict that is included in the trace but does NOT
    rescue the retrieval health score.  Agreement cannot compensate for
    weak direct answer support; this function only explains why.

    Design invariant: agreement must NOT rescue poor evidence.
    """
    query_terms = _extract_query_terms(query)
    if not query_terms:
        return {"lexical_mismatch_detected": False, "warnings": []}

    warnings: list[str] = []

    # Build primary topic set from dense top-1 (semantic search is less
    # susceptible to incidental keyword matches than BM25).
    dense_top1_fname = (dense_results[0].get("filename") or "").lower() if dense_results else ""
    bm25_top1_fname = (bm25_results[0].get("filename") or "").lower() if bm25_results else ""

    # Check whether BM25 #1 differs from dense #1 AND the BM25 #1 candidate
    # has a filename that does NOT match any query term.
    bm25_primary_mismatch = (
        bm25_top1_fname
        and dense_top1_fname
        and bm25_top1_fname != dense_top1_fname
        and not any(t in bm25_top1_fname for t in query_terms)
    )

    # Also check each candidate that is in BM25 but not dense, to see if
    # it only scores because of incidental term occurrence (low density).
    bm25_only_candidates: list[dict[str, Any]] = []
    dense_fnames = {(r.get("filename") or "") for r in dense_results}
    for cand in candidates:
        if cand.in_bm25 and not cand.in_dense:
            fname = (cand.filename or "").lower()
            fname_term_match = any(t in fname for t in query_terms)
            density = _term_density_in_text(cand.text or "", query_terms)
            defs = _find_definitional_sentences(cand.text or "", query_terms)
            # Flag: BM25-only, filename does NOT match query, very low density, no definitional sentences
            if not fname_term_match and density < 0.05 and not defs:
                bm25_only_candidates.append({
                    "filename": cand.filename,
                    "chunk_id": cand.chunk_id,
                    "density": round(density, 4),
                    "bm25_rank": cand.bm25_rank,
                })

    if bm25_primary_mismatch:
        warnings.append(
            f"BM25 top result '{bm25_results[0].get('filename', '')}' does not match any query term "
            f"in its filename, while dense top result is '{dense_results[0].get('filename', '')}'. "
            "This suggests BM25 is promoting a source due to incidental keyword overlap, not primary topic coverage. "
            "Retriever agreement here does NOT indicate correct evidence."
        )

    for c in bm25_only_candidates[:3]:
        warnings.append(
            f"BM25-only candidate '{c['filename']}' (rank {c['bm25_rank']}) has very low query-term "
            f"density ({c['density']:.4f}) and no definitional sentences. "
            "This is likely a lexical false positive — the source mentions the term incidentally."
        )

    return {
        "lexical_mismatch_detected": bool(warnings),
        "bm25_primary_topic_mismatch": bm25_primary_mismatch,
        "bm25_only_weak_candidates": bm25_only_candidates,
        "warnings": warnings,
        "note": (
            "Agreement between Dense and BM25 does not establish factual correctness. "
            "When BM25 promotes sources that mention a term incidentally, "
            "the retrieved evidence may not actually answer the query."
        ),
    }


def evaluate_facet_coverage(
    query: str,
    selected_candidates: list[Candidate],
) -> CriterionScore:
    """
    Evaluates coverage of distinct query facets by the selected evidence.

    A facet is a distinct sub-question or concept within the query.
    Simple single-concept queries have 1 facet; multi-part queries have N.

    Calculation:
        facets = split query on 'and', 'vs', 'versus', ','
        covered = facets where any selected chunk covers the facet's terms
        score   = (covered / total_facets) * 100

    Note: facet detection is lexical and approximate. For single-concept
    queries this always returns 100 if evidence was selected.
    """
    if not selected_candidates:
        return CriterionScore(
            score=0.0,
            label="Facet Coverage",
            definition="Coverage of distinct query sub-concepts by selected evidence.",
            reason="No evidence selected — no facets covered.",
            supporting_chunks=[],
            metrics={"facets": [], "covered": 0, "total": 0},
        )

    # Split query into facets
    raw_parts = re.split(r"\s+(?:and|vs|versus|versus|or)\s+|,\s*", query, flags=re.I)
    facets = [p.strip() for p in raw_parts if len(_extract_query_terms(p)) > 0]
    if not facets:
        facets = [query]

    covered_facets: list[str] = []
    missing_facets: list[str] = []

    for facet in facets:
        facet_terms = set(_extract_query_terms(facet))
        covered = any(
            facet_terms & set(_extract_query_terms(c.text or ""))
            for c in selected_candidates
        )
        if covered:
            covered_facets.append(facet)
        else:
            missing_facets.append(facet)

    score = round(100.0 * len(covered_facets) / max(1, len(facets)), 1)

    if not missing_facets:
        reason = f"All {len(facets)} detected facet(s) are covered by selected evidence."
    else:
        reason = (
            f"{len(covered_facets)}/{len(facets)} facet(s) covered. "
            f"Missing: {', '.join(missing_facets[:3])}."
        )

    return CriterionScore(
        score=score,
        label="Facet Coverage",
        definition="Proportion of distinct query sub-concepts covered by selected evidence.",
        reason=reason,
        supporting_chunks=[c.to_dict() for c in selected_candidates[:2]],
        metrics={
            "facets": facets,
            "covered_facets": covered_facets,
            "missing_facets": missing_facets,
            "total": len(facets),
            "covered": len(covered_facets),
        },
    )


def evaluate_answer_quality(
    query: str,
    answer: str,
    citations: list[Any],
    coverage: Any,
    selected_candidates: list[Candidate],
) -> dict[str, Any]:
    """
    Explainable live answer-quality checks.

    These are deterministic heuristics computed from recorded data only.
    They are NOT accuracy metrics. Every metric includes:
      - score (0-100)
      - reason (what was found)
      - calculation (exact formula used)

    Critical invariants:
      - Lexical overlap ≠ answer support
      - Retriever agreement ≠ correctness
      - Agreement cannot rescue poor evidence scores
    """
    terms = set(_extract_query_terms(query))
    answer_terms = set(_extract_query_terms(answer))

    # ── Answer Relevance ──────────────────────────────────────────────
    # Proportion of query content terms that appear in the answer.
    # Does NOT imply the answer is factually correct.
    overlap = len(terms & answer_terms)
    relevance_score = round(100.0 * overlap / max(1, len(terms)), 1)

    # ── Facet Coverage ────────────────────────────────────────────────
    # Uses the standalone function for consistency
    facet_result = evaluate_facet_coverage(query, selected_candidates)
    facets = facet_result.metrics.get("facets", [query])
    covered = facet_result.metrics.get("covered", 0)
    completeness_score = facet_result.score
    missing_facets = facet_result.metrics.get("missing_facets", [])

    # ── Evidence/Claim Grounding ──────────────────────────────────────
    # supported_citations / all_citations × 100
    # A citation is "supported" only when the passage verifiably contains
    # the claim's named entities and numeric values — not just topic words.
    supported = sum(1 for c in citations if c.status == "supported")
    grounding_score = round(100.0 * supported / max(1, len(citations)), 1)

    # ── Citation Coverage ─────────────────────────────────────────────
    # From the CitationCoverage object: supported_claims / total_claims × 100
    coverage_pct = getattr(coverage, "coverage_percentage", 0.0) if coverage else 0.0
    coverage_score = round(float(coverage_pct), 1)

    # ── Citation Correctness ──────────────────────────────────────────
    # Citations where direct_answer_support=True / all citations × 100
    # direct_answer_support means the passage contains an answer-bearing fact,
    # not just a topical mention. Lexical overlap alone is NOT sufficient.
    direct_supported = sum(1 for c in citations if c.direct_answer_support)
    correctness_score = round(100.0 * direct_supported / max(1, len(citations)), 1)

    def metric(score: float, reason: str, calculation: str) -> dict[str, Any]:
        return {
            "score": round(score, 1),
            "reason": reason,
            "calculation": calculation,
            "label": "",  # filled below per metric
        }

    result = {
        "answer_relevance": {
            **metric(
                relevance_score,
                f"{overlap}/{len(terms)} query content terms appear in the answer.",
                "len(query_terms ∩ answer_terms) / len(query_terms) × 100. "
                "This measures term overlap, NOT factual accuracy.",
            ),
            "label": "Answer Relevance",
            "definition": "Proportion of query content terms that appear in the generated answer.",
        },
        "answer_completeness": {
            **metric(
                completeness_score,
                f"{covered}/{len(facets)} query facet(s) covered by selected evidence. "
                + (f"Missing: {', '.join(missing_facets[:2])}." if missing_facets else ""),
                "Facets detected by splitting query on 'and/vs/,'. "
                "covered = facets where any selected chunk contains the facet's terms. "
                "score = covered/total × 100.",
            ),
            "label": "Answer Completeness",
            "definition": "Whether selected evidence covers the distinct sub-concepts of the query.",
        },
        "evidence_claim_grounding": {
            **metric(
                grounding_score,
                f"{supported}/{max(1,len(citations))} claim(s) verified against selected passages. "
                "A claim is 'supported' only when named entities and numeric values match.",
                "supported_citations / total_citations × 100. "
                "Support requires named entity + numeric value match, not just keyword overlap. "
                "Lexical overlap ≠ answer support.",
            ),
            "label": "Evidence/Claim Grounding",
            "definition": "Generated claims verified against retrieved passages; requires entity match, not just topic overlap.",
        },
        "citation_coverage": {
            **metric(
                coverage_score,
                f"CitationCoverage.coverage_percentage = {coverage_score}% "
                f"({getattr(coverage, 'supported_claims', 0)} of {getattr(coverage, 'total_claims', 0)} claims).",
                "CitationCoverage.supported_claims / CitationCoverage.total_claims × 100. "
                "Recorded directly from post-generation grounding pass.",
            ),
            "label": "Citation Coverage",
            "definition": "Proportion of generated claims that are directly supported by retrieved passages.",
        },
        "citation_correctness": {
            **metric(
                correctness_score,
                f"{direct_supported}/{max(1,len(citations))} citation(s) have direct answer support "
                "(passage contains an answer-bearing fact, not just a topical mention). "
                "Lexical overlap alone is NOT sufficient.",
                "citations_with_direct_answer_support / total_citations × 100. "
                "direct_answer_support is set only when the passage directly contains the claimed fact. "
                "Retriever agreement does NOT influence this metric.",
            ),
            "label": "Citation Correctness",
            "definition": "Citations backed by answer-bearing passages, not merely topically related text.",
        },
        "facet_coverage": {
            **metric(
                completeness_score,
                facet_result.reason,
                "Same calculation as answer_completeness — shown separately for multi-part query visibility.",
            ),
            "label": "Facet Coverage",
            "definition": facet_result.definition,
            "metrics": facet_result.metrics,
        },
    }

    return result
