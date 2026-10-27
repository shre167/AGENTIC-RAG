from __future__ import annotations

import logging
import math
import re

from src.rag_pipeline.config import (
    RRF_ABSTAIN_FLOOR,
    STRONG_RRF_THRESHOLD,
)
from src.rag_pipeline.context import (
    deduplicate_candidates,
    select_context_candidates,
)
from src.rag_pipeline.models import Candidate, Evidence
from src.rag_pipeline.query import get_retrieval_mode


logger = logging.getLogger(__name__)


def top_score_percentile(
    scores: list[float],
) -> float | None:
    """Return the percentile position of the best finite score."""
    cleaned = [
        float(s)
        for s in scores
        if math.isfinite(float(s))
    ]

    if not cleaned:
        return None

    if len(cleaned) == 1:
        return 1.0

    top = max(cleaned)

    below = sum(
        s < top
        for s in cleaned
    )

    equal = sum(
        s == top
        for s in cleaned
    )

    return max(
        0.0,
        min(
            1.0,
            (below + (equal + 1) / 2) / len(cleaned),
        ),
    )


_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did",
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


def _query_tokens(
    search_text: str,
) -> set[str]:
    raw_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            (search_text or "").lower(),
        )
    )
    content_tokens = raw_tokens - _STOPWORDS
    return content_tokens if content_tokens else raw_tokens


def _lexical_overlap_ratio(
    candidate: Candidate,
    query_tokens: set[str],
) -> float:
    if not query_tokens:
        return 0.0

    text_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            (candidate.text or "").lower(),
        )
    )

    return (
        len(query_tokens & text_tokens)
        / len(query_tokens)
    )


def _has_page_support(candidate: Candidate) -> bool:
    return bool(candidate.metadata.get("is_page_match"))


def _has_direct_text_support(
    candidate: Candidate,
    query_tokens: set[str],
) -> bool:
    if _has_page_support(candidate):
        return True

    if not query_tokens:
        return False

    overlap = _lexical_overlap_ratio(
        candidate,
        query_tokens,
    )

    if overlap > 0.0:
        return True

    # Fallback for stem/substring matches (e.g., galaxy vs galaxies, telescope vs telescopes)
    text_lower = (candidate.text or "").lower()
    for token in query_tokens:
        if len(token) >= 3 and (token in text_lower or (token[:-1] in text_lower if token.endswith("s") else False)):
            return True

    return False



def _has_lexical_support(
    candidate: Candidate,
    query_tokens: set[str],
) -> bool:
    if (
        candidate.bm25_rank is None
        or candidate.bm25_score is None
    ):
        return False

    try:
        score = float(
            candidate.bm25_score
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        math.isfinite(score)
        and score > 0.0
        and _has_direct_text_support(
            candidate,
            query_tokens,
        )
    )


def _dense_score(
    candidates: list[Candidate],
) -> tuple[float, float | None]:
    """
    Estimate relative Dense quality from distances.

    Lower distance = stronger semantic match.

    This is NOT treated as a probability.
    """
    distances: list[tuple[Candidate, float]] = []

    for candidate in candidates:
        if candidate.dense_distance is None:
            continue

        try:
            distance = float(
                candidate.dense_distance
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(distance):
            distances.append(
                (candidate, distance)
            )

    if not distances:
        return 0.0, None

    distances.sort(
        key=lambda item: item[1]
    )

    top_distance = distances[0][1]

    second_distance = (
        distances[1][1]
        if len(distances) > 1
        else top_distance
    )

    spread = max(
        0.0,
        second_distance - top_distance,
    )

    total_span = max(
        0.0,
        distances[-1][1] - top_distance,
    )

    if (
        total_span <= 1e-12
        and spread > 0
    ):
        separation = 1.0
    elif total_span > 1e-12:
        separation = spread / total_span
    else:
        separation = 0.0

    return (
        max(
            0.0,
            min(
                1.0,
                separation,
            ),
        ),
        top_distance,
    )


def _mode_ranked(
    candidates: list[Candidate],
    mode: str,
) -> list[Candidate]:
    """
    Rank candidates according to the selected retrieval mode.

    Hybrid  -> RRF
    Vector  -> Dense rank/distance
    BM25    -> BM25 rank/score
    """

    if mode == "vector":
        return sorted(
            candidates,
            key=lambda c: (
                c.dense_rank
                if c.dense_rank is not None
                else 999999,
                (
                    float(c.dense_distance)
                    if c.dense_distance is not None
                    else float("inf")
                ),
            ),
        )

    if mode == "bm25":
        return sorted(
            candidates,
            key=lambda c: (
                c.bm25_rank
                if c.bm25_rank is not None
                else 999999,
                -(
                    float(c.bm25_score)
                    if c.bm25_score is not None
                    else float("-inf")
                ),
            ),
        )

    return sorted(
        candidates,
        key=lambda c: (
            float(c.rrf_score),
            c.in_both_retrievers,
        ),
        reverse=True,
    )


def evaluate_evidence(
    candidates: list[Candidate],
    reranked: bool,
    search_text: str,
    retrieval_mode: str | None = None,
) -> tuple[list[Candidate], Evidence]:
    """
    Evaluate evidence according to the retrieval strategy.

    The explicit retrieval_mode takes priority over any inferred mode.

    Hybrid:
        Uses RRF + agreement + direct/lexical support.

    Vector:
        Uses Dense rank/distance + direct semantic/text support.
        RRF is NOT used.

    BM25:
        Uses BM25 rank/score + lexical/direct support.
        RRF is NOT used.

    CrossEncoder reranking is intentionally not required.
    """

    if not candidates:
        return [], Evidence(
            level="none",
            should_answer=False,
            top_score=0.0,
            score_source="none",
            supporting_chunks=0,
            top_gap=None,
            agreement=False,
            top_percentile=None,
            retrieval_mode=(
                get_retrieval_mode(retrieval_mode)
                if retrieval_mode is not None
                else "unknown"
            ),
            reason="No candidates were retrieved.",
        )

    # Use the explicit retrieval_mode when provided.
    # This is the canonical mode passed from answer_question().
    if retrieval_mode is not None:
        mode = get_retrieval_mode(retrieval_mode)
    else:
        # Fallback: infer from candidate fields when no explicit mode
        # is supplied (e.g. called directly in tests).
        has_dense = any(c.dense_rank is not None for c in candidates)
        has_bm25 = any(c.bm25_rank is not None for c in candidates)
        if has_dense and has_bm25:
            mode = "hybrid"
        elif has_bm25:
            mode = "bm25"
        else:
            mode = "vector"

    ranked = deduplicate_candidates(
        _mode_ranked(
            candidates,
            mode,
        )
    )

    if not ranked:
        return [], Evidence(
            level="none",
            should_answer=False,
            top_score=0.0,
            score_source="none",
            supporting_chunks=0,
            top_gap=None,
            agreement=False,
            top_percentile=None,
            retrieval_mode=mode,
            reason=(
                "No usable unique candidates "
                "remained after deduplication."
            ),
        )

    query_tokens = _query_tokens(
        search_text
    )

    top = ranked[0]

    agreement = bool(
        top.in_both_retrievers
    )

    direct = _has_direct_text_support(
        top,
        query_tokens,
    )

    lexical = _has_lexical_support(
        top,
        query_tokens,
    )

    # ==============================================================
    # HYBRID / RRF
    # ==============================================================

    if mode == "hybrid":

        top_score = float(
            top.rrf_score
        )

        score_source = "rrf"

        scores = [
            float(c.rrf_score)
            for c in ranked
            if math.isfinite(
                float(c.rrf_score)
            )
        ]

        top_percentile = (
            top_score_percentile(
                scores
            )
        )

        top_gap = None

        if len(ranked) > 1:
            top_gap = (
                top_score
                - float(
                    ranked[1].rrf_score
                )
            )

        # ----------------------------------------------------------
        # Minimum RRF floor
        # ----------------------------------------------------------

        if (
            not math.isfinite(
                top_score
            )
            or top_score
            < RRF_ABSTAIN_FLOOR
        ):
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=agreement,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "The best Hybrid candidate "
                    "did not reach the minimum "
                    "RRF evidence floor."
                ),
            )

        # ----------------------------------------------------------
        # Supporting candidates
        #
        # Agreement alone is NOT proof.
        # Direct/lexical evidence is also required.
        # ----------------------------------------------------------

        supported = [
            candidate
            for candidate in ranked
            if (
                candidate.in_both_retrievers
                or _has_lexical_support(
                    candidate,
                    query_tokens,
                )
                or _has_direct_text_support(
                    candidate,
                    query_tokens,
                )
            )
        ]

        if (
            not supported
            or top not in supported
        ):
            return [], Evidence(
                level="weak",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=agreement,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "Hybrid retrieval returned "
                    "candidates, but the best "
                    "candidate lacked sufficient "
                    "direct or lexical support."
                ),
            )

        # ----------------------------------------------------------
        # Strong
        #
        # Requires:
        # - verified page support OR
        # - both retrievers agree + RRF above strong threshold + direct/lexical support
        # ----------------------------------------------------------

        page_supported = _has_page_support(top)

        strong = bool(
            page_supported
            or (
                agreement
                and top_score
                >= STRONG_RRF_THRESHOLD
                and (
                    lexical
                    or direct
                )
            )
        )

        # ----------------------------------------------------------
        # Moderate
        #
        # RRF floor + at least one supporting signal.
        # ----------------------------------------------------------

        moderate = bool(
            page_supported
            or (
                top_score
                >= RRF_ABSTAIN_FLOOR
                and (
                    lexical
                    or direct
                    or agreement
                )
            )
        )

        level = (
            "strong"
            if strong
            else "moderate"
            if moderate
            else "weak"
        )

        if level == "weak":
            return [], Evidence(
                level="weak",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=agreement,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "The Hybrid evidence "
                    "was too weak to answer "
                    "reliably."
                ),
            )

        final = select_context_candidates(
            supported,
            reranked=False,
        )

        if not final:
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=agreement,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "No evidence-backed "
                    "context remained "
                    "after filtering."
                ),
            )

        return final, Evidence(
            level=level,
            should_answer=True,
            top_score=top_score,
            score_source=score_source,
            supporting_chunks=len(final),
            top_gap=top_gap,
            agreement=agreement,
            top_percentile=top_percentile,
            retrieval_mode=mode,
            reason=(
                f"Answer supported by verified content from page {top.metadata.get('target_page', 'requested')}."
                if page_supported
                else
                "Answer supported by agreement "
                "between Dense and BM25 with "
                "direct/lexical support."
                if strong
                else
                "Answer supported by Hybrid "
                "retrieval with direct or "
                "lexical evidence."
            ),
        )

    # ==============================================================
    # VECTOR / DENSE ONLY
    # ==============================================================

    if mode == "vector":

        # IMPORTANT:
        # Dense distance is NOT a probability.
        # Lower distance = stronger match.
        top_score = (
            float(top.dense_distance)
            if top.dense_distance is not None
            else float("inf")
        )

        score_source = (
            "dense_distance"
        )

        distances = [
            float(c.dense_distance)
            for c in ranked
            if (
                c.dense_distance is not None
                and math.isfinite(
                    float(
                        c.dense_distance
                    )
                )
            )
        ]

        # Convert lower-is-better distance
        # into higher-is-better values only
        # for percentile comparison.
        top_percentile = (
            top_score_percentile(
                [-d for d in distances]
            )
            if distances
            else None
        )

        top_gap = None

        if (
            len(ranked) > 1
            and top.dense_distance
            is not None
            and ranked[1].dense_distance
            is not None
        ):
            top_gap = (
                float(
                    ranked[1].dense_distance
                )
                - float(
                    top.dense_distance
                )
            )

        dense_strong, _ = _dense_score(
            ranked
        )

        # ----------------------------------------------------------
        # Strong Dense evidence
        #
        # Requires a very high Dense rank
        # AND meaningful separation
        # AND direct textual support.
        # ----------------------------------------------------------

        strong = bool(
            top.dense_rank is not None
            and top.dense_rank <= 1
            and dense_strong >= 0.5
            and direct
        )

        # ----------------------------------------------------------
        # Moderate Dense evidence
        #
        # Top few Dense candidates may answer,
        # but we do not call them strong merely
        # because they are rank #1.
        # ----------------------------------------------------------

        moderate = bool(
            top.dense_rank is not None
            and top.dense_rank <= 3
            and (
                direct
                or dense_strong >= 0.25
            )
        )

        if not moderate:
            return [], Evidence(
                level="weak",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=False,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "Vector retrieval found "
                    "nearby semantic neighbours, "
                    "but the evidence was not "
                    "strong enough to support "
                    "an answer."
                ),
            )

        final = select_context_candidates(
            ranked,
            reranked=False,
        )

        if not final:
            return [], Evidence(
                level="none",
                should_answer=False,
                top_score=top_score,
                score_source=score_source,
                supporting_chunks=0,
                top_gap=top_gap,
                agreement=False,
                top_percentile=top_percentile,
                retrieval_mode=mode,
                reason=(
                    "No usable Vector "
                    "context remained "
                    "after filtering."
                ),
            )

        level = (
            "strong"
            if strong
            else "moderate"
        )

        return final, Evidence(
            level=level,
            should_answer=True,
            top_score=top_score,
            score_source=score_source,
            supporting_chunks=len(final),
            top_gap=top_gap,
            agreement=False,
            top_percentile=top_percentile,
            retrieval_mode=mode,
            reason=(
                "Answer supported by a strong "
                "Dense retrieval signal and "
                "direct text support."
                if strong
                else
                "Answer supported by Dense "
                "retrieval with sufficient "
                "semantic/direct evidence."
            ),
        )

    # ==============================================================
    # BM25 ONLY
    # ==============================================================

    top_score = (
        float(top.bm25_score)
        if top.bm25_score is not None
        else 0.0
    )

    score_source = "bm25_score"

    scores = [
        float(c.bm25_score)
        for c in ranked
        if (
            c.bm25_score is not None
            and math.isfinite(
                float(c.bm25_score)
            )
        )
    ]

    top_percentile = (
        top_score_percentile(
            scores
        )
    )

    top_gap = None

    if (
        len(ranked) > 1
        and top.bm25_score is not None
        and ranked[1].bm25_score is not None
    ):
        top_gap = (
            top_score
            - float(
                ranked[1].bm25_score
            )
        )

    # --------------------------------------------------------------
    # Supporting candidates
    #
    # Check lexical or direct support across the ranked pool,
    # not just the single top-ranked-by-score candidate. A chunk
    # can win on raw term frequency without being the chunk that
    # actually answers the question.
    # --------------------------------------------------------------

    supported = [
        candidate
        for candidate in ranked
        if (
            _has_lexical_support(
                candidate,
                query_tokens,
            )
            or _has_direct_text_support(
                candidate,
                query_tokens,
            )
        )
    ]

    best_supported = supported[0] if supported else None

    # --------------------------------------------------------------
    # Strong BM25
    # --------------------------------------------------------------

    strong = bool(
        best_supported is not None
        and best_supported.bm25_rank == 1
        and _has_lexical_support(
            best_supported,
            query_tokens,
        )
    )

    # --------------------------------------------------------------
    # Moderate BM25
    # --------------------------------------------------------------

    moderate = bool(
        best_supported is not None
        and best_supported.bm25_rank is not None
        and best_supported.bm25_rank <= 5
    )

    if not moderate:
        return [], Evidence(
            level="weak",
            should_answer=False,
            top_score=top_score,
            score_source=score_source,
            supporting_chunks=0,
            top_gap=top_gap,
            agreement=False,
            top_percentile=top_percentile,
            retrieval_mode=mode,
            reason=(
                "BM25 found limited lexical "
                "evidence, so the result was "
                "not strong enough to answer "
                "reliably."
            ),
        )

    final = select_context_candidates(
        supported,
        reranked=False,
    )

    if not final:
        return [], Evidence(
            level="none",
            should_answer=False,
            top_score=top_score,
            score_source=score_source,
            supporting_chunks=0,
            top_gap=top_gap,
            agreement=False,
            top_percentile=top_percentile,
            retrieval_mode=mode,
            reason=(
                "No usable BM25 context "
                "remained after filtering."
            ),
        )

    level = (
        "strong"
        if strong
        else "moderate"
    )

    return final, Evidence(
        level=level,
        should_answer=True,
        top_score=top_score,
        score_source=score_source,
        supporting_chunks=len(final),
        top_gap=top_gap,
        agreement=False,
        top_percentile=top_percentile,
        retrieval_mode=mode,
        reason=(
            "Answer supported by direct "
            "lexical evidence in BM25 "
            "retrieval."
            if strong
            else
            "Answer supported by sufficient "
            "BM25 lexical evidence."
        ),
    )