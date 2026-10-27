"""Rule-based single-controller agent for adaptive RAG retrieval."""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from src.rag_pipeline.evidence import evaluate_evidence
from src.rag_pipeline.models import Evidence
from src.rag_pipeline.query import (
    extract_chapter_query_info,
    extract_page_query_info,
    is_multi_intent_query,
    prepare_search_query,
)
from src.rag_pipeline.retrieval_runner import (
    RetrievalAttemptResult,
    RetrievalFilters,
    execute_retrieval_attempt,
)

if TYPE_CHECKING:
    from src.rag_pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2


@dataclass
class AgentAttemptRecord:
    attempt: int
    retrieval_mode: str
    retrieval_plan: str
    tools_called: list[str]
    search_text: str
    fingerprint: str
    evidence_sufficient: bool
    evidence_level: str
    evidence_reason: str
    should_answer: bool
    retry_reason: str = ""
    changes_from_previous: str = ""
    retrieval_ms: float = 0.0
    candidate_count: int = 0
    fused_count: int = 0
    page_match_found: bool = False
    chapter_match_found: bool = False


@dataclass
class AgentRetrievalPhase:
    """Output of the controller retrieval phase for the main pipeline."""

    question_type: str
    retrieval_plan: str
    search_text: str
    retrieval_mode: str
    attempts: list[AgentAttemptRecord] = field(default_factory=list)
    final_decision: str = "pending"
    retry_reason: str = ""
    stop_reason: str = ""
    limitation_message: str = ""
    evidence_assessment: dict[str, Any] = field(default_factory=dict)
    agent_activity: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    # Best attempt payloads (for downstream generation)
    attempt_result: RetrievalAttemptResult | None = None
    fused: list = field(default_factory=list)
    final_candidates: list = field(default_factory=list)
    evidence: Any = None
    page_info: dict[str, Any] | None = None
    chapter_info: dict[str, Any] | None = None
    page_match_found: bool = False
    chapter_match_found: bool = False
    reranked: bool = False


def _classify_question_type(
    question: str,
    page_info: dict[str, Any] | None,
    chapter_info: dict[str, Any] | None,
) -> str:
    if is_multi_intent_query(question):
        return "multi_intent"
    if page_info and page_info.get("is_page_query"):
        return "page_specific"
    if chapter_info and chapter_info.get("is_chapter_query"):
        return "chapter_specific"
    return "general"


def _build_retrieval_plan(
    question_type: str,
    user_mode: str,
    page_info: dict[str, Any] | None,
    chapter_info: dict[str, Any] | None,
) -> str:
    parts: list[str] = [f"base_mode={user_mode}"]
    if question_type == "page_specific" and page_info:
        parts.append(
            f"page={page_info.get('page_start')}"
            f"-{page_info.get('page_end')}"
        )
        if page_info.get("book_match"):
            parts.append(f"book={page_info['book_match']}")
        if page_info.get("is_pure_page_query"):
            parts.append("pure_page_lookup")
        elif page_info.get("semantic_query"):
            parts.append("page_plus_semantic")
    if question_type == "chapter_specific" and chapter_info:
        parts.append(f"chapter={chapter_info.get('chapter_num')}")
        if chapter_info.get("book_match"):
            parts.append(f"book={chapter_info['book_match']}")
    if question_type == "general":
        parts.append("hybrid_semantic")
    return "; ".join(parts)


def _explain_retrieval_plan(
    question_type: str,
    user_mode: str,
    page_info: dict[str, Any] | None,
    chapter_info: dict[str, Any] | None,
) -> str:
    if question_type == "page_specific" and page_info:
        p_start = page_info.get("page_start")
        p_end = page_info.get("page_end") or p_start
        book = page_info.get("book_match")
        book_clause = f" in '{book}'" if book else ""
        if page_info.get("is_pure_page_query"):
            return f"Exact page query for page {p_start}{book_clause}. Applying strict page metadata filter to prevent irrelevant chunk bleed."
        return f"Page query targeting page {p_start}–{p_end}{book_clause} combined with semantic concept ranking."
    if question_type == "chapter_specific" and chapter_info:
        ch = chapter_info.get("chapter_num")
        book = chapter_info.get("book_match")
        book_clause = f" of '{book}'" if book else ""
        return f"Chapter-scoped query for Chapter {ch}{book_clause}. Applying chapter filter to target section context."
    if question_type == "multi_intent":
        return "Multi-intent query with multiple sub-questions. Planning decomposed retrieval for each sub-topic."
    return f"Conceptual inquiry. Executing {user_mode} retrieval combining dense embeddings and lexical matching."


def _initial_filters(
    question_type: str,
    page_info: dict[str, Any] | None,
    chapter_info: dict[str, Any] | None,
) -> RetrievalFilters:
    filters = RetrievalFilters()
    if question_type == "page_specific" and page_info:
        filters.page_start = page_info.get("page_start")
        filters.page_end = page_info.get("page_end")
        filters.book_match = page_info.get("book_match")
        if page_info.get("is_pure_page_query"):
            filters.require_page_match = True
    if question_type == "chapter_specific" and chapter_info:
        filters.chapter_num = chapter_info.get("chapter_num")
        filters.book_match = chapter_info.get("book_match") or filters.book_match
    return filters


def _search_text_for_attempt(
    question: str,
    page_info: dict[str, Any] | None,
    attempt_index: int,
) -> str:
    prepared = prepare_search_query(question)
    base = str(prepared.get("search_query") or question).strip()

    if attempt_index == 0:
        return base

    if page_info and page_info.get("semantic_query"):
        semantic = page_info["semantic_query"].strip()
        if semantic:
            return semantic

    # Strip page/chapter anchors for a broader lexical pass
    text = base
    text = re.sub(
        r"\b(?:pages?|p\.?|pg\.?)\s*\d{1,4}(?:\s*(?:-|–|—|to)\s*\d{1,4})?\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:chapter|ch\.?)\s*\d{1,3}\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 8 else base


def _retry_mode_and_filters(
    *,
    attempt_index: int,
    user_mode: str,
    question_type: str,
    prev: AgentAttemptRecord,
    page_info: dict[str, Any] | None,
    chapter_info: dict[str, Any] | None,
    prev_fingerprint: str,
) -> tuple[str, RetrievalFilters, str]:
    """Return (mode, filters, human-readable change summary)."""
    filters = _initial_filters(question_type, page_info, chapter_info)

    if attempt_index == 1:
        if question_type == "page_specific" and not prev.page_match_found:
            is_pure = bool(page_info and page_info.get("is_pure_page_query"))
            if is_pure:
                # Do not broaden strict pure page queries to other pages
                return user_mode, filters, "Strict page query not found; abstaining."
            filters.require_page_match = False
            mode = "bm25" if user_mode == "vector" else user_mode
            change = (
                "Page metadata lookup returned no chunks; "
                f"retrying with {mode} search using semantic keywords "
                "without requiring an exact page match."
            )
            return mode, filters, change

        if prev.evidence_level in {"none", "weak"}:
            if user_mode == "hybrid":
                mode = "bm25"
                change = (
                    "Hybrid evidence was insufficient; "
                    "retrying with BM25-only lexical retrieval."
                )
            elif user_mode == "vector":
                mode = "hybrid"
                change = (
                    "Dense evidence was insufficient; "
                    "retrying with hybrid dense+BM25 fusion."
                )
            else:
                mode = "hybrid"
                change = (
                    "BM25 evidence was insufficient; "
                    "retrying with hybrid dense+BM25 fusion."
                )
            return mode, filters, change

    mode = user_mode
    change = "Retry with adjusted query wording."
    return mode, filters, change


def _plan_retry_reason(prev: AgentAttemptRecord) -> str:
    if not prev.should_answer:
        return prev.evidence_reason or "Retrieved evidence did not meet the answer threshold."
    return ""


class AgentController:
    """Single rule-based controller: plan → retrieve → evaluate → retry → stop."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        emit_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.emit_event = emit_event or (lambda event: None)

    def run_retrieval_phase(
        self,
        question: str,
        retrieval_mode: str,
    ) -> AgentRetrievalPhase:
        t0 = time.perf_counter()

        page_info = extract_page_query_info(question, self.pipeline.vector_store)
        chapter_info = extract_chapter_query_info(question, self.pipeline.vector_store)
        question_type = _classify_question_type(question, page_info, chapter_info)
        retrieval_plan = _build_retrieval_plan(
            question_type,
            retrieval_mode,
            page_info,
            chapter_info,
        )

        explanation = _explain_retrieval_plan(
            question_type,
            retrieval_mode,
            page_info,
            chapter_info,
        )

        # Emit plan event
        self.emit_event({
            "type": "plan",
            "question_type": question_type,
            "retrieval_plan": retrieval_plan,
            "explanation": explanation,
        })

        phase = AgentRetrievalPhase(
            question_type=question_type,
            retrieval_plan=retrieval_plan,
            search_text=_search_text_for_attempt(question, page_info, 0),
            retrieval_mode=retrieval_mode,
            page_info=page_info,
            chapter_info=chapter_info,
        )

        seen_fingerprints: set[str] = set()
        prev_record: AgentAttemptRecord | None = None
        best: tuple[AgentAttemptRecord, RetrievalAttemptResult, list, Any, list] | None = None

        for attempt_index in range(MAX_RETRIEVAL_ATTEMPTS):
            if attempt_index == 0:
                mode = retrieval_mode
                filters = _initial_filters(question_type, page_info, chapter_info)
                search_text = _search_text_for_attempt(question, page_info, 0)
                changes = ""
            else:
                assert prev_record is not None
                mode, filters, changes = _retry_mode_and_filters(
                    attempt_index=attempt_index,
                    user_mode=retrieval_mode,
                    question_type=question_type,
                    prev=prev_record,
                    page_info=page_info,
                    chapter_info=chapter_info,
                    prev_fingerprint=prev_record.fingerprint,
                )
                search_text = _search_text_for_attempt(question, page_info, attempt_index)
                phase.retry_reason = _plan_retry_reason(prev_record)

                # Emit adaptive retry decision event
                self.emit_event({
                    "type": "adaptive_decision",
                    "attempt": attempt_index + 1,
                    "new_mode": mode,
                    "new_search_text": search_text,
                    "strategy_change": changes,
                    "reason": phase.retry_reason or (prev_record.evidence_reason if prev_record else ""),
                })

            # Emit retrieval start event
            self.emit_event({
                "type": "retrieval",
                "attempt": attempt_index + 1,
                "mode": mode,
                "search_text": search_text,
            })

            attempt_result = execute_retrieval_attempt(
                self.pipeline,
                search_text=search_text,
                retrieval_mode=mode,
                filters=filters,
                page_info=page_info,
                chapter_info=chapter_info,
            )

            # Emit tool calls & results event
            self.emit_event({
                "type": "tool_calls",
                "attempt": attempt_index + 1,
                "mode": mode,
                "tools_called": attempt_result.tools_called,
                "search_text": search_text,
                "candidate_count": len(attempt_result.candidates),
                "fused_count": len(attempt_result.fused),
                "retrieval_ms": round(attempt_result.retrieval_ms, 1),
                "page_match_found": attempt_result.page_match_found,
                "chapter_match_found": attempt_result.chapter_match_found,
            })

            if attempt_result.fingerprint in seen_fingerprints:
                record = AgentAttemptRecord(
                    attempt=attempt_index + 1,
                    retrieval_mode=mode,
                    retrieval_plan=retrieval_plan,
                    tools_called=attempt_result.tools_called,
                    search_text=search_text,
                    fingerprint=attempt_result.fingerprint,
                    evidence_sufficient=False,
                    evidence_level="none",
                    evidence_reason="Skipped duplicate retrieval (same query and strategy).",
                    should_answer=False,
                    retry_reason=phase.retry_reason,
                    changes_from_previous=changes or "Duplicate attempt skipped.",
                    retrieval_ms=attempt_result.retrieval_ms,
                    candidate_count=len(attempt_result.candidates),
                    fused_count=len(attempt_result.fused),
                    page_match_found=attempt_result.page_match_found,
                    chapter_match_found=attempt_result.chapter_match_found,
                )
                phase.attempts.append(record)
                prev_record = record
                continue

            seen_fingerprints.add(attempt_result.fingerprint)

            reranked = self.pipeline._rerank(question, attempt_result.fused)
            final_candidates, evidence = self.pipeline._evaluate_evidence(
                attempt_result.fused,
                reranked,
                search_text,
                retrieval_mode=mode,
            )

            # Page-specific: do not answer from unrelated pages when user asked for a page
            if (
                question_type == "page_specific"
                and page_info
                and page_info.get("is_pure_page_query")
                and not attempt_result.page_match_found
            ):
                msg = (
                    "The requested page was not found in the index; "
                    "refusing to answer from other pages."
                )
                evidence = Evidence(
                    level="none",
                    should_answer=False,
                    top_score=evidence.top_score,
                    score_source=evidence.score_source,
                    supporting_chunks=0,
                    top_gap=evidence.top_gap,
                    agreement=evidence.agreement,
                    top_percentile=evidence.top_percentile,
                    retrieval_mode=evidence.retrieval_mode,
                    reason=msg,
                )
                final_candidates = []
                phase.stop_reason = "Page not found in index"
                phase.limitation_message = msg

            sufficient = bool(evidence.should_answer and final_candidates)

            # Emit evidence event
            self.emit_event({
                "type": "evidence",
                "attempt": attempt_index + 1,
                "level": evidence.level,
                "sufficient": sufficient,
                "reason": evidence.reason,
                "supporting_chunks": len(final_candidates),
                "top_score": round(evidence.top_score, 4) if getattr(evidence, "top_score", None) is not None else 0.0,
                "score_source": getattr(evidence, "score_source", mode),
            })

            record = AgentAttemptRecord(
                attempt=attempt_index + 1,
                retrieval_mode=mode,
                retrieval_plan=retrieval_plan,
                tools_called=attempt_result.tools_called,
                search_text=search_text,
                fingerprint=attempt_result.fingerprint,
                evidence_sufficient=sufficient,
                evidence_level=evidence.level,
                evidence_reason=evidence.reason,
                should_answer=evidence.should_answer,
                retry_reason=phase.retry_reason if attempt_index > 0 else "",
                changes_from_previous=changes,
                retrieval_ms=attempt_result.retrieval_ms,
                candidate_count=len(attempt_result.candidates),
                fused_count=len(attempt_result.fused),
                page_match_found=attempt_result.page_match_found,
                chapter_match_found=attempt_result.chapter_match_found,
            )
            phase.attempts.append(record)
            prev_record = record

            # Page-not-found check: immediately abstain if require_page_match is True and no page match was found
            if filters.require_page_match and not attempt_result.page_match_found:
                p_target = (page_info or {}).get("page_start") or filters.page_start or "specified"
                book_target = (page_info or {}).get("book_match") or filters.book_match
                book_clause = f" in '{book_target}'" if book_target else ""
                limitation_msg = (
                    f"Page {p_target}{book_clause} was not found in the index; "
                    "refusing to answer from unrelated pages."
                )
                phase.stop_reason = "page_not_found"
                phase.limitation_message = limitation_msg
                best = (
                    record,
                    attempt_result,
                    [],
                    evidence,
                    attempt_result.fused,
                )
                self.emit_event({
                    "type": "decision",
                    "decision": "abstain",
                    "evidence_level": "none",
                    "reason": limitation_msg,
                    "attempts_count": len(phase.attempts),
                })
                break

            if sufficient:
                best = (
                    record,
                    attempt_result,
                    final_candidates,
                    evidence,
                    attempt_result.fused,
                )
                break

            # Emit retry event if not last attempt
            if attempt_index + 1 < MAX_RETRIEVAL_ATTEMPTS:
                retry_reason = _plan_retry_reason(record)
                self.emit_event({
                    "type": "retry",
                    "attempt": attempt_index + 1,
                    "reason": retry_reason,
                })

            if best is None or (
                evidence.top_score
                > (best[3].top_score if best[3] else 0)
            ):
                best = (
                    record,
                    attempt_result,
                    final_candidates,
                    evidence,
                    attempt_result.fused,
                )

            if attempt_index + 1 >= MAX_RETRIEVAL_ATTEMPTS:
                break

        total_ms = (time.perf_counter() - t0) * 1000

        if best and best[0].evidence_sufficient:
            (
                _rec,
                attempt_result,
                final_candidates,
                evidence,
                fused,
            ) = best
            phase.final_decision = "answer"
            phase.attempt_result = attempt_result
            phase.fused = fused
            phase.final_candidates = final_candidates
            phase.evidence = evidence
            phase.search_text = attempt_result.search_text
            phase.retrieval_mode = attempt_result.retrieval_mode
            phase.page_match_found = attempt_result.page_match_found
            phase.chapter_match_found = attempt_result.chapter_match_found
            phase.reranked = self.pipeline._rerank(question, fused)
        else:
            phase.final_decision = "abstain"
            if best:
                (
                    _rec,
                    attempt_result,
                    final_candidates,
                    evidence,
                    fused,
                ) = best
                phase.attempt_result = attempt_result
                phase.fused = fused
                phase.final_candidates = final_candidates
                phase.evidence = evidence
                phase.search_text = attempt_result.search_text
                phase.retrieval_mode = attempt_result.retrieval_mode
                phase.page_match_found = attempt_result.page_match_found
                phase.chapter_match_found = attempt_result.chapter_match_found
            elif phase.attempts:
                last = phase.attempts[-1]
                phase.evidence_assessment = {
                    "level": last.evidence_level,
                    "should_answer": False,
                    "reason": last.evidence_reason,
                }

        if phase.evidence is not None:
            phase.evidence_assessment = phase.evidence.as_dict()

        # Emit final decision event (if not already emitted during short-circuit abstain)
        if phase.stop_reason != "page_not_found":
            self.emit_event({
                "type": "decision",
                "decision": phase.final_decision,
                "evidence_level": phase.evidence.level if phase.evidence else "none",
                "reason": phase.evidence.reason if phase.evidence else "No candidates found across all retrieval attempts",
                "attempts_count": len(phase.attempts),
            })

        phase.timings_ms = {
            "controller_total": round(total_ms, 2),
            "last_retrieval": round(
                (phase.attempt_result.retrieval_ms if phase.attempt_result else 0),
                2,
            ),
        }

        phase.agent_activity = {
            "question_type": phase.question_type,
            "retrieval_plan": phase.retrieval_plan,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "retrieval_mode": a.retrieval_mode,
                    "tools_called": a.tools_called,
                    "search_text": a.search_text,
                    "evidence_sufficient": a.evidence_sufficient,
                    "evidence_level": a.evidence_level,
                    "evidence_reason": a.evidence_reason,
                    "retry_reason": a.retry_reason,
                    "changes_from_previous": a.changes_from_previous,
                    "retrieval_ms": a.retrieval_ms,
                    "candidate_count": a.candidate_count,
                    "fused_count": a.fused_count,
                    "page_match_found": a.page_match_found,
                    "chapter_match_found": a.chapter_match_found,
                }
                for a in phase.attempts
            ],
            "attempt_count": len(phase.attempts),
            "retrieval_mode": phase.retrieval_mode,
            "evidence_assessment": phase.evidence_assessment,
            "retry_reason": phase.retry_reason,
            "stop_reason": phase.stop_reason,
            "final_decision": phase.final_decision,
            "timings_ms": phase.timings_ms,
            "page_match_found": phase.page_match_found,
            "chapter_match_found": phase.chapter_match_found,
        }

        return phase
