"""Langfuse tracing for RAG pipeline runs (one trace per user query).

Langfuse Best Practices Applied:
1. Hierarchical trace structure (trace → span → observation)
2. Consistent naming conventions
3. Rich metadata for debugging
4. Error handling that doesn't break the app
5. Flush control for reliable data delivery
6. Score attachment for evaluation metrics
7. RAGAS integration for automated evaluation
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.rag_pipeline.observability_config import RetrievalMode, langfuse_enabled

logger = logging.getLogger(__name__)


def _safe_langfuse_import() -> Any | None:
    """Safely import Langfuse SDK with error handling."""
    try:
        # Use the operating system trust store (including locally installed CA
        # certificates) so HTTPS verification works on managed Windows hosts.
        import truststore

        truststore.inject_into_ssl()
        from langfuse import get_client
        return get_client
    except ImportError:
        logger.warning("Langfuse SDK not installed. Tracing will be disabled.")
        return None
    except Exception as e:
        logger.warning(f"Langfuse import error: {e}")
        return None


def _chunk_rows(items: list[dict[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    """Format chunk data for Langfuse observation output."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        filename = item.get("filename") or ""
        file_type = item.get("file_type") or ""
        page_number = item.get("page_number") or 0
        if (file_type == "pdf" or filename.lower().endswith(".pdf")) and page_number > 0:
            source_label = f"{filename} — p. {page_number}"
        else:
            source_label = item.get("source_label") or filename
        rows.append(
            {
                "rank": item.get("dense_rank")
                or item.get("bm25_rank")
                or index,
                "filename": filename,
                "chunk_id": item.get("chunk_id"),
                "text_preview": (item.get("text") or "")[:240],
                score_key: item.get(score_key),
                "rrf_score": item.get("rrf_score"),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "file_type": file_type,
                "page_number": page_number,
                "source_label": source_label,
            }
        )
    return rows


def _attach_ragas_scores(
    observation: Any,
    *,
    query: str,
    retrieval_mode: RetrievalMode,
    ragas_payload: dict[str, Any],
) -> None:
    """Attach RAGAS evaluation scores to a Langfuse observation.
    
    Langfuse Best Practices:
    - Use consistent naming: ragas_{mode}_{metric}
    - Include query context for traceability
    - Add metadata for filtering/analysis
    - Handle score normalization (0-1 range)
    - Skip metrics that couldn't be evaluated
    """
    scores = ragas_payload.get("scores") or {}
    skipped_metrics = ragas_payload.get("skipped_metrics", [])
    status = ragas_payload.get("status", "ok")
    
    # Log skipped metrics
    if skipped_metrics:
        logger.info(
            f"RAGAS skipped metrics for {retrieval_mode}: {skipped_metrics}"
        )
    
    for metric_name, value in scores.items():
        try:
            # Normalize score to 0-1 range if needed
            score_value = float(value)
            if score_value < 0 or score_value > 1:
                logger.warning(f"RAGAS score {metric_name} out of range: {score_value}")
                score_value = max(0.0, min(1.0, score_value))
            
            observation.score(
                name=f"ragas_{retrieval_mode}_{metric_name}",
                value=score_value,
                comment=f"query={query[:120]}",
                metadata={
                    "retrieval_method": retrieval_mode,
                    "query": query[:200],  # Truncate for storage
                    "metric": metric_name,
                    "ragas_status": status,
                    "evaluation_timestamp": time.time(),
                    "metrics_plan": ragas_payload.get("metrics_plan", {}),
                },
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to attach RAGAS score {metric_name}: {e}")


def _attach_deepeval_scores(
    observation: Any,
    *,
    query: str,
    retrieval_mode: RetrievalMode,
    deepeval_payload: dict[str, Any],
) -> None:
    """Attach DeepEval evaluation scores to a Langfuse observation.
    
    Langfuse Best Practices:
    - Use consistent naming: deepeval_{mode}_{metric}
    - Include query context for traceability
    - Add metadata for filtering/analysis
    - Handle score normalization (0-1 range)
    - Include reason/explanation from DeepEval judges
    """
    scores = deepeval_payload.get("scores") or {}
    status = deepeval_payload.get("status", "ok")
    
    for metric_name, metric_data in scores.items():
        try:
            # Extract score and reason from DeepEval format: {"score": float, "reason": str}
            if isinstance(metric_data, dict):
                score_value = float(metric_data.get("score", 0.0))
                reason = metric_data.get("reason", "")
            else:
                score_value = float(metric_data)
                reason = ""
            
            # Normalize score to 0-1 range if needed
            if score_value < 0 or score_value > 1:
                logger.warning(f"DeepEval score {metric_name} out of range: {score_value}")
                score_value = max(0.0, min(1.0, score_value))
            
            observation.score(
                name=f"deepeval_{retrieval_mode}_{metric_name}",
                value=score_value,
                comment=f"query={query[:120]} | reason={reason[:100]}" if reason else f"query={query[:120]}",
                metadata={
                    "retrieval_method": retrieval_mode,
                    "query": query[:200],  # Truncate for storage
                    "metric": metric_name,
                    "deepeval_status": status,
                    "evaluation_timestamp": time.time(),
                    "reason": reason[:300] if reason else "",
                },
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to attach DeepEval score {metric_name}: {e}")


def emit_langfuse_trace(
    *,
    query: str,
    retrieval_mode: RetrievalMode,
    response: dict[str, Any],
    total_latency_ms: float,
    retrieval_timings_ms: dict[str, float] | None = None,
    deepeval_by_mode: dict[str, dict[str, Any]] | None = None,
    ragas_by_mode: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Emit a complete Langfuse trace for a RAG pipeline execution.
    
    Langfuse Best Practices Applied:
    1. One trace per user query
    2. Hierarchical structure: query → retrieval → evidence → generation → evaluation
    3. Rich metadata for debugging and analysis
    4. Error handling that doesn't break the application
    5. Async flush for performance
    6. Trace ID return for correlation
    
    Hierarchy:
    └── rag-query (span)
        ├── retrieval-dense (retriever)
        ├── retrieval-bm25 (retriever)
        ├── retrieval-hybrid-rrf (retriever)
        ├── evidence-selection (span)
        ├── llm-answer (generation)
        └── ragas-eval-{mode} (evaluator) [optional]
    """
    if not langfuse_enabled():
        return None

    get_client = _safe_langfuse_import()
    if not get_client:
        return None

    trace = response.get("trace") or {}
    retrieval = trace.get("retrieval") or {}
    evidence = trace.get("evidence_gate") or response.get("evidence") or {}
    context = trace.get("context") or {}
    generation = trace.get("generation") or {}
    timings = retrieval_timings_ms or {}

    langfuse = get_client()
    trace_id: str | None = None

    # Extract common metadata
    trace_data = response.get("trace") or {}
    kb_state = trace_data.get("kb_state") or response.get("kb_state") or {}
    generation_data = trace_data.get("generation") or {}
    
    # Create root trace with comprehensive metadata
    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-query",
        input={
            "query": query,
            "retrieval_mode": retrieval_mode,
            "query_length": len(query),
        },
        metadata={
            # System information
            "response_type": response.get("response_type"),
            "total_latency_ms": round(total_latency_ms, 2),
            "trace_timestamp": time.time(),
            
            # Knowledge base state
            "kb_version": kb_state.get("version"),
            "kb_document_count": kb_state.get("document_count"),
            "kb_chroma_chunks": kb_state.get("chroma_chunks"),
            "kb_bm25_chunks": kb_state.get("bm25_chunks"),
            "kb_indexes_consistent": kb_state.get("indexes_consistent"),
            
            # Generation info
            "llm_model": generation_data.get("model"),
            "generation_status": generation_data.get("status"),
            
            # Performance metrics
            "has_ragas_evaluation": bool(ragas_by_mode),
            "retrieval_methods_evaluated": list(ragas_by_mode.keys()) if ragas_by_mode else [],
        },
    ) as root:
        trace_id = root.trace_id

        dense_executed = retrieval_mode in {"vector", "hybrid"} and bool(
            retrieval.get("raw_dense")
        )
        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="retrieval-dense",
            input={"query": trace.get("query", {}).get("prepared", query)},
            metadata={"latency_ms": timings.get("dense"), "executed": dense_executed},
        ) as dense_span:
            dense_span.update(
                output=_chunk_rows(
                    list(retrieval.get("raw_dense") or []),
                    score_key="dense_distance",
                )
            )

        bm25_executed = retrieval_mode in {"bm25", "hybrid"} and bool(
            retrieval.get("raw_bm25")
        )
        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="retrieval-bm25",
            input={"query": trace.get("query", {}).get("prepared", query)},
            metadata={"latency_ms": timings.get("bm25"), "executed": bm25_executed},
        ) as bm25_span:
            bm25_span.update(
                output=_chunk_rows(
                    list(retrieval.get("raw_bm25") or []),
                    score_key="bm25_score",
                )
            )

        hybrid_executed = retrieval_mode == "hybrid"
        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="retrieval-hybrid-rrf",
            input={
                "query": trace.get("query", {}).get("prepared", query),
                "rrf_k": retrieval.get("rrf_k"),
            },
            metadata={
                "latency_ms": timings.get("hybrid_fusion"),
                "executed": hybrid_executed,
                "rrf_k": retrieval.get("rrf_k"),
                "note": "rrf_score is fusion rank score, not confidence",
            },
        ) as hybrid_span:
            hybrid_span.update(
                output={
                    "rrf_calculations": retrieval.get("rrf_calculations") or [],
                    "fused_ranking": _chunk_rows(
                        list(retrieval.get("rrf_results") or []),
                        score_key="rrf_score",
                    ),
                }
            )

        with langfuse.start_as_current_observation(
            as_type="span",
            name="evidence-selection",
            input={"retrieval_mode": retrieval_mode},
        ) as evidence_span:
            evidence_span.update(
                output={
                    "level": evidence.get("level"),
                    "should_answer": evidence.get("should_answer"),
                    "selected_chunks": trace.get("selected_evidence") or [],
                    "context_chars": context.get("chars"),
                    "reason": evidence.get("reason"),
                }
            )

        gen_latency = timings.get("generation")
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-answer",
            model=generation.get("model"),
            input={
                "prompt": generation.get("user_instruction")
                or generation.get("question"),
                "context_chars": generation.get("context_chars")
                or context.get("chars"),
            },
            metadata={"latency_ms": gen_latency},
        ) as gen_span:
            gen_span.update(
                output={
                    "answer": response.get("raw_answer") or response.get("answer"),
                    "status": generation.get("status"),
                }
            )

        if deepeval_by_mode:
            for mode, payload in deepeval_by_mode.items():
                with langfuse.start_as_current_observation(
                    as_type="evaluator",
                    name=f"deepeval-eval-{mode}",
                    input={"query": query, "retrieval_method": mode},
                    metadata={
                        "status": payload.get("status"),
                    },
                ) as eval_span:
                    eval_span.update(output=payload.get("scores") or {})
                    _attach_deepeval_scores(
                        eval_span,
                        query=query,
                        retrieval_mode=mode,  # type: ignore[arg-type]
                        deepeval_payload=payload,
                    )

        if ragas_by_mode:
            for mode, payload in ragas_by_mode.items():
                with langfuse.start_as_current_observation(
                    as_type="evaluator",
                    name=f"ragas-eval-{mode}",
                    input={"query": query, "retrieval_method": mode},
                    metadata={
                        "metrics_plan": payload.get("metrics_plan"),
                        "status": payload.get("status"),
                    },
                ) as eval_span:
                    eval_span.update(output=payload.get("scores") or {})
                    _attach_ragas_scores(
                        eval_span,
                        query=query,
                        retrieval_mode=mode,  # type: ignore[arg-type]
                        ragas_payload=payload,
                    )

        root.update(
            output={
                "answer": response.get("raw_answer") or response.get("answer"),
                "retrieval_mode": retrieval_mode,
            }
        )

    try:
        langfuse.flush()
    except Exception:
        logger.exception("Langfuse flush failed.")

    return trace_id
