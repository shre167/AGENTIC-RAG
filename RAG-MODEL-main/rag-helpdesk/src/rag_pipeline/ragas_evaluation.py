from __future__ import annotations

import logging
import math
from typing import Any

from src.config import (
    EMBEDDING_MODEL,
    GE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from src.rag_pipeline.observability_config import RetrievalMode

logger = logging.getLogger(__name__)

# RAGAS 0.4.3:
# Reference-free metrics do not require ground-truth data.
_METRICS_WITHOUT_REFERENCE: list[Any] = []

# Reference-based metrics require benchmark/reference data.
_METRICS_WITH_REFERENCE: list[Any] = []


def _load_metric_classes() -> None:
    """Lazy-load RAGAS metrics.

    Reference-free:
        - faithfulness
        - answer_relevancy

    Reference-based:
        - context_precision
        - context_recall

    IMPORTANT:
    RAGAS AnswerRelevancy uses `strictness` to determine how many
    artificial questions it generates from the answer. Gemini's
    OpenAI-compatible endpoint does not allow multiple candidates,
    so AnswerRelevancy is explicitly configured with strictness=1.
    """
    global _METRICS_WITHOUT_REFERENCE, _METRICS_WITH_REFERENCE

    if _METRICS_WITHOUT_REFERENCE:
        return

    try:
        from ragas.metrics import (
            AnswerRelevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # RAGAS answer_relevancy normally has strictness=3.
        # That causes RAGAS to request multiple generated candidates,
        # which Gemini rejects with:
        #
        # "Multiple candidates is not enabled for this model"
        #
        # strictness=1 means only one question is generated.
        try:
            answer_relevancy_metric = AnswerRelevancy(strictness=1)
        except Exception:
            # Compatibility fallback for installations where the
            # singleton/class API differs.
            from ragas.metrics import answer_relevancy

            answer_relevancy.strictness = 1
            answer_relevancy_metric = answer_relevancy

        _METRICS_WITHOUT_REFERENCE = [
            faithfulness,
            answer_relevancy_metric,
        ]

        _METRICS_WITH_REFERENCE = [
            context_precision,
            context_recall,
        ]

        logger.info(
            "RAGAS metrics loaded: faithfulness, "
            "answer_relevancy(strictness=1), "
            "context_precision, context_recall"
        )

    except ImportError as exc:
        logger.warning("RAGAS metrics import failed: %s", exc)
        _METRICS_WITHOUT_REFERENCE = []
        _METRICS_WITH_REFERENCE = []
    except Exception as exc:
        logger.exception("Failed to initialize RAGAS metrics: %s", exc)
        _METRICS_WITHOUT_REFERENCE = []
        _METRICS_WITH_REFERENCE = []


def _build_ragas_llm() -> Any:
    """Build the LLM used by RAGAS.

    Uses the project's existing LLM configuration.

    The underlying Gemini-compatible endpoint is configured with
    n=1 as an additional safeguard. AnswerRelevancy itself is also
    configured with strictness=1 above.
    """
    api_key = GE_API_KEY or LLM_API_KEY

    if not api_key:
        logger.warning("RAGAS LLM: API key not configured.")
        return None

    if not LLM_BASE_URL:
        logger.warning("RAGAS LLM: LLM_BASE_URL not configured.")
        return None

    if not LLM_MODEL:
        logger.warning("RAGAS LLM: LLM_MODEL not configured.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=LLM_BASE_URL.rstrip("/") + "/",
            model=LLM_MODEL,
            temperature=0.0,
            n=1,
        )

        return llm

    except Exception as exc:
        logger.exception(
            "Failed to construct RAGAS LLM wrapper: %s",
            exc,
        )
        return None


def _build_ragas_embeddings() -> Any:
    """Build embeddings for RAGAS.

    Uses the project's EmbeddingService so Gemini embeddings are
    handled consistently with the rest of the RAG pipeline.
    """
    api_key = GE_API_KEY or LLM_API_KEY

    if not api_key or not EMBEDDING_MODEL:
        logger.warning(
            "RAGAS embeddings: API key or embedding model not configured."
        )
        return None

    if not LLM_BASE_URL:
        logger.warning(
            "RAGAS embeddings: LLM_BASE_URL not configured."
        )
        return None

    try:
        from src.embeddings import EmbeddingService

        embedding_service = EmbeddingService(
            api_key=api_key,
            model=EMBEDDING_MODEL,
            base_url=LLM_BASE_URL,
        )

        return embedding_service

    except Exception as exc:
        logger.warning(
            "Failed to construct RAGAS embeddings: %s",
            exc,
        )
        return None


def _contexts_from_response(response: dict[str, Any]) -> list[str]:
    """Extract the actual contexts used by the pipeline.

    Priority:
        1. trace.context.final_chunks
        2. response.retrieved_chunks
        3. trace.selected_evidence
    """
    trace = response.get("trace") or {}
    context_block = trace.get("context") or {}

    chunks = (
        context_block.get("final_chunks")
        or response.get("retrieved_chunks")
        or []
    )

    texts: list[str] = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or ""

            if text:
                texts.append(str(text))

        elif chunk:
            texts.append(str(chunk))

    # Fallback if final_chunks/retrieved_chunks were unavailable.
    if not texts:
        for candidate in trace.get("selected_evidence") or []:
            if isinstance(candidate, dict) and candidate.get("text"):
                texts.append(str(candidate["text"]))

    return texts


def _reference_fields(
    labels: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract reference/ground-truth fields from benchmark labels."""
    if not labels:
        return {}

    fields: dict[str, Any] = {}

    if labels.get("reference_answer"):
        fields["ground_truths"] = [
            str(labels["reference_answer"])
        ]

    if labels.get("ground_truth"):
        fields["ground_truths"] = [
            str(labels["ground_truth"])
        ]

    contexts = (
        labels.get("reference_contexts")
        or labels.get("ground_truth_contexts")
    )

    if contexts:
        fields["ground_truths_contexts"] = [
            str(c) for c in contexts
        ]

    return fields


def metrics_plan(
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the RAGAS metrics that will run."""
    _load_metric_classes()

    has_reference = bool(_reference_fields(labels))

    runnable = [
        getattr(metric, "name", str(metric))
        for metric in _METRICS_WITHOUT_REFERENCE
    ]

    reference_metrics = [
        getattr(metric, "name", str(metric))
        for metric in _METRICS_WITH_REFERENCE
    ]

    skipped: dict[str, str] = {}

    if has_reference:
        runnable.extend(reference_metrics)
    else:
        for metric_name in reference_metrics:
            skipped[metric_name] = (
                "requires reference answer/contexts "
                "in benchmark labels"
            )

    return {
        "runnable_metrics": runnable,
        "skipped_metrics": skipped,
        "has_reference_labels": has_reference,
        "reference_available": has_reference,
    }


def _is_valid_score(value: Any) -> bool:
    """Return True only for finite numeric metric scores."""
    try:
        numeric_value = float(value)
        return math.isfinite(numeric_value)
    except (TypeError, ValueError):
        return False


def evaluate_response_with_ragas(
    *,
    query: str,
    response: dict[str, Any],
    retrieval_mode: RetrievalMode,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run RAGAS evaluation on one pipeline response.

    Uses only the actual answer and retrieved contexts.

    Reference-free metrics:
        - faithfulness
        - answer_relevancy

    Reference-based metrics:
        - context_precision
        - context_recall

    Reference-based metrics are skipped when benchmark labels are
    unavailable. No ground truth is fabricated.
    """
    _load_metric_classes()

    answer = (
        response.get("raw_answer")
        or response.get("answer")
        or ""
    )

    contexts = _contexts_from_response(response)

    if not answer or not contexts:
        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "status": "skipped",
            "reason": "missing answer or retrieved contexts",
            "metrics_plan": metrics_plan(labels),
            "scores": {},
            "skipped_metrics": [],
        }

    llm = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()

    if llm is None:
        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "status": "skipped",
            "reason": "RAGAS LLM not configured",
            "metrics_plan": metrics_plan(labels),
            "scores": {},
            "skipped_metrics": [],
        }

    # RAGAS 0.4.x dataset.
    data_dict: dict[str, list[Any]] = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }

    # Add references only when they actually exist.
    ref_fields = _reference_fields(labels)

    if ref_fields:
        data_dict.update(ref_fields)

    # Reference-free metrics always run.
    metrics = list(_METRICS_WITHOUT_REFERENCE)

    skipped_metrics: list[str] = []

    # Reference-based metrics run only when references are available.
    if ref_fields:
        metrics.extend(_METRICS_WITH_REFERENCE)
    else:
        for metric in _METRICS_WITH_REFERENCE:
            metric_name = getattr(metric, "name", str(metric))
            skipped_metrics.append(metric_name)

    if not metrics:
        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "status": "skipped",
            "reason": "No RAGAS metrics available",
            "metrics_plan": metrics_plan(labels),
            "scores": {},
            "skipped_metrics": skipped_metrics,
        }

    try:
        from datasets import Dataset
        from ragas import evaluate

        dataset = Dataset.from_dict(data_dict)

        logger.info(
            "Running RAGAS evaluation for mode=%s with metrics=%s",
            retrieval_mode,
            [
                getattr(metric, "name", str(metric))
                for metric in metrics
            ],
        )

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

        scores: dict[str, float] = {}

        # RAGAS 0.4.3 normally exposes result.scores as:
        # [{"faithfulness": ..., "answer_relevancy": ...}]
        scores_list = getattr(result, "scores", None)

        if isinstance(scores_list, list) and scores_list:
            row = scores_list[0]

            if isinstance(row, dict):
                for metric in metrics:
                    metric_name = getattr(
                        metric,
                        "name",
                        str(metric),
                    )

                    if metric_name in row:
                        value = row[metric_name]

                        if _is_valid_score(value):
                            scores[metric_name] = float(value)

        # Compatibility fallback for alternate EvaluationResult formats.
        if not scores:
            for metric in metrics:
                metric_name = getattr(
                    metric,
                    "name",
                    str(metric),
                )

                try:
                    value = result[metric_name]

                    if hasattr(value, "iloc"):
                        value = value.iloc[0]
                    elif isinstance(value, (list, tuple)):
                        if not value:
                            continue
                        value = value[0]

                    if _is_valid_score(value):
                        scores[metric_name] = float(value)

                except (KeyError, TypeError, AttributeError):
                    continue

        # IMPORTANT:
        # RAGAS can complete while an individual metric internally fails,
        # resulting in NaN. Do not call that a successful evaluation.
        expected_metric_names = [
            getattr(metric, "name", str(metric))
            for metric in metrics
        ]

        failed_metrics = [
            name
            for name in expected_metric_names
            if name not in scores
        ]

        if failed_metrics:
            return {
                "retrieval_mode": retrieval_mode,
                "query": query,
                "status": "error",
                "error": (
                    "RAGAS did not produce valid scores for: "
                    + ", ".join(failed_metrics)
                ),
                "metrics_plan": metrics_plan(labels),
                "scores": scores,
                "skipped_metrics": skipped_metrics,
                "failed_metrics": failed_metrics,
            }

        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "status": "ok",
            "metrics_plan": metrics_plan(labels),
            "scores": scores,
            "skipped_metrics": skipped_metrics,
        }

    except Exception as exc:
        logger.exception(
            "RAGAS evaluation failed for mode=%s",
            retrieval_mode,
        )

        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "status": "error",
            "error": str(exc),
            "metrics_plan": metrics_plan(labels),
            "scores": {},
            "skipped_metrics": skipped_metrics,
        }