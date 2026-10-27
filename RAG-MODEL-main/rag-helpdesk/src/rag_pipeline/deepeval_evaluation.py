"""LLM-judged RAG metrics with no fabricated ground truth.

The project uses a small OpenAI-compatible adapter so it can evaluate through
the configured model endpoint without relying on DeepEval's model wrappers.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from src.config import GE_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.rag_pipeline.observability_config import RetrievalMode

logger = logging.getLogger(__name__)

# Sentinel strings that indicate the pipeline itself failed to generate an answer.
_LLM_FAILURE_MARKERS = (
    "language model request failed",
    "could not generate",
)


class DeepEvalLLMAdapter:
    """Adapter to make ChatOpenAI compatible with DeepEval's LLM interface.

    This adapter keeps metric evaluation compatible with the configured
    OpenAI-compatible model endpoint.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI-compatible client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/") + "/",
            )
        except Exception as e:
            logger.warning(f"Failed to init OpenAI client: {e}")
            self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM."""
        if not self.client:
            return ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""


def _build_deepeval_llm() -> Any:
    """Build a minimal LLM adapter for DeepEval evaluation.

    Uses direct OpenAI API instead of langchain to avoid import issues.
    """
    api_key = GE_API_KEY or LLM_API_KEY
    if not api_key:
        logger.warning("DeepEval LLM: API key not configured.")
        return None
    if not LLM_BASE_URL:
        logger.warning("DeepEval LLM: LLM_BASE_URL not configured.")
        return None
    if not LLM_MODEL:
        logger.warning("DeepEval LLM: LLM_MODEL not configured.")
        return None

    try:
        return DeepEvalLLMAdapter(
            api_key=api_key,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
        )
    except Exception as exc:
        logger.exception("Failed to construct DeepEval LLM: %s", exc)
        return None


def _parse_score_and_reason(response: str, fallback_reason: str) -> tuple[float, str]:
    """Parse a '<score> <reason>' response string.

    Returns (score_float, reason_str). Score is clamped to [0, 1].
    """
    if not response or not response.strip():
        return 0.0, "Evaluation failed: the model returned an empty response."
    try:
        parts = response.strip().split(None, 1)
        score = float(parts[0])
        reason = parts[1] if len(parts) > 1 else fallback_reason
        score = max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        score = 0.0
        reason = "Evaluation failed: the response did not start with a numeric score."
    return score, reason


def evaluate_rag_response(
    *,
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    retrieval_mode: RetrievalMode,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a RAG response using 10 metrics implemented as direct LLM calls.

    All 10 metrics run concurrently via ThreadPoolExecutor(max_workers=10).

    Uses the actual query, answer, and contexts from the pipeline.
    No fabricated ground truth or additional generation.

    Metrics (Reference-Free, all implemented directly):
    - faithfulness        — answer grounded in context?
    - answer_relevancy    — answer relevant to query?
    - contextual_relevancy  — retrieved chunks relevant to query?
    - contextual_precision  — best chunks ranked first?
    - contextual_recall     — context covers what the answer needs?
    - answer_correctness    — factually aligned with retrieved evidence?
    - answer_completeness   — all parts of the query addressed?
    - citation_correctness  — bracketed citations point to supporting passages?
    - citation_completeness — all major claims have a citation?
    - groundedness          — no hallucinated statements?

    Returns dict with:
        - status: "ok" | "skipped" | "error"
        - scores: {metric_name: {"score": float, "reason": str}}
        - retrieval_mode: The retrieval method used
        - error: Error message if status is "error"
    """
    if not query or not answer or not retrieved_contexts:
        return {
            "status": "skipped",
            "reason": "missing query, answer, or contexts",
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

    # Skip evaluation when the pipeline itself failed to produce an answer.
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in _LLM_FAILURE_MARKERS):
        return {
            "status": "skipped",
            "reason": "LLM generation failed",
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

    llm = _build_deepeval_llm()
    if llm is None or not llm.client:
        return {
            "status": "skipped",
            "reason": "DeepEval LLM not configured",
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }

    try:
        # ------------------------------------------------------------------
        # Build all metric callables up-front (closed over shared inputs)
        # ------------------------------------------------------------------

        context_text = "\n".join(retrieved_contexts[:3])
        context_top3 = "\n---\n".join(
            f"[Chunk {i + 1}]: {c[:300]}" for i, c in enumerate(retrieved_contexts[:3])
        )
        context_full = "\n---\n".join(
            f"[Chunk {i + 1}]: {c[:300]}" for i, c in enumerate(retrieved_contexts)
        )
        ctx_snippets = "\n---\n".join(
            f"[Chunk {i + 1}]: {c[:300]}" for i, c in enumerate(retrieved_contexts)
        )
        ctx_ranked = "\n---\n".join(
            f"[Rank {i + 1}]: {c[:300]}" for i, c in enumerate(retrieved_contexts)
        )
        ctx_indexed = "\n---\n".join(
            f"[{i + 1}]: {c[:300]}" for i, c in enumerate(retrieved_contexts)
        )

        # ------------------------------------------------------------------
        # 1. FAITHFULNESS
        # What it measures: Whether the answer is grounded in the retrieved
        #   context and does not contradict it.
        # Why it matters: A faithful answer means the model did not invent
        #   facts beyond what was retrieved.
        # Inputs: answer + retrieved_contexts (top 3)
        # How evaluated: LLM checks each claim in the answer against the
        #   context text and rates overall grounding.
        # Score: 0 = completely unfaithful / contradictory; 1 = fully grounded.
        # ------------------------------------------------------------------
        def _eval_faithfulness() -> dict[str, Any]:
            try:
                faithfulness_prompt = f"""
Evaluate if the following answer is faithful to the provided context.
Answer faithfulness means the answer is grounded in the context and does not contradict it.

Context:
{context_text}

Answer:
{answer}

Rate the faithfulness on a scale of 0 to 1, where 1 is completely faithful.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                faith_score, faith_reason = _parse_score_and_reason(
                    llm.generate(faithfulness_prompt),
                    "Faithfulness evaluated",
                )
                return {"score": faith_score, "reason": faith_reason}
            except Exception as exc:
                logger.warning("Faithfulness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 2. ANSWER RELEVANCY
        # What it measures: Whether the answer directly addresses the query.
        # Why it matters: A high-faithfulness answer that ignores the
        #   question is still useless.
        # Inputs: query + answer
        # How evaluated: LLM checks whether the answer responds to the
        #   specific question asked.
        # Score: 0 = completely irrelevant; 1 = fully responsive.
        # ------------------------------------------------------------------
        def _eval_answer_relevancy() -> dict[str, Any]:
            try:
                relevancy_prompt = f"""
Evaluate if the following answer is relevant and responsive to the query.
Answer relevancy means the answer directly addresses the query.

Query:
{query}

Answer:
{answer}

Rate the relevancy on a scale of 0 to 1, where 1 is completely relevant.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                rel_score, rel_reason = _parse_score_and_reason(
                    llm.generate(relevancy_prompt),
                    "Relevancy evaluated",
                )
                return {"score": rel_score, "reason": rel_reason}
            except Exception as exc:
                logger.warning("Answer relevancy evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 3. CONTEXTUAL RELEVANCY
        # What it measures: How many of the retrieved passages are topically
        #   on-point for the question asked.
        # Why it matters: Irrelevant chunks dilute generation context and
        #   increase the risk of off-topic or confabulated answers.
        # Inputs: query + retrieved_contexts (all)
        # How evaluated: LLM inspects each retrieved passage against the
        #   query topic and rates the overall proportion that are relevant.
        # Score: 0 = all chunks are off-topic; 1 = every chunk is relevant.
        # ------------------------------------------------------------------
        def _eval_contextual_relevancy() -> dict[str, Any]:
            try:
                contextual_relevancy_prompt = f"""
Evaluate how relevant the retrieved passages are to the query.
Contextual relevancy measures what proportion of retrieved passages are actually on-topic for the question.

Query:
{query}

Retrieved Passages:
{ctx_snippets}

Rate the contextual relevancy on a scale of 0 to 1, where 1 means all passages are relevant.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                cr_score, cr_reason = _parse_score_and_reason(
                    llm.generate(contextual_relevancy_prompt),
                    "Contextual relevancy evaluated",
                )
                return {"score": cr_score, "reason": cr_reason}
            except Exception as exc:
                logger.warning("Contextual relevancy evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 4. CONTEXTUAL PRECISION
        # What it measures: Whether the most useful evidence appears at the
        #   top of the retrieved set (ranking quality).
        # Why it matters: Generation models attend more strongly to early
        #   context; relevant chunks ranked low are effectively wasted.
        # Inputs: query + retrieved_contexts (ordered, index = rank)
        # How evaluated: LLM determines which chunks are relevant to the
        #   query and checks whether they appear before irrelevant ones.
        # Score: 0 = relevant chunks all at the bottom; 1 = perfectly ranked.
        # ------------------------------------------------------------------
        def _eval_contextual_precision() -> dict[str, Any]:
            try:
                contextual_precision_prompt = f"""
Evaluate whether the retrieved passages are ranked in order of relevance to the query.
Contextual precision measures if the most useful chunks appear at the top of the ranked list.

Query:
{query}

Retrieved Passages (in retrieval order):
{ctx_ranked}

Rate the contextual precision on a scale of 0 to 1, where 1 means the most relevant passages are ranked highest.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                cp_score, cp_reason = _parse_score_and_reason(
                    llm.generate(contextual_precision_prompt),
                    "Contextual precision evaluated",
                )
                return {"score": cp_score, "reason": cp_reason}
            except Exception as exc:
                logger.warning("Contextual precision evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 5. CONTEXTUAL RECALL
        # What it measures: How much of the answer's information is
        #   traceable back to the retrieved passages.
        # Why it matters: Low recall means the answer relies on knowledge
        #   not present in the retrieved set — a hallucination signal.
        # Inputs: query + answer + retrieved_contexts
        # How evaluated: LLM checks each factual statement in the answer
        #   against the retrieved passages to estimate coverage.
        # Score: 0 = answer not supported by context at all; 1 = fully covered.
        # ------------------------------------------------------------------
        def _eval_contextual_recall() -> dict[str, Any]:
            try:
                contextual_recall_prompt = f"""
Evaluate whether the retrieved passages contain enough information to support the given answer.
Contextual recall measures how much of the answer's information is traceable to the retrieved context.

Query:
{query}

Retrieved Passages:
{context_full}

Answer:
{answer}

Rate the contextual recall on a scale of 0 to 1, where 1 means the answer is fully supported by the retrieved passages.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                rcl_score, rcl_reason = _parse_score_and_reason(
                    llm.generate(contextual_recall_prompt),
                    "Contextual recall evaluated",
                )
                return {"score": rcl_score, "reason": rcl_reason}
            except Exception as exc:
                logger.warning("Contextual recall evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 6. ANSWER CORRECTNESS
        # What it measures: Factual alignment between the answer and the
        #   retrieved evidence passages.
        # Why it matters: An answer can be relevant and well-structured yet
        #   still contain factual errors relative to the source material.
        # Inputs: query + answer + retrieved_contexts (top 3)
        # How evaluated: LLM cross-checks specific claims in the answer
        #   against the retrieved passages for factual accuracy.
        # Score: 0 = factually wrong relative to context; 1 = fully correct.
        # ------------------------------------------------------------------
        def _eval_answer_correctness() -> dict[str, Any]:
            try:
                answer_correctness_prompt = f"""
Evaluate whether the answer is factually correct based on the retrieved evidence.
Answer correctness measures factual alignment between the answer and the source passages.

Query:
{query}

Retrieved Evidence:
{context_top3}

Answer:
{answer}

Rate the answer correctness on a scale of 0 to 1, where 1 means all facts in the answer are accurate relative to the evidence.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                ac_score, ac_reason = _parse_score_and_reason(
                    llm.generate(answer_correctness_prompt),
                    "Answer correctness evaluated",
                )
                return {"score": ac_score, "reason": ac_reason}
            except Exception as exc:
                logger.warning("Answer correctness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 7. ANSWER COMPLETENESS
        # What it measures: Whether the answer fully resolves the query or
        #   leaves sub-questions unanswered.
        # Why it matters: Partial answers mislead users into thinking a
        #   question has been fully answered when it has not.
        # Inputs: query + answer
        # How evaluated: LLM decomposes the query into its components and
        #   checks whether each part is addressed by the answer.
        # Score: 0 = major parts of the query left unanswered; 1 = fully addressed.
        # ------------------------------------------------------------------
        def _eval_answer_completeness() -> dict[str, Any]:
            try:
                answer_completeness_prompt = f"""
Evaluate whether the answer completely addresses all parts of the query.
Answer completeness measures if the answer fully resolves the question without leaving sub-questions unanswered.

Query:
{query}

Answer:
{answer}

Rate the answer completeness on a scale of 0 to 1, where 1 means all parts of the query are fully addressed.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                acomp_score, acomp_reason = _parse_score_and_reason(
                    llm.generate(answer_completeness_prompt),
                    "Answer completeness evaluated",
                )
                return {"score": acomp_score, "reason": acomp_reason}
            except Exception as exc:
                logger.warning("Answer completeness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 8. CITATION CORRECTNESS
        # What it measures: Whether bracketed citations [1], [2], etc. in
        #   the answer point to passages that actually support the claim
        #   they are attached to.
        # Why it matters: Incorrect citations give a false sense of
        #   grounding; users who follow citations will find irrelevant text.
        # Inputs: answer + retrieved_contexts (indexed to match [N] refs)
        # How evaluated: LLM cross-references each [N] citation in the
        #   answer text with chunk N in the retrieved list to verify support.
        # Score: 0 = all citations are misleading; 1 = every citation is accurate.
        # ------------------------------------------------------------------
        def _eval_citation_correctness() -> dict[str, Any]:
            try:
                citation_correctness_prompt = f"""
Evaluate whether the bracketed citations in the answer correctly reference supporting passages.
Citation correctness measures if [1], [2], etc. in the answer actually point to passages that support the cited claim.

Retrieved Passages (numbered to match citations):
{ctx_indexed}

Answer (may contain [1], [2] style citations):
{answer}

Rate the citation correctness on a scale of 0 to 1, where 1 means every citation points to a truly supporting passage.
If the answer has no citations, score 0.5 (neutral).
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                cc_score, cc_reason = _parse_score_and_reason(
                    llm.generate(citation_correctness_prompt),
                    "Citation correctness evaluated",
                )
                return {"score": cc_score, "reason": cc_reason}
            except Exception as exc:
                logger.warning("Citation correctness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 9. CITATION COMPLETENESS
        # What it measures: Whether all major factual claims in the answer
        #   are backed by at least one citation.
        # Why it matters: Uncited claims cannot be verified and are
        #   indistinguishable from hallucinations from the user's perspective.
        # Inputs: answer only
        # How evaluated: LLM counts major factual claims and checks what
        #   proportion carry a bracketed citation.
        # Score: 0 = no claims are cited; 1 = every claim has a citation.
        # ------------------------------------------------------------------
        def _eval_citation_completeness() -> dict[str, Any]:
            try:
                citation_completeness_prompt = f"""
Evaluate whether all major factual claims in the answer are supported by a bracketed citation.
Citation completeness measures whether claims without citations exist in the answer.

Answer:
{answer}

Rate the citation completeness on a scale of 0 to 1, where 1 means every factual claim has a citation.
If the answer makes no factual claims, score 1.0.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                ccmp_score, ccmp_reason = _parse_score_and_reason(
                    llm.generate(citation_completeness_prompt),
                    "Citation completeness evaluated",
                )
                return {"score": ccmp_score, "reason": ccmp_reason}
            except Exception as exc:
                logger.warning("Citation completeness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 10. GROUNDEDNESS
        # What it measures: Whether the answer contains any statements that
        #   are not present or inferable from the retrieved context
        #   (i.e., hallucinations).
        # Why it matters: Grounded answers are the core promise of RAG;
        #   ungrounded statements mean the model fabricated information.
        # Inputs: answer + retrieved_contexts (all)
        # How evaluated: LLM reads the answer and checks each statement
        #   against the retrieved passages, flagging any that cannot be
        #   found or reasonably inferred from the context.
        # Score: 0 = answer is full of hallucinations; 1 = fully grounded.
        # ------------------------------------------------------------------
        def _eval_groundedness() -> dict[str, Any]:
            try:
                groundedness_prompt = f"""
Evaluate whether every statement in the answer can be found in or inferred from the retrieved context.
Groundedness measures the absence of hallucinated content — statements not supported by the retrieved passages.

Retrieved Context:
{context_full}

Answer:
{answer}

Rate the groundedness on a scale of 0 to 1, where 1 means the answer contains no hallucinations and every statement is traceable to the context.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                gnd_score, gnd_reason = _parse_score_and_reason(
                    llm.generate(groundedness_prompt),
                    "Groundedness evaluated",
                )
                return {"score": gnd_score, "reason": gnd_reason}
            except Exception as exc:
                logger.warning("Groundedness evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 11. NOISE SENSITIVITY
        # What it measures: Whether irrelevant retrieved context negatively affects the answer.
        # ------------------------------------------------------------------
        def _eval_noise_sensitivity() -> dict[str, Any]:
            try:
                noise_sensitivity_prompt = f"""
Evaluate whether the presence of any irrelevant retrieved context negatively affected the answer.
Noise sensitivity measures if the model was confused by irrelevant chunks and hallucinated or degraded the answer quality.

Query:
{query}

Retrieved Passages:
{ctx_snippets}

Answer:
{answer}

Rate the noise sensitivity on a scale of 0 to 1, where 1 means the answer was perfectly robust and unaffected by any noise, and 0 means irrelevant context severely degraded the answer.
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                ns_score, ns_reason = _parse_score_and_reason(
                    llm.generate(noise_sensitivity_prompt),
                    "Noise sensitivity evaluated",
                )
                return {"score": ns_score, "reason": ns_reason}
            except Exception as exc:
                logger.warning("Noise sensitivity evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # 12. VISUAL/MULTIMODAL GROUNDING
        # What it measures: Whether answers correctly use information from image_description metadata.
        # ------------------------------------------------------------------
        def _eval_visual_multimodal_grounding() -> dict[str, Any]:
            try:
                multimodal_prompt = f"""
Evaluate whether the answer correctly uses information from image metadata (PDF images, charts, and figures) provided in the retrieved context.
Visual/Multimodal Grounding measures if the generated answer accurately reflects the visual descriptions.

Query:
{query}

Retrieved Passages (including image descriptions):
{context_full}

Answer:
{answer}

Rate the visual/multimodal grounding on a scale of 0 to 1, where 1 means the answer perfectly utilizes available image descriptions without misinterpretation.
If the answer makes no reference to visual elements or no image descriptions were retrieved, score 1.0 (neutral/not applicable).
Respond with just a number between 0 and 1, followed by a brief reason.
Format: <score> <reason>
"""
                vmg_score, vmg_reason = _parse_score_and_reason(
                    llm.generate(multimodal_prompt),
                    "Visual/multimodal grounding evaluated",
                )
                return {"score": vmg_score, "reason": vmg_reason}
            except Exception as exc:
                logger.warning("Visual/multimodal grounding evaluation failed: %s", exc)
                return {"score": 0.0, "reason": f"Evaluation error: {exc}"}

        # ------------------------------------------------------------------
        # Run all metrics concurrently
        # ------------------------------------------------------------------
        metric_fns = {
            "faithfulness": _eval_faithfulness,
            "answer_relevancy": _eval_answer_relevancy,
            "contextual_relevancy": _eval_contextual_relevancy,
            "contextual_precision": _eval_contextual_precision,
            "contextual_recall": _eval_contextual_recall,
            "answer_correctness": _eval_answer_correctness,
            "answer_completeness": _eval_answer_completeness,
            "citation_correctness": _eval_citation_correctness,
            "citation_completeness": _eval_citation_completeness,
            "groundedness": _eval_groundedness,
            "noise_sensitivity": _eval_noise_sensitivity,
            "visual_multimodal_grounding": _eval_visual_multimodal_grounding,
        }
        selected_metrics = set(metrics or metric_fns)
        metric_fns = {
            name: fn
            for name, fn in metric_fns.items()
            if name in selected_metrics
        } or metric_fns

        scores: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_metric = {
                executor.submit(fn): name
                for name, fn in metric_fns.items()
            }
            for future, metric_name in future_to_metric.items():
                try:
                    scores[metric_name] = future.result(timeout=30)
                except FuturesTimeoutError:
                    logger.warning("Metric %s timed out", metric_name)
                    scores[metric_name] = {
                        "score": 0.0,
                        "reason": "Evaluation timed out or failed",
                    }
                except Exception as exc:
                    logger.warning("Metric %s raised: %s", metric_name, exc)
                    scores[metric_name] = {
                        "score": 0.0,
                        "reason": "Evaluation timed out or failed",
                    }

        failed_metrics = [
            name
            for name, result in scores.items()
            if str(result.get("reason", "")).startswith("Evaluation failed:")
        ]
        return {
            "status": (
                "error"
                if failed_metrics and len(failed_metrics) == len(scores)
                else "partial" if failed_metrics else "ok"
            ),
            "scores": scores,
            "retrieval_mode": retrieval_mode,
            "failed_metrics": failed_metrics,
        }

    except Exception as exc:
        logger.exception("DeepEval evaluation failed for mode=%s", retrieval_mode)
        return {
            "status": "error",
            "error": str(exc),
            "scores": {},
            "retrieval_mode": retrieval_mode,
        }
