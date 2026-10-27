from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from src.config import (
    GE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from src.rag_pipeline.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ======================================================================
# LLM GENERATION
# ======================================================================


def ask_llm(
    question: str,
    context: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    trace: dict[str, Any] | None = None,
    retrieval_hints: dict[str, Any] | None = None,
) -> str:
    """
    Send the question and retrieved context to the LLM.

    The function keeps API credentials private.

    If a trace dictionary is provided, it records the LLM input and
    runtime information needed by the RAG Observatory, but never stores
    the API key.
    """
    resolved_api_key = (
        api_key
        or GE_API_KEY
        or LLM_API_KEY
    )

    resolved_model = (
        model_name
        or LLM_MODEL
    )

    resolved_base_url = (
        base_url
        or LLM_BASE_URL
    )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    if not resolved_api_key:
        logger.error(
            "LLM API key is not configured."
        )

        return (
            "I could not generate the answer "
            "because the language-model API "
            "configuration is incomplete."
        )

    if not resolved_base_url:
        logger.error(
            "LLM base URL is not configured."
        )

        return (
            "I could not generate the answer "
            "because the language-model endpoint "
            "is not configured."
        )

    if not resolved_model:
        logger.error(
            "LLM model is not configured."
        )

        return (
            "I could not generate the answer "
            "because the language-model model "
            "is not configured."
        )

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    question_lower = question.lower()

    is_explanation = any(
        word in question_lower
        for word in (
            "explain",
            "describe",
            "why",
            "difference",
            "compare",
            "how does",
        )
    )

    is_procedural = any(
        phrase in question_lower
        for phrase in (
            "how to",
            "how do i",
            "steps",
            "procedure",
            "configure",
            "setup",
            "set up",
            "install",
        )
    )

    # ------------------------------------------------------------------
    # Build LLM instruction
    # ------------------------------------------------------------------

    user_instruction = (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Information:\n"
        f"{context}\n\n"
        "Answer the question using ONLY "
        "the retrieved information above.\n\n"
        "Do not use outside knowledge.\n\n"
        "If the retrieved information is "
        "insufficient, explicitly say that "
        "the knowledge base does not contain "
        "enough information to answer.\n\n"
        "CITATION REQUIREMENT:\n"
        "Ground every factual statement with a bracketed citation marker "
        "pointing to the supporting source chunk (for example: [1] for Source 1, "
        "[2] for Source 2). When an answer contains multiple factual claims, "
        "support each claim individually where practical.\n\n"
        "Only use source filenames that appear "
        "in the retrieved information.\n\n"
        "Include the source filenames under "
        "Sources."
    )

    if is_explanation:
        user_instruction += (
            "\nFor explanations, organize "
            "the answer clearly and use "
            "bullet points where useful."
        )

    if is_procedural:
        user_instruction += (
            "\nFor procedures, use numbered "
            "steps and include prerequisites "
            "only when they are supported by "
            "the retrieved information."
        )

    hints = retrieval_hints or {}
    page_requested = hints.get("page_requested")
    page_found = hints.get("page_found", True)

    try:
        from src.rag_pipeline.query import extract_page_query_info

        page_info = extract_page_query_info(question)
        if page_info and page_info.get("is_page_query"):
            page_requested = True
            if "page_found" in hints:
                page_found = bool(hints.get("page_found"))
    except Exception:
        page_info = None

    if page_requested:
        if page_found:
            user_instruction += (
                "\n\nPAGE RETRIEVAL INSTRUCTION:\n"
                "The user is asking for content from a specific page or page range. "
                "Present the key content and summary found on that page accurately "
                "based on the retrieved sources. "
                "If the retrieved sources cover multiple books on that page number, "
                "organize the response clearly with separate sections or markdown "
                "headings for each book."
            )
        else:
            user_instruction += (
                "\n\nPAGE RETRIEVAL INSTRUCTION:\n"
                "The user asked for a specific page, but retrieval did NOT find "
                "indexed content for that page in the knowledge base. "
                "Do NOT claim that page content was found. "
                "State clearly that the requested page is not available in the "
                "indexed sources and do not substitute unrelated pages."
            )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_instruction,
        },
    ]

    # ------------------------------------------------------------------
    # Observatory trace
    # ------------------------------------------------------------------
    #
    # Store the exact prompt/context going into the LLM.
    # NEVER store the API key.
    #

    if trace is not None:
        trace.clear()

        trace.update(
            {
                "stage": "generation",
                "model": resolved_model,
                "base_url": resolved_base_url,
                "question": question,
                "system_prompt": SYSTEM_PROMPT,
                "user_instruction": user_instruction,
                "context": context,
                "context_chars": len(context),
                "message_count": len(messages),
                "temperature": 0.1,
                "explanation_request": is_explanation,
                "procedural_request": is_procedural,
                "status": "ready",
            }
        )

    # ------------------------------------------------------------------
    # LLM request
    # ------------------------------------------------------------------

    try:
        normalized_base_url = (
            resolved_base_url.strip()
        )

        if not normalized_base_url.endswith("/"):
            normalized_base_url += "/"

        import httpx
        http_client = httpx.Client(verify=False, timeout=httpx.Timeout(45.0, connect=10.0))

        client = OpenAI(
            api_key=resolved_api_key,
            base_url=normalized_base_url,
            http_client=http_client,
        )

        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=0.1,
        )

        # --------------------------------------------------------------
        # Validate response
        # --------------------------------------------------------------

        if not response.choices:
            logger.error(
                "Language model returned no choices."
            )

            if trace is not None:
                trace["status"] = "empty_choices"

            return (
                "The language model returned "
                "no response."
            )

        content = response.choices[0].message.content

        if not content:
            logger.error(
                "Language model returned empty content."
            )

            if trace is not None:
                trace["status"] = "empty_content"

            return (
                "The language model returned "
                "an empty response."
            )

        answer = content.strip()

        if trace is not None:
            usage_payload: dict[str, Any] = {}
            usage = getattr(response, "usage", None)
            if usage is not None:
                usage_payload = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(
                        usage, "completion_tokens", None
                    ),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            trace.update(
                {
                    "status": "success",
                    "response_chars": len(answer),
                    "response_preview": answer[:500],
                    "token_usage": usage_payload or None,
                }
            )

        return answer

    except Exception as exc:
        logger.exception(
            "LLM request failed: %s",
            exc,
        )

        if trace is not None:
            trace.update(
                {
                    "status": "error",
                    "error": str(exc),
                }
            )

        return (
            f"The language model request failed ({type(exc).__name__}: {exc}). "
            "Please check the model configuration, API key, and connectivity."
        )
