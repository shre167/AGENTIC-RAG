from __future__ import annotations

from src.config import (
    BM25_TOP_K,
    HYBRID_TOP_K,
    MIN_RERANK_SCORE,
    RERANK_TOP_K,
    VECTOR_TOP_K,
)

# ======================================================================
# GENERAL SETTINGS
# ======================================================================

ABSTAIN_MESSAGE = (
    "I couldn't find enough relevant information in the knowledge base "
    "to answer that reliably."
)


# ======================================================================
# RETRIEVAL SETTINGS
# ======================================================================

# Reciprocal Rank Fusion constant.
# RRF is based on rank rather than raw score, so dense and BM25 scores
# do not need to be on the same numerical scale.
RRF_K = 60

# Retrieve substantially more candidates than are finally sent to the LLM.
# This prevents the first retriever ranking from becoming the final answer
# ranking, especially as the knowledge base grows.
MIN_RERANK_CANDIDATES = 20
MAX_RERANK_CANDIDATES = 30

RERANK_CANDIDATE_LIMIT = min(
    max(
        HYBRID_TOP_K,
        VECTOR_TOP_K,
        BM25_TOP_K,
        MIN_RERANK_CANDIDATES,
    ),
    MAX_RERANK_CANDIDATES,
)

# Number of chunks that can ultimately reach the LLM.
# Keeping this bounded prevents context bloat as the KB grows.
MAX_CONTEXT_CHUNKS = min(
    max(RERANK_TOP_K, 8),
    10,
)

# Prevent one source file from dominating the final context.
MAX_CHUNKS_PER_SOURCE = 3

# Maximum characters sent to the LLM as retrieved context.
MAX_CONTEXT_CHARS = 18000


# ======================================================================
# RERANKING / EVIDENCE SETTINGS
# ======================================================================

# MIN_RERANK_SCORE sanity floor.
RERANK_ABSTAIN_FLOOR = MIN_RERANK_SCORE

# CrossEncoder reranking is intentionally disabled.
# Retrieval uses Dense + BM25 + RRF only.
ENABLE_RERANKER = False

# RRF is a ranking signal, not a probability.
# With RRF_K=60, a rank-0 result in one retriever is ~0.01639;
# a rank-0 result in both retrievers is ~0.03279.
RRF_ABSTAIN_FLOOR = 0.008
STRONG_RRF_THRESHOLD = 0.022
MODERATE_RRF_THRESHOLD = RRF_ABSTAIN_FLOOR

# Relative separation from the runner-up.
MIN_RERANK_GAP = 0.25
STRONG_RERANK_GAP = 0.75

# Support window threshold.
SUPPORT_SCORE_WINDOW = 1.0

# Percentile score distribution signals.
MIN_TOP_PERCENTILE = 0.70
STRONG_TOP_PERCENTILE = 0.85
MIN_PERCENTILE_POOL = 5


# ======================================================================
# GENERIC RAG SYSTEM PROMPT
# ======================================================================

SYSTEM_PROMPT = """
You are a Knowledge Base Assistant.

Your job is to answer the user's question using ONLY the
retrieved information supplied in the conversation.

Rules:

1. Answer directly and clearly.

2. Use only information supported by the retrieved context.

3. Do not use outside knowledge to fill missing information.

4. Do not invent facts, numbers, names, dates, procedures,
   explanations, or examples that are not supported by the context.

5. You may combine multiple retrieved chunks only when they
   directly support the same answer.

6. If the retrieved information is insufficient, say so clearly.

7. If the user asks for a complete list and the retrieved context
   contains only part of the list, explicitly say that the available
   information is partial. Do not invent the missing items.

8. Prefer concise answers unless the user asks for a detailed explanation.

9. For processes or procedures, use numbered steps.

10. Preserve important technical terminology from the retrieved context.

11. Do not mention retrieval scores, embeddings, BM25, reranking,
    vector databases, or internal pipeline details unless the user
    explicitly asks about them.

12. Sources must contain only filenames that are explicitly present
    in the retrieved information.

13. Do not create or guess source filenames.

Response format:

Answer:
<direct answer>

Details:
<supporting explanation if useful>

Sources:
<source filenames>
"""
