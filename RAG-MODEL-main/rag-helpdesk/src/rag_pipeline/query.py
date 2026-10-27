from __future__ import annotations

import re
from typing import Any, Dict, Optional


def normalize_text(text: Any) -> str:
    """
    Normalize text for comparison, matching, and diagnostics.

    Keeps the actual semantic content while removing excessive
    whitespace and normalizing case.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        if value is None:
            return default

        result = float(value)

        if result != result:  # NaN
            return default

        return result

    except (TypeError, ValueError):
        return default


def candidate_key(
    candidate_or_filename: Any,
    chunk_id: Optional[Any] = None,
) -> str:
    """
    Build a stable identifier for a retrieved chunk.

    Supports both calling styles used by the pipeline:

        candidate_key(candidate_dict)

    and:

        candidate_key(filename, chunk_id)

    This compatibility is important because retrieval.py uses:

        candidate_key(filename, chunk_id)

    while other parts of the pipeline may pass a candidate dictionary.
    """

    # Style 1:
    # candidate_key(filename, chunk_id)
    if chunk_id is not None:
        filename = str(candidate_or_filename or "")
        return f"{filename}::chunk-{chunk_id}"

    # Style 2:
    # candidate_key(candidate_dict)
    if isinstance(candidate_or_filename, dict):
        filename = str(
            candidate_or_filename.get("filename", "")
        )

        cid = candidate_or_filename.get(
            "chunk_id",
            "",
        )

        return f"{filename}::chunk-{cid}"

    # Fallback for a plain string/object.
    return str(candidate_or_filename or "")


def get_retrieval_mode(
    mode: Optional[str] = None,
) -> str:
    """
    Normalize the retrieval mode.

    Supported modes:
        vector / dense
        bm25
        hybrid / both / rrf

    Hybrid is the default.
    """

    if mode is None:
        return "hybrid"

    normalized = normalize_text(mode)

    if normalized in {
        "vector",
        "dense",
        "vector only",
        "dense only",
        "vector-only",
        "dense-only",
    }:
        return "vector"

    if normalized in {
        "bm25",
        "keyword",
        "sparse",
        "bm25 only",
        "keyword only",
        "sparse only",
        "bm25-only",
    }:
        return "bm25"

    if normalized in {
        "hybrid",
        "both",
        "dense + bm25",
        "bm25 + dense",
        "dense and bm25",
        "bm25 and dense",
        "rrf",
    }:
        return "hybrid"

    # Unknown mode → safe default.
    return "hybrid"


def is_greeting(query: str) -> bool:
    """
    Detect simple conversational greetings so they don't trigger
    unnecessary knowledge-base retrieval.
    """

    text = normalize_text(query)

    if not text:
        return False

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "bye",
        "goodbye",
    }

    if text in greetings:
        return True

    # Small variations such as "hi there".
    greeting_patterns = [
        r"^hi there$",
        r"^hello there$",
        r"^hey there$",
        r"^hey everyone$",
        r"^hello everyone$",
        r"^hi everyone$",
    ]

    return any(
        re.fullmatch(pattern, text)
        for pattern in greeting_patterns
    )


def is_multi_intent_query(query: str) -> bool:
    """
    Detect whether a query appears to contain multiple distinct
    information requests.

    This is intentionally conservative. It is used for diagnostics
    and query preparation, not as a hard retrieval filter.
    """

    text = normalize_text(query)

    if not text:
        return False

    # Explicit conjunctions often indicate multiple topics.
    separators = [
        " and ",
        " also ",
        " as well as ",
        " along with ",
        " plus ",
    ]

    separator_hits = sum(
        text.count(separator)
        for separator in separators
    )

    # Multiple question marks are a strong signal.
    question_count = text.count("?")

    # Comma-separated long queries can sometimes contain multiple
    # intents, but we avoid treating every comma as multi-intent.
    comma_count = text.count(",")

    if question_count >= 2:
        return True

    if separator_hits >= 1 and len(text.split()) >= 6:
        return True

    if comma_count >= 2 and len(text.split()) >= 10:
        return True

    return False


# ---------------------------------------------------------------------------
# Book alias registry — static seed kept for backwards compatibility.
# get_book_aliases() merges these with every doc currently in the vector
# store, so newly ingested books are recognised in page/chapter queries
# automatically without any code changes.
# ---------------------------------------------------------------------------

_STATIC_BOOK_ALIASES: dict[str, list[str]] = {
    "Deep Work": [
        "deep work",
        "deepwork",
        "cal newport",
        "newport",
    ],
    "cant hurt me": [
        "cant hurt me",
        "can't hurt me",
        "david goggins",
        "goggins",
    ],
    "Mind Management, Not Time Management (1)": [
        "mind management",
        "not time management",
        "mind management not time management",
        "david kadavy",
        "kadavy",
    ],
}

# Keep the old name so any direct imports still work.
KNOWN_BOOK_ALIASES = _STATIC_BOOK_ALIASES


def _aliases_for_filename(filename: str) -> list[str]:
    """Auto-generate query aliases from a document filename.

    Examples
    --------
    "Atomic Habits (James Clear).pdf"
        -> ["atomic habits (james clear)", "atomic habits", "james clear",
            "clear", "atomic", "habits"]
    "Deep_Work_Newport_2016.pdf"
        -> ["deep_work_newport_2016", "deep work newport 2016",
            "deep", "work", "newport", ...]
    """
    from pathlib import Path as _Path

    stem = _Path(filename).stem
    base = stem.lower()
    aliases: list[str] = [base]

    # version without underscores/hyphens
    spaced = re.sub(r"[_\-]+", " ", base).strip()
    if spaced != base:
        aliases.append(spaced)

    # strip parenthesised author/edition segments and keep them separately
    inner_parts = re.findall(r"\(([^)]+)\)", base)
    clean = re.sub(r"\s*\([^)]*\)", "", spaced).strip()
    if clean and clean not in aliases:
        aliases.append(clean)
    for part in inner_parts:
        part = part.strip()
        if part and part not in aliases:
            aliases.append(part)

    # drop common edition/volume noise
    clean2 = re.sub(
        r"\b(vol|volume|edition|ed|book|part|no|num|number)\b\.?\s*\d*",
        "",
        clean,
    ).strip()
    clean2 = re.sub(r"\s+", " ", clean2).strip()
    if clean2 and clean2 not in aliases:
        aliases.append(clean2)

    # individual significant words (>=4 chars, skip stop words)
    _stop = {
        "with", "from", "that", "this", "they", "your", "have", "been",
        "what", "when", "will", "more", "time", "into", "over", "also",
    }
    for word in re.findall(r"[a-z]{4,}", spaced):
        if word not in _stop and word not in aliases:
            aliases.append(word)

    return [a for a in aliases if a]


def get_book_aliases(vector_store: Any | None = None) -> dict[str, list[str]]:
    """Return the full alias map: static entries + every ingested document.

    Parameters
    ----------
    vector_store:
        Optional UnifiedVectorStore / ChromaVectorStore instance.  When
        supplied the collection is scanned for distinct filenames and
        auto-aliases are generated for each, so page/chapter queries work
        for any book without hardcoding it here.
    """
    aliases: dict[str, list[str]] = dict(_STATIC_BOOK_ALIASES)

    if vector_store is None:
        return aliases

    try:
        backend = getattr(vector_store, "_backend", vector_store)
        col = getattr(backend, "collection", None)
        if col is None:
            return aliases

        data = col.get(include=["metadatas"])
        seen: set[str] = set()
        for meta in (data.get("metadatas") or []):
            fn = (
                meta.get("filename")
                or meta.get("book_title")
                or meta.get("book_name")
                or ""
            )
            if fn and fn not in seen:
                seen.add(fn)
                if fn not in aliases:
                    aliases[fn] = _aliases_for_filename(fn)
    except Exception:
        pass

    return aliases


def extract_page_query_info(query: str, vector_store: Any | None = None) -> Optional[Dict[str, Any]]:
    """
    Detect if the user is asking for content from a specific page or page range.
    Extracts the page number, optional page end, target book if mentioned, and
    determines if the query is a pure page lookup or contains additional semantic keywords.

    Parameters
    ----------
    query:        Raw user query string.
    vector_store: Optional vector store used to resolve book titles dynamically.
                  When supplied, any ingested document is recognised by name —
                  not just the hardcoded static entries.
    """
    if not query:
        return None

    text = str(query).strip()
    text_lower = text.lower()

    # 1. Page range: e.g. "page 60-65", "pages 60 to 65", "p. 60-65"
    range_match = re.search(
        r"\b(?:pages?|p\.?|pg\.?)\s*(\d{1,4})\s*(?:-|–|—|to)\s*(\d{1,4})\b",
        text_lower,
    )
    if range_match:
        page_start = int(range_match.group(1))
        page_end = int(range_match.group(2))
        matched_span = range_match.span()
    else:
        # 2. Single page: e.g. "page 60", "p. 60", "on page 60", "in page 60", "page no 60", "page number 60"
        single_match = re.search(
            r"\b(?:(?:on|in|at)\s+(?:the\s+)?)?(?:pages?|p\.?|pg\.?|page\s+no\.?|page\s+number)\s*[:#]?\s*(\d{1,4})\b",
            text_lower,
        )
        if single_match:
            page_start = int(single_match.group(1))
            page_end = page_start
            matched_span = single_match.span()
        else:
            return None

    # Check for book mentions — use dynamic aliases so any ingested doc is recognised
    _book_aliases = get_book_aliases(vector_store)
    book_match = None
    for canonical_name, aliases in _book_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                book_match = canonical_name
                break
        if book_match:
            break

    # Determine whether there is additional semantic intent beyond the page request
    residual = text_lower[:matched_span[0]] + " " + text_lower[matched_span[1]:]
    if book_match:
        for alias in _book_aliases[book_match]:
            residual = re.sub(rf"\b{re.escape(alias)}\b", " ", residual)

    filler_words = {
        "give", "me", "the", "content", "in", "on", "at", "of", "from", "for",
        "what", "is", "written", "there", "show", "tell", "about", "summarize",
        "summary", "read", "details", "explain", "text", "information", "info",
        "please", "can", "you", "does", "say", "saying", "having", "with",
        "book", "ebook", "pdf", "chapter", "bro", "hey", "a", "an",
    }
    residual_tokens = [
        w for w in re.findall(r"[a-z0-9]+", residual)
        if w not in filler_words and len(w) > 2
    ]

    is_pure_page_query = len(residual_tokens) == 0
    semantic_query = " ".join(residual_tokens) if not is_pure_page_query else ""

    return {
        "is_page_query": True,
        "page_start": page_start,
        "page_end": page_end,
        "book_match": book_match,
        "is_pure_page_query": is_pure_page_query,
        "semantic_query": semantic_query,
    }


_CHAPTER_ORDINALS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def extract_chapter_query_info(query: str, vector_store: Any | None = None) -> Optional[Dict[str, Any]]:
    """
    Detect if the user is asking about a specific chapter number.

    Handles patterns like:
      - "chapter 3", "chapter three", "ch 3", "ch. 3"
      - "in chapter 5", "show me chapter 2", "what is in chapter 4"
      - "summarize chapter 1"

    Parameters
    ----------
    query:        Raw user query string.
    vector_store: Optional vector store for dynamic book title resolution.

    Returns a dict with ``is_chapter_query``, ``chapter_num``, and
    ``book_match``, or *None* if no chapter reference is found.
    """
    if not query:
        return None

    text_lower = str(query).strip().lower()

    # Numeric chapter: "chapter 3", "ch. 3", "ch 3"
    num_match = re.search(
        r"\b(?:chapter|ch\.?)\s*(\d{1,3})\b",
        text_lower,
    )
    if num_match:
        chapter_num = int(num_match.group(1))
        matched_span = num_match.span()
    else:
        # Ordinal word: "chapter three", "chapter one"
        ordinal_pattern = "|".join(re.escape(w) for w in _CHAPTER_ORDINALS)
        word_match = re.search(
            rf"\b(?:chapter|ch\.?)\s+({ordinal_pattern})\b",
            text_lower,
        )
        if word_match:
            chapter_num = _CHAPTER_ORDINALS[word_match.group(1)]
            matched_span = word_match.span()
        else:
            return None

    if chapter_num < 1:
        return None

    # Check for known book mentions using dynamic aliases
    book_match = None
    _book_aliases = get_book_aliases(vector_store)
    for canonical_name, aliases in _book_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                book_match = canonical_name
                break
        if book_match:
            break

    return {
        "is_chapter_query": True,
        "chapter_num": chapter_num,
        "book_match": book_match,
    }


def prepare_search_query(
    query: str,
) -> Dict[str, Any]:
    """
    Prepare a query for retrieval while preserving the original
    user wording.

    The original query is always retained.

    Returns:
        {
            "original_query": ...,
            "search_query": ...,
            "normalized_query": ...,
            "is_greeting": ...,
            "is_multi_intent": ...,
            "page_info": ...
        }
    """

    original_query = str(query or "").strip()
    normalized_query = normalize_text(original_query)

    greeting = is_greeting(original_query)
    multi_intent = is_multi_intent_query(original_query)
    page_info = extract_page_query_info(original_query)

    # IMPORTANT:
    # Do not aggressively rewrite the user's query.
    # Dense and BM25 retrieval should see the actual semantic query.
    search_query = original_query

    return {
        "original_query": original_query,
        "search_query": search_query,
        "normalized_query": normalized_query,
        "is_greeting": greeting,
        "is_multi_intent": multi_intent,
        "page_info": page_info,
    }


def tokenize_query(query: str) -> list[str]:
    """
    Tokenize a query using the same simple lexical style used by
    the BM25 retriever.
    """

    text = normalize_text(query)

    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+",
        text,
    )


def get_query_diagnostics(
    query: str,
    retrieval_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return runtime diagnostics describing how the query will be
    processed.

    This function does not perform retrieval.
    """

    prepared = prepare_search_query(query)

    tokens = tokenize_query(
        prepared["search_query"]
    )

    mode = get_retrieval_mode(
        retrieval_mode
    )

    return {
        "original_query": prepared["original_query"],
        "normalized_query": prepared["normalized_query"],
        "search_query": prepared["search_query"],
        "retrieval_mode": mode,
        "token_count": len(tokens),
        "tokens": tokens,
        "is_greeting": prepared["is_greeting"],
        "is_multi_intent": prepared["is_multi_intent"],
        "page_info": prepared.get("page_info"),
    }

