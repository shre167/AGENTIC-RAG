from __future__ import annotations

import re
from typing import Any, Dict

# ============================================================
# ALIAS MAP
#
# Keys are canonical names (lowercase).
# Values are aliases that should resolve to that canonical name.
#
# Used by canonicalize_query() to expand the search query with
# the canonical term so both the alias and the standard form
# are present in the retrieval query.
#
# NOTE: replicon / successfactors / workzone / neo are kept for
# backwards compatibility with the existing regression test suite.
# ============================================================

ALIAS_MAP: Dict[str, list[str]] = {
    # ── IT / HR legacy (keep for regression tests) ──────────────────
    "successfactors": [
        "sf",
        "success factors",
        "successfactors",
        "hr portal",
        "employee profile system",
        "sap successfactors",
    ],
    "replicon": [
        "replicon",
        "timesheet",
        "time tracking",
        "timesheet system",
    ],
    "workzone": [
        "workzone",
        "support portal",
        "ticket portal",
    ],
    "neo": [
        "neo",
        "neo assistant",
        "virtual assistant",
    ],
    # ── Astronomy / Space Science ────────────────────────────────────
    "jwst": [
        "james webb",
        "james webb space telescope",
        "webb telescope",
        "webb",
    ],
    "hst": [
        "hubble",
        "hubble space telescope",
    ],
    "isro": [
        "indian space research organisation",
        "indian space research organization",
        "isro",
    ],
    "nasa": [
        "national aeronautics and space administration",
        "nasa",
    ],
    "chandrayaan": [
        "chandrayaan",
        "chandrayan",
        "chandrayaan 1",
        "chandrayaan 2",
        "chandrayaan 3",
        "chandrayaan-1",
        "chandrayaan-2",
        "chandrayaan-3",
        "pragyan",
        "vikram lander",
    ],
    "black hole": [
        "blackhole",
        "black hole",
        "singularity",
        "event horizon",
        "schwarzschild",
    ],
    "exoplanet": [
        "exoplanet",
        "exoplanets",
        "extrasolar planet",
        "extrasolar planets",
    ],
    "cme": [
        "coronal mass ejection",
        "cme",
    ],
    "grb": [
        "gamma ray burst",
        "gamma-ray burst",
        "grb",
    ],
    "iss": [
        "international space station",
        "iss",
    ],
    "esa": [
        "european space agency",
        "esa",
    ],
    "jaxa": [
        "japan aerospace exploration agency",
        "jaxa",
    ],
}


def clean_text(text: str) -> str:
    """Lower-case, remove punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_query(query: str) -> Dict[str, Any]:
    """Normalise a user query and return an expanded form for retrieval.

    The returned ``expanded_query`` always preserves the full cleaned user
    question.  When a canonical alias is detected, the canonical term is
    *appended* (if not already present) so both the alias and the standard
    form appear in the retrieval query.

    Returns a dict with keys:
        original      – the raw input string
        cleaned       – lower-cased, punctuation-stripped version
        canonical     – matched canonical key, or None
        matched_alias – the specific alias phrase that matched, or None
        expanded_query – the query to send to vector + BM25 retrieval
    """
    cleaned = clean_text(query)

    for canonical_name, aliases in ALIAS_MAP.items():
        # Check canonical name itself first
        if canonical_name in cleaned:
            expanded = (
                cleaned
                if cleaned.endswith(canonical_name)
                else f"{cleaned} {canonical_name}"
            )
            return {
                "original": query,
                "cleaned": cleaned,
                "canonical": canonical_name,
                "matched_alias": canonical_name,
                "expanded_query": expanded,
            }

        for alias in aliases:
            if alias in cleaned:
                expanded = (
                    cleaned
                    if cleaned.endswith(canonical_name)
                    else f"{cleaned} {canonical_name}"
                )
                return {
                    "original": query,
                    "cleaned": cleaned,
                    "canonical": canonical_name,
                    "matched_alias": alias,
                    "expanded_query": expanded,
                }

    # No alias matched — return cleaned query unchanged
    return {
        "original": query,
        "cleaned": cleaned,
        "canonical": None,
        "matched_alias": None,
        "expanded_query": cleaned,
    }
