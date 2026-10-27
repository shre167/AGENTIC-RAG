from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.rag_pipeline.models import Candidate, Citation, CitationCoverage

if TYPE_CHECKING:
    from src.embeddings import EmbeddingService

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "who", "what", "when", "where", "why", "how", "has", "have", "had", "been", "will", "would", "could", "should", "may", "might", "can"}


def _split_into_claims(answer: str) -> list[str]:
    """
    Extract claims from Answer AND Details sections.
    Stop before Sources section.
    
    Expected format:
        Answer:
        <claims to extract>
        
        Details:
        <also extract from here>
        
        Sources:
        <ignore this and below>
    """
    lines = []
    in_extractable_section = False
    
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Check if we're entering Answer or Details section
        if re.match(r"^(?:\*\*)?(?:answer|details)(?:\*\*)?:\s*$", line, re.I):
            in_extractable_section = True
            continue
        
        # Check if we're hitting Sources or other terminal section
        if re.match(r"^(?:\*\*)?(?:sources|confidence|references|note)(?:\*\*)?:\s*$", line, re.I):
            break  # Stop processing entirely
        
        # If we're in Answer or Details section, collect the line
        if in_extractable_section:
            # Remove bullet points or numbering
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line)
            lines.append(cleaned)
    
    # If no explicit "Answer:" or "Details:" section found, process all lines until terminal section
    if not lines:
        for line in answer.splitlines():
            line = line.strip()
            if not line:
                continue
            # Stop at terminal markers
            if re.match(r"^(?:\*\*)?(?:sources|confidence|references|note)(?:\*\*)?:\s*$", line, re.I):
                break
            # Remove bullet points or numbering
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line)
            lines.append(cleaned)
    
    # Split into sentences and filter by minimum length
    combined_text = " ".join(lines)
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined_text) if len(s.strip()) >= 15]
    
    return claims


def _find_best_passage_lexical(claim: str, text: str) -> tuple[str, float]:
    """Fast lexical matching: find passage with most term overlap."""
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    if not claim_terms:
        return text[:300].strip(), 0.0
    
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    best_passage = ""
    best_score = 0.0
    
    for idx, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower())) - _STOPWORDS
        overlap = len(claim_terms & terms) / max(1, len(claim_terms))
        
        if overlap > best_score:
            ctx_start = max(0, idx - 1)
            ctx_end = min(len(sentences), idx + 2)
            best_passage = " ".join(sentences[ctx_start:ctx_end])
            best_score = overlap
    
    return best_passage or text[:300].strip(), best_score


def _verify_claim_support(
    claim: str,
    passage: str,
    chunk_text: str,
    embedding_service: "EmbeddingService | None" = None
) -> tuple[str, float, str]:
    """
    Fast lexical verification - NO embedding calls during citation checking.
    This runs after retrieval and must be quick.
    """
    if not passage.strip() and not chunk_text.strip():
        return "unsupported", 0.0, "No source content available."
    
    # Fast lexical verification only
    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOPWORDS
    search_scope = f"{passage} {chunk_text}".lower()
    passage_terms = set(re.findall(r"[a-z0-9]+", search_scope)) - _STOPWORDS
    
    if not claim_terms:
        return "supported", 1.0, "Generic statement with no specific claims."
    
    lexical_coverage = len(claim_terms & passage_terms) / max(1, len(claim_terms))
    
    # Check for critical numeric mismatches
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    passage_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", search_scope))
    has_numeric_mismatch = bool(claim_numbers and (claim_numbers - passage_numbers))
    
    if has_numeric_mismatch:
        missing = ", ".join(sorted(claim_numbers - passage_numbers))
        if lexical_coverage >= 0.30:
            return "partially_supported", lexical_coverage, f"Topically relevant but numeric value '{missing}' not in source."
        else:
            return "unsupported", lexical_coverage, f"Key numeric value '{missing}' not found in source."
    
    # Lexical thresholds - optimized for speed
    if lexical_coverage >= 0.35:
        return "supported", lexical_coverage, "Strong lexical overlap with source."
    elif lexical_coverage >= 0.20:
        return "supported", lexical_coverage, "Good lexical match with source."
    elif lexical_coverage >= 0.12:
        return "partially_supported", lexical_coverage, "Partial lexical overlap with source."
    else:
        return "unsupported", lexical_coverage, "Minimal lexical alignment with source."


def _origin(candidate: Candidate) -> str:
    parts = []
    if candidate.dense_rank is not None:
        parts.append(f"Dense #{candidate.dense_rank}")
    if candidate.bm25_rank is not None:
        parts.append(f"BM25 #{candidate.bm25_rank}")
    if candidate.rrf_score:
        parts.append(f"RRF {candidate.rrf_score:.4f}")
    return ", ".join(parts) or "Retrieved context"


def generate_claim_citations(
    answer: str,
    selected_candidates: list[Candidate],
    kb_version: int | str = 1,
    embedding_service: "EmbeddingService | None" = None
) -> tuple[list[Citation], CitationCoverage]:
    """
    Generate fast, source-aware citations from Answer and Details sections.
    
    Extracts claims from both Answer AND Details sections, stops at Sources.
    Uses fast lexical matching - no embedding calls during citation verification.
    Embeddings are used during retrieval, not citation checking.
    """
    claims = _split_into_claims(answer) if answer.strip() else []
    if not selected_candidates:
        return [], CitationCoverage(len(claims), 0, len(claims), 0.0, bool(claims), claims)
    
    citations: list[Citation] = []
    unsupported: list[str] = []
    parser_failures: list[str] = []
    fully_supported = partial = 0
    chunk_map = {index + 1: candidate for index, candidate in enumerate(selected_candidates)}
    
    for claim_id, claim in enumerate(claims, start=1):
        display_claim = re.sub(r"\[(?:Source\s*|Chunk\s*)?[^\]]+\]", "", claim).strip()
        display_claim = re.sub(r"\((?:Source\s*|Chunk\s*)\d+\)", "", display_claim).strip()
        if not display_claim:
            display_claim = claim.strip()
        
        # Check if the sentence is an explicit abstention (e.g. "knowledge base does not contain...")
        is_abstention = any(
            phrase in display_claim.lower()
            for phrase in (
                "does not contain", "do not contain", "no information", "not found in",
                "insufficient information", "not mentioned in", "knowledge base does not",
            )
        )
        if is_abstention:
            # Abstention statements are not unsupported factual claims
            candidate = selected_candidates[0]
            citations.append(Citation(
                id=claim_id,
                marker=f"[{claim_id}]",
                claim=display_claim,
                filename=candidate.filename or "System",
                chunk_id="abstain",
                passage=candidate.text[:200] if candidate and candidate.text else "[Context Insufficient]",
                retrieval_source="Knowledge Boundary",
                status="supported",
                verification_reason="Standard abstention: Correctly acknowledges absent information.",
                kb_version=kb_version,
                support_score=1.0,
                lexical_overlap=1.0,
                direct_answer_support=True,
            ))
            fully_supported += 1
            continue

        # Extract explicit source citations: [1], [1, 2], [Source 1], [Chunk 2], (Source 1)
        valid_indices: list[int] = []
        out_of_range_indices: list[int] = []
        brackets = re.findall(r"\[([^\]]+)\]", claim) + re.findall(r"\((?:Source|Chunk)\s*([^)]+)\)", claim)
        has_bracket_marker = bool(brackets)

        for bracket_content in brackets:
            found_nums = [int(v) for v in re.findall(r"\b\d+\b", bracket_content)]
            for num in found_nums:
                if num in chunk_map:
                    if num not in valid_indices:
                        valid_indices.append(num)
                else:
                    if num not in out_of_range_indices:
                        out_of_range_indices.append(num)

        explicit_candidates = [chunk_map[idx] for idx in valid_indices]
        candidates = explicit_candidates if explicit_candidates else selected_candidates
        
        # Find best supporting passage using fast lexical matching
        scored = [
            (_find_best_passage_lexical(display_claim, candidate.text or ""), candidate)
            for candidate in candidates
        ]
        (passage, match_score), candidate = max(scored, key=lambda item: item[0][1])
        
        # Verify claim support using fast lexical analysis (NO embeddings)
        status, support_score, reason = _verify_claim_support(
            display_claim, passage, candidate.text or ""
        )
        
        # Distinguish parser failures from genuine unsupported answers
        if out_of_range_indices and not valid_indices:
            # Model attempted a citation, but referenced an index beyond retrieved context
            status = "parser_failure"
            reason = f"Citation Parser: Referenced source index {out_of_range_indices} exceeds available retrieved chunks (1..{len(chunk_map)})."
            parser_failures.append(display_claim)
        elif status == "supported":
            fully_supported += 1
        elif status == "partially_supported":
            partial += 1
        else:
            # Genuine unsupported answer claim
            unsupported.append(display_claim)
        
        # Extract rich metadata with proper page and file fallbacks
        meta = candidate.metadata if candidate else {}
        file_type = meta.get("file_type") or (
            Path(candidate.filename).suffix.lstrip(".").lower() 
            if candidate and candidate.filename else ""
        )
        
        page_start = meta.get("page_start") or meta.get("page") or meta.get("page_number")
        page_end = meta.get("page_end") or page_start
        try:
            page_number = int(page_start) if page_start and int(page_start) > 0 else None
        except (ValueError, TypeError):
            page_number = None

        page_range_str = None
        if page_start and page_end and str(page_start) != str(page_end):
            page_range_str = f"{page_start}–{page_end}"
        elif page_number is not None:
            page_range_str = str(page_number)

        source_name = (
            candidate.filename
            or meta.get("filename")
            or meta.get("book_title")
            or meta.get("book_name")
            or meta.get("source_name")
            or meta.get("source")
            or f"Document-{candidate.chunk_id}"
        )
        
        # Create citation
        citations.append(Citation(
            id=claim_id,
            marker=f"[{claim_id}]",
            claim=display_claim,
            filename=source_name,
            chunk_id=candidate.chunk_id if candidate and candidate.chunk_id is not None else "N/A",
            passage=passage if passage else "[No supporting passage found]",
            retrieval_source=_origin(candidate) if candidate else "Unverified",
            status=status,
            verification_reason=f"{status.replace('_', ' ').title()}: {reason}",
            kb_version=kb_version,
            support_score=support_score,
            lexical_overlap=match_score,
            direct_answer_support=status == "supported",
            file_type=file_type,
            page_number=page_number,
            page_range=page_range_str,
        ))
    
    # Calculate overall coverage
    total = len(claims)
    cov_pct = round(
        ((fully_supported * 1.0 + partial * 0.5) / total * 100.0) if total else 100.0,
        1
    )
    
    return citations, CitationCoverage(
        total_claims=total,
        supported_claims=fully_supported,
        unsupported_claims=len(unsupported),
        coverage_percentage=cov_pct,
        has_unsupported=bool(unsupported),
        unsupported_claims_list=unsupported,
        partially_supported_claims=partial,
        parser_failure_claims=len(parser_failures),
        parser_failures_list=parser_failures,
    )


