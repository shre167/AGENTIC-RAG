from src.rag_pipeline.citations import generate_claim_citations
from src.rag_pipeline.models import Candidate


def _candidate(text: str) -> Candidate:
    return Candidate(key="mission::0", filename="mission.txt", chunk_id=0, text=text)


def test_related_lexical_match_is_not_direct_answer_evidence():
    """Mission-name overlap must not validate a claim with the wrong destination."""
    citations, coverage = generate_claim_citations(
        "Chandrayaan landed on Mars.",
        [_candidate("Chandrayaan landed near the Moon's south pole.")],
    )

    assert citations[0].status == "unsupported"
    assert citations[0].lexical_overlap > 0.5
    assert "Mars" in citations[0].verification_reason
    assert coverage.coverage_percentage == 0.0
    assert coverage.has_unsupported is True


def test_direct_answer_evidence_is_cited_at_claim_level():
    citations, coverage = generate_claim_citations(
        "Chandrayaan landed near the Moon's south pole.",
        [_candidate("Chandrayaan landed near the Moon's south pole.")],
    )

    assert citations[0].status == "supported"
    assert citations[0].direct_answer_support is True
    assert coverage.coverage_percentage == 100.0
