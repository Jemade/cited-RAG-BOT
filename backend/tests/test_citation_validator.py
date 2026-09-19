import pytest
from app.services.generation.citation_validator import CitationValidator

def test_valid_citation_extraction():
    validator = CitationValidator()

    answer = "The Transformer employs 8 parallel attention heads [Page 5] and achieved 28.4 BLEU [Page 8]."
    retrieved_sources = [
        {
            "source_idx": 1,
            "page_number": 5,
            "chunk_id": "c_head",
            "document_id": "doc1",
            "text": "Multi-head attention uses h = 8 parallel attention layers or heads."
        },
        {
            "source_idx": 2,
            "page_number": 8,
            "chunk_id": "c_bleu",
            "document_id": "doc1",
            "text": "Establishing a new state-of-the-art BLEU score of 28.4 on English-to-German."
        }
    ]

    cleaned_answer, citations = validator.extract_and_validate(answer, retrieved_sources)

    assert len(citations) == 2
    assert citations[0].page_number == 5
    assert citations[0].chunk_id == "c_head"
    assert citations[0].validated is True
    assert "8" in citations[0].snippet

    assert citations[1].page_number == 8
    assert citations[1].chunk_id == "c_bleu"
    assert citations[1].validated is True
    assert "28.4" in citations[1].snippet

def test_hallucinated_citation_rejection():
    validator = CitationValidator()

    # Model cited Page 99 which is not in retrieved sources
    answer = "The model was trained on Mars [Page 99]."
    retrieved_sources = [
        {
            "source_idx": 1,
            "page_number": 5,
            "chunk_id": "c_head",
            "document_id": "doc1",
            "text": "Multi-head attention uses 8 heads."
        }
    ]

    cleaned_answer, citations = validator.extract_and_validate(answer, retrieved_sources)

    # Hallucinated citation [Page 99] should NOT produce a valid citation
    assert len(citations) == 0
    # The hallucinated marker should be removed from the text
    assert "[Page 99]" not in cleaned_answer
