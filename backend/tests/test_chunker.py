import pytest
from app.services.ingestion.pdf_parser import ExtractedPage
from app.services.ingestion.chunker import ParagraphAwareChunker

def test_heading_detection():
    chunker = ParagraphAwareChunker()
    assert chunker.is_likely_heading("1. Introduction") is True
    assert chunker.is_likely_heading("3.2 Multi-Head Attention") is True
    assert chunker.is_likely_heading("IV. EXPERIMENTS") is True
    assert chunker.is_likely_heading("This is a standard paragraph discussing models and datasets.") is False

def test_sentence_splitting():
    chunker = ParagraphAwareChunker()
    text = "The Transformer is fast. It uses attention! Does it work well? Yes."
    sentences = chunker.split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "The Transformer is fast."
    assert sentences[1] == "It uses attention!"

def test_chunk_page_preservation():
    chunker = ParagraphAwareChunker(chunk_size=300, chunk_overlap=50)
    pages = [
        ExtractedPage(page_number=1, text="Page one text content. " * 15, blocks=[]),
        ExtractedPage(page_number=2, text="Page two text content. " * 10, blocks=[])
    ]

    chunks = chunker.chunk_document(pages, document_id="doc_123", source_filename="test.pdf")

    assert len(chunks) > 0
    # Verify page numbers match the source page
    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(p1_chunks) > 0
    assert len(p2_chunks) > 0
    for c in p1_chunks:
        assert c.page_number == 1
        assert c.document_id == "doc_123"
        assert c.metadata_json["page_number"] == 1

def test_empty_page_handling():
    chunker = ParagraphAwareChunker()
    pages = [
        ExtractedPage(page_number=1, text="Valid content here that exceeds the minimum chunk length.", blocks=[]),
        ExtractedPage(page_number=2, text="", blocks=[]), # Empty page
        ExtractedPage(page_number=3, text="    \n   ", blocks=[]) # Whitespace only
    ]
    chunks = chunker.chunk_document(pages, document_id="doc_test", source_filename="empty.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
