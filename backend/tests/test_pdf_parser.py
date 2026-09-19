import os
import pytest
from pathlib import Path
from app.services.ingestion.pdf_parser import PDFParser

def test_pdf_sha256(tmp_path):
    parser = PDFParser(preview_dir=tmp_path / "previews")
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("Hello CiteRAG")
    hash_val = parser.compute_sha256(str(dummy_file))
    assert len(hash_val) == 64
    assert isinstance(hash_val, str)

def test_parse_real_pdf():
    pdf_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_documents" / "attention_is_all_you_need.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found")

    parser = PDFParser()
    extracted_pages = parser.parse_pdf(file_path=str(pdf_path), document_id="test_doc_001")

    assert len(extracted_pages) == 15
    assert extracted_pages[0].page_number == 1
    assert extracted_pages[14].page_number == 15
    assert "Attention Is All You Need" in extracted_pages[0].text
    assert len(extracted_pages[0].blocks) > 0
