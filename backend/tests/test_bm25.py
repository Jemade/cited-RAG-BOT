import pytest
from app.services.retrieval.bm25 import BM25Retriever
from app.db.models import Chunk

def test_bm25_tokenization():
    text = "The Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 10-9 on WMT-2014."
    tokens = BM25Retriever.tokenize(text)
    assert "adam" in tokens
    assert "0.9" in tokens
    assert "0.98" in tokens
    assert "10-9" in tokens
    assert "wmt-2014" in tokens

def test_bm25_retrieval(db_session):
    # Populate chunks
    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            page_number=1,
            chunk_index=0,
            text="The Transformer relies entirely on an attention mechanism.",
            token_count=10,
            metadata_json={"page": 1}
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            page_number=7,
            chunk_index=1,
            text="We used the Adam optimizer with beta1 = 0.9 and beta2 = 0.98.",
            token_count=15,
            metadata_json={"page": 7}
        ),
        Chunk(
            id="c3",
            document_id="doc1",
            page_number=8,
            chunk_index=2,
            text="On the WMT 2014 English-to-German task, we achieve 28.4 BLEU.",
            token_count=12,
            metadata_json={"page": 8}
        )
    ]
    for c in chunks:
        db_session.add(c)
    db_session.commit()

    retriever = BM25Retriever()
    BM25Retriever.invalidate_cache("doc1")

    # Query for exact keywords
    results = retriever.search(db=db_session, query="Adam optimizer beta1 0.9", document_id="doc1", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"
    assert results[0].page_number == 7

    # Query for BLEU score
    results_bleu = retriever.search(db=db_session, query="28.4 BLEU WMT 2014", document_id="doc1", top_k=2)
    assert len(results_bleu) > 0
    assert results_bleu[0].chunk_id == "c3"
    assert results_bleu[0].page_number == 8
