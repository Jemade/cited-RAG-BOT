import pytest
from app.services.retrieval.dense import RetrievalResult
from app.services.retrieval.hybrid import HybridRetriever

def test_reciprocal_rank_fusion():
    retriever = HybridRetriever(rrf_k=60)

    # Chunk A is rank 1 in dense, rank 2 in BM25
    dense_res = [
        RetrievalResult(chunk_id="chunkA", document_id="doc1", page_number=1, text="Text A", score=0.9, rank=1),
        RetrievalResult(chunk_id="chunkB", document_id="doc1", page_number=2, text="Text B", score=0.8, rank=2)
    ]

    bm25_res = [
        RetrievalResult(chunk_id="chunkC", document_id="doc1", page_number=3, text="Text C", score=10.5, rank=1),
        RetrievalResult(chunk_id="chunkA", document_id="doc1", page_number=1, text="Text A", score=8.0, rank=2)
    ]

    fused = retriever.reciprocal_rank_fusion(dense_res, bm25_res, top_k=3)

    # chunkA RRF score: 1/(60+1) + 1/(60+2) = 0.01639 + 0.016129 = 0.0325
    # chunkC RRF score: 1/(60+1) = 0.01639
    # chunkB RRF score: 1/(60+2) = 0.016129
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunkA"
    assert fused[0].rank == 1
    assert "rrf_score" in fused[0].metadata
    assert fused[0].metadata["dense_rank"] == 1
    assert fused[0].metadata["bm25_rank"] == 2

def test_weighted_score_fusion():
    retriever = HybridRetriever()

    dense_res = [
        RetrievalResult(chunk_id="chunkA", document_id="doc1", page_number=1, text="Text A", score=0.9, rank=1),
        RetrievalResult(chunk_id="chunkB", document_id="doc1", page_number=2, text="Text B", score=0.3, rank=2)
    ]

    bm25_res = [
        RetrievalResult(chunk_id="chunkA", document_id="doc1", page_number=1, text="Text A", score=10.0, rank=1),
        RetrievalResult(chunk_id="chunkB", document_id="doc1", page_number=2, text="Text B", score=2.0, rank=2)
    ]

    results = retriever.weighted_score_fusion(dense_res, bm25_res, alpha=0.5, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunkA"
    assert results[0].score > results[1].score
