import pytest
from app.services.retrieval.dense import RetrievalResult
from app.services.retrieval.reranker import CrossEncoderReranker

def test_reranker_ranking():
    reranker = CrossEncoderReranker()

    candidates = [
        RetrievalResult(
            chunk_id="c1", document_id="d1", page_number=1,
            text="The weather in Paris is sunny today.", score=0.8, rank=1
        ),
        RetrievalResult(
            chunk_id="c2", document_id="d1", page_number=7,
            text="We used the Adam optimizer with beta1 = 0.9 and beta2 = 0.98.", score=0.6, rank=2
        ),
        RetrievalResult(
            chunk_id="c3", document_id="d1", page_number=8,
            text="The recipe calls for two cups of flour and one cup of sugar.", score=0.4, rank=3
        )
    ]

    reranked = reranker.rerank(query="What optimizer and beta1 were used?", candidates=candidates, top_n=2)

    assert len(reranked) == 2
    # c2 must be ranked first because it directly mentions optimizer and beta1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert "rerank_score" in reranked[0].metadata
