import asyncio
import pytest
from app.services.pipeline import RAGPipeline
from app.db.models import Chunk, Document

def test_low_confidence_refusal(db_session):
    async def _test():
        # Setup a document and unrelated chunk
        doc = Document(id="doc_refuse", filename="doc.pdf", file_hash="hash1", file_path="/fake/path", status="indexed")
        db_session.add(doc)

        chunk = Chunk(
            id="c1",
            document_id="doc_refuse",
            page_number=1,
            chunk_index=0,
            text="The recipe calls for two cups of organic sugar and vanilla.",
            token_count=12,
            embedding=[0.0] * 384,
            metadata_json={}
        )
        db_session.add(chunk)
        db_session.commit()

        pipeline = RAGPipeline()

        # Query completely unrelated to recipe
        out_of_scope_query = "What is the capital of Mongolia and its population?"

        # Set threshold high enough to force refusal
        result = await pipeline.execute_query(
            db=db_session,
            query=out_of_scope_query,
            document_id="doc_refuse",
            retrieval_mode="hybrid_rerank",
            confidence_threshold=0.0  # Unrelated query score will be negative
        )

        assert result.refused is True
        assert "don't know" in result.answer.lower()
        assert result.refusal_reason is not None
        assert len(result.citations) == 0

    asyncio.run(_test())
