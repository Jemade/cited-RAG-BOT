from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db
from app.db.models import Document
from app.schemas.query import QueryRequest, QueryResponse, CitationResponse, RetrievedChunkResponse
from app.services.pipeline import RAGPipeline

router = APIRouter(tags=["query"])

pipeline = RAGPipeline()

@router.post("/documents/{id}/query", response_model=QueryResponse)
async def query_document(
    id: str,
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query a specific indexed PDF document."""
    doc = db.query(Document).filter(Document.id == id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{id}' not found.")

    if doc.status != "indexed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is not ready for querying (Current status: {doc.status})."
        )

    threshold = request.confidence_threshold if request.confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD

    result = await pipeline.execute_query(
        db=db,
        query=request.query,
        document_id=id,
        retrieval_mode=request.retrieval_mode,
        top_k=request.top_k,
        top_n=request.top_n,
        confidence_threshold=threshold
    )

    return QueryResponse(
        request_id=result.request_id,
        query=result.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                citation_marker=c.citation_marker,
                snippet=c.snippet,
                validated=c.validated,
                match_score=c.match_score,
                warning=c.warning
            )
            for c in result.citations
        ],
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=rc.chunk_id,
                document_id=rc.document_id,
                page_number=rc.page_number,
                text=rc.text,
                score=rc.score,
                rank=rc.rank,
                section=rc.section,
                metadata=rc.metadata
            )
            for rc in result.retrieved_chunks
        ],
        retrieval_mode=result.retrieval_mode,
        confidence_score=result.confidence_score,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        latency_ms=result.latency_ms
    )

@router.post("/query", response_model=QueryResponse)
async def query_all_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Query across all indexed documents.
    """
    threshold = request.confidence_threshold if request.confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD

    result = await pipeline.execute_query(
        db=db,
        query=request.query,
        document_id=None,
        retrieval_mode=request.retrieval_mode,
        top_k=request.top_k,
        top_n=request.top_n,
        confidence_threshold=threshold
    )

    return QueryResponse(
        request_id=result.request_id,
        query=result.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                citation_marker=c.citation_marker,
                snippet=c.snippet,
                validated=c.validated,
                match_score=c.match_score,
                warning=c.warning
            )
            for c in result.citations
        ],
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=rc.chunk_id,
                document_id=rc.document_id,
                page_number=rc.page_number,
                text=rc.text,
                score=rc.score,
                rank=rc.rank,
                section=rc.section,
                metadata=rc.metadata
            )
            for rc in result.retrieved_chunks
        ],
        retrieval_mode=result.retrieval_mode,
        confidence_score=result.confidence_score,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        latency_ms=result.latency_ms
    )
