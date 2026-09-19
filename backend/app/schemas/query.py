from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to be answered by the RAG system")
    retrieval_mode: str = Field(
        default="hybrid_rerank",
        description="Retrieval strategy: 'vector', 'bm25', 'hybrid', or 'hybrid_rerank'"
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Number of candidates to retrieve initially")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of top candidates after reranking")
    confidence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold below which the system will refuse to answer (cross-encoder logit)"
    )

class CitationResponse(BaseModel):
    document_id: str
    page_number: int
    chunk_id: str
    citation_marker: str
    snippet: str
    validated: bool = True
    match_score: float = 1.0
    warning: Optional[str] = None

class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    score: float
    rank: int
    section: Optional[str] = None
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    citations: List[CitationResponse]
    retrieved_chunks: List[RetrievedChunkResponse]
    retrieval_mode: str
    confidence_score: Optional[float] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    latency_ms: float
