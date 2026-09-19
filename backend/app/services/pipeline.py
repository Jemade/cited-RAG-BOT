import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.models import QueryRecord, CitationRecord
from app.services.retrieval.dense import DenseRetriever, RetrievalResult
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.generation.context_builder import ContextBuilder
from app.services.generation.llm_client import LLMClient
from app.services.generation.citation_validator import CitationValidator, ValidatedCitation

class RAGPipelineResult:
    def __init__(
        self,
        request_id: str,
        query: str,
        answer: str,
        citations: List[ValidatedCitation],
        retrieved_chunks: List[RetrievalResult],
        retrieval_mode: str,
        confidence_score: Optional[float],
        refused: bool,
        refusal_reason: Optional[str],
        latency_ms: float
    ):
        self.request_id = request_id
        self.query = query
        self.answer = answer
        self.citations = citations
        self.retrieved_chunks = retrieved_chunks
        self.retrieval_mode = retrieval_mode
        self.confidence_score = confidence_score
        self.refused = refused
        self.refusal_reason = refusal_reason
        self.latency_ms = round(latency_ms, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "retrieval_mode": self.retrieval_mode,
            "confidence_score": round(self.confidence_score, 4) if self.confidence_score is not None else None,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "latency_ms": self.latency_ms
        }

class RAGPipeline:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder()
        self.llm_client = LLMClient()
        self.citation_validator = CitationValidator()

    async def execute_query(
        self,
        db: Session,
        query: str,
        document_id: Optional[str] = None,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = settings.DENSE_TOP_K,
        top_n: int = settings.RERANK_TOP_N,
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD
    ) -> RAGPipelineResult:
        """Execute retrieval, reranking, confidence check, and answer generation."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        retrieval_mode = retrieval_mode.lower()

        logger.info(f"[{request_id}] Executing query: '{query}' (Mode: {retrieval_mode}, Doc: {document_id})")

        # 1. Retrieval Phase
        retrieved_candidates: List[RetrievalResult] = []

        if retrieval_mode == "vector":
            retrieved_candidates = self.dense_retriever.search_pgvector(
                db=db, query=query, document_id=document_id, top_k=top_n
            )
        elif retrieval_mode == "bm25":
            retrieved_candidates = self.bm25_retriever.search(
                db=db, query=query, document_id=document_id, top_k=top_n
            )
        elif retrieval_mode == "hybrid":
            dense_res = self.dense_retriever.search_pgvector(
                db=db, query=query, document_id=document_id, top_k=top_k
            )
            bm25_res = self.bm25_retriever.search(
                db=db, query=query, document_id=document_id, top_k=top_k
            )
            retrieved_candidates = self.hybrid_retriever.reciprocal_rank_fusion(
                dense_results=dense_res, bm25_results=bm25_res, top_k=top_n
            )
        else:  # hybrid_rerank (Default)
            dense_res = self.dense_retriever.search_pgvector(
                db=db, query=query, document_id=document_id, top_k=top_k
            )
            bm25_res = self.bm25_retriever.search(
                db=db, query=query, document_id=document_id, top_k=top_k
            )
            fused = self.hybrid_retriever.reciprocal_rank_fusion(
                dense_results=dense_res, bm25_results=bm25_res, top_k=top_k
            )
            retrieved_candidates = self.reranker.rerank(
                query=query, candidates=fused, top_n=top_n
            )

        # 2. Check for empty retrieval
        if not retrieved_candidates:
            latency_ms = (time.time() - start_time) * 1000.0
            return RAGPipelineResult(
                request_id=request_id,
                query=query,
                answer="I don't know based on the documents available. No relevant content was found in the indexed documents.",
                citations=[],
                retrieved_chunks=[],
                retrieval_mode=retrieval_mode,
                confidence_score=None,
                refused=True,
                refusal_reason="No matching chunks found in the database.",
                latency_ms=latency_ms
            )

        # 3. Confidence Threshold & Refusal Evaluation
        top_score = retrieved_candidates[0].score
        refused = False
        refusal_reason = None

        if retrieval_mode == "hybrid_rerank" and top_score < confidence_threshold:
            refused = True
            refusal_reason = (
                f"Top cross-encoder relevance score ({round(top_score, 4)}) is below "
                f"confidence threshold ({confidence_threshold})."
            )
            logger.info(f"[{request_id}] Query refused due to low confidence: {refusal_reason}")
            latency_ms = (time.time() - start_time) * 1000.0
            answer = "I don't know based on the documents available. No sufficiently relevant source was found to answer this question."

            # Persist refusal query record
            self._persist_query(
                db=db,
                query_id=request_id,
                document_id=document_id,
                query_text=query,
                answer=answer,
                retrieval_mode=retrieval_mode,
                confidence_score=top_score,
                refused=True,
                latency_ms=latency_ms,
                citations=[]
            )

            return RAGPipelineResult(
                request_id=request_id,
                query=query,
                answer=answer,
                citations=[],
                retrieved_chunks=retrieved_candidates,
                retrieval_mode=retrieval_mode,
                confidence_score=top_score,
                refused=True,
                refusal_reason=refusal_reason,
                latency_ms=latency_ms
            )

        # 4. Context Assembly
        context_str, source_meta = self.context_builder.build_context(retrieved_candidates)

        # 5. LLM Answer Generation
        raw_answer = await self.llm_client.generate_answer(
            query=query,
            context_str=context_str,
            source_meta=source_meta
        )

        # 6. Check if LLM explicitly stated refusal
        if "don't know" in raw_answer.lower() or "not contain sufficient" in raw_answer.lower():
            refused = True
            refusal_reason = "Model determined retrieved sources contain insufficient evidence."
            latency_ms = (time.time() - start_time) * 1000.0

            self._persist_query(
                db=db,
                query_id=request_id,
                document_id=document_id,
                query_text=query,
                answer=raw_answer,
                retrieval_mode=retrieval_mode,
                confidence_score=top_score,
                refused=True,
                latency_ms=latency_ms,
                citations=[]
            )

            return RAGPipelineResult(
                request_id=request_id,
                query=query,
                answer=raw_answer,
                citations=[],
                retrieved_chunks=retrieved_candidates,
                retrieval_mode=retrieval_mode,
                confidence_score=top_score,
                refused=True,
                refusal_reason=refusal_reason,
                latency_ms=latency_ms
            )

        # 7. Citation Validation & Source Mapping
        cleaned_answer, validated_citations = self.citation_validator.extract_and_validate(
            answer_text=raw_answer,
            retrieved_sources=source_meta
        )

        latency_ms = (time.time() - start_time) * 1000.0

        # 8. Persist query & citations
        self._persist_query(
            db=db,
            query_id=request_id,
            document_id=document_id,
            query_text=query,
            answer=cleaned_answer,
            retrieval_mode=retrieval_mode,
            confidence_score=top_score,
            refused=False,
            latency_ms=latency_ms,
            citations=validated_citations
        )

        return RAGPipelineResult(
            request_id=request_id,
            query=query,
            answer=cleaned_answer,
            citations=validated_citations,
            retrieved_chunks=retrieved_candidates,
            retrieval_mode=retrieval_mode,
            confidence_score=top_score,
            refused=False,
            refusal_reason=None,
            latency_ms=latency_ms
        )

    def _persist_query(
        self,
        db: Session,
        query_id: str,
        document_id: Optional[str],
        query_text: str,
        answer: str,
        retrieval_mode: str,
        confidence_score: Optional[float],
        refused: bool,
        latency_ms: float,
        citations: List[ValidatedCitation]
    ):
        """Persist query record and citations in database."""
        try:
            q_rec = QueryRecord(
                id=query_id,
                document_id=document_id,
                query_text=query_text,
                answer=answer,
                retrieval_mode=retrieval_mode,
                confidence_score=confidence_score,
                refused=refused,
                latency_ms=latency_ms
            )
            db.add(q_rec)

            for c in citations:
                cit_rec = CitationRecord(
                    query_id=query_id,
                    chunk_id=c.chunk_id,
                    page_number=c.page_number,
                    citation_marker=c.citation_marker,
                    snippet=c.snippet,
                    validated=c.validated,
                    match_score=c.match_score
                )
                db.add(cit_rec)

            db.commit()
        except Exception as e:
            logger.error(f"Failed to persist query record {query_id}: {e}")
            db.rollback()
