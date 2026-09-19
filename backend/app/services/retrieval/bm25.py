import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.db.models import Chunk
from app.services.retrieval.dense import RetrievalResult

class BM25Retriever:
    _index_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text preserving alphanumeric terms, hyphens, and decimals."""
        text = text.lower()
        # Find alphanumeric tokens including hyphens or decimals within words
        tokens = re.findall(r"\b[a-z0-9]+(?:[-.][a-z0-9]+)*\b", text)
        return tokens

    def build_index(self, chunks: List[Chunk], cache_key: str):
        """Build and cache BM25 index for a set of chunks."""
        tokenized_corpus = [self.tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        BM25Retriever._index_cache[cache_key] = {
            "bm25": bm25,
            "chunks": chunks
        }
        logger.info(f"Built BM25 index for key '{cache_key}' with {len(chunks)} chunks.")

    def search(
        self,
        db: Session,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = settings.BM25_TOP_K
    ) -> List[RetrievalResult]:
        """
        Retrieve top_k chunks matching query keywords using BM25.
        """
        cache_key = document_id if document_id else "global"

        if cache_key not in BM25Retriever._index_cache:
            # Query chunks from database to build index
            query_obj = db.query(Chunk)
            if document_id:
                query_obj = query_obj.filter(Chunk.document_id == document_id)
            chunks = query_obj.order_by(Chunk.chunk_index).all()
            if not chunks:
                return []
            self.build_index(chunks, cache_key)

        cached = BM25Retriever._index_cache[cache_key]
        bm25: BM25Okapi = cached["bm25"]
        chunks: List[Chunk] = cached["chunks"]

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = bm25.get_scores(tokenized_query)
        # Filter non-zero scores or sort all
        scored_pairs = [(score, chunk) for score, chunk in zip(doc_scores, chunks) if score > 0]
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, chunk) in enumerate(scored_pairs[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                text=chunk.text,
                score=float(score),
                rank=rank,
                section=chunk.section,
                metadata=chunk.metadata_json
            ))

        return results

    @classmethod
    def invalidate_cache(cls, document_id: Optional[str] = None):
        """Invalidate cached BM25 index."""
        if document_id and document_id in cls._index_cache:
            del cls._index_cache[document_id]
        elif document_id is None:
            cls._index_cache.clear()
