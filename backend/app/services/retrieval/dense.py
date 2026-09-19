from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.core.logging import logger
from app.db.models import Chunk

class RetrievalResult:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        page_number: int,
        text: str,
        score: float,
        rank: int = 1,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_number = page_number
        self.text = text
        self.score = float(score)
        self.rank = rank
        self.section = section
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "text": self.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "section": self.section,
            "metadata": self.metadata
        }

class DenseRetriever:
    _instance = None
    _model = None

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._init_model()

    def _init_model(self):
        if DenseRetriever._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading dense embedding model '{self.model_name}'...")
                DenseRetriever._model = SentenceTransformer(self.model_name)
                logger.info("Dense embedding model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ('{e}'). Using normalized hash embedding fallback.")
                DenseRetriever._model = "fallback"

    def embed_text(self, text_input: str) -> List[float]:
        """Embed a single string into a 384-dimensional normalized vector."""
        if DenseRetriever._model != "fallback" and DenseRetriever._model is not None:
            embedding = DenseRetriever._model.encode(text_input, normalize_embeddings=True)
            return embedding.tolist()
        else:
            # Deterministic fallback embedding for testing / fallback mode
            import hashlib
            h = hashlib.sha256(text_input.encode()).digest()
            vec = np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)
            for i, byte in enumerate(h):
                vec[i % settings.EMBEDDING_DIM] += (byte - 128) / 128.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        if not texts:
            return []
        if DenseRetriever._model != "fallback" and DenseRetriever._model is not None:
            embeddings = DenseRetriever._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        else:
            return [self.embed_text(t) for t in texts]

    def search_pgvector(
        self,
        db: Session,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = settings.DENSE_TOP_K
    ) -> List[RetrievalResult]:
        """Retrieve top_k chunks by cosine similarity."""
        query_vec = self.embed_text(query)
        vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        # Check if database is PostgreSQL
        if db.bind.dialect.name == "postgresql":
            sql = """
                SELECT 
                    id, document_id, page_number, chunk_index, text, section, metadata_json,
                    1 - (embedding <=> :query_vec) AS similarity
                FROM chunks
                WHERE (:doc_id IS NULL OR document_id = :doc_id)
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> :query_vec ASC
                LIMIT :limit;
            """
            rows = db.execute(text(sql), {
                "query_vec": vec_str,
                "doc_id": document_id,
                "limit": top_k
            }).fetchall()

            results = []
            for rank, r in enumerate(rows, start=1):
                results.append(RetrievalResult(
                    chunk_id=r[0],
                    document_id=r[1],
                    page_number=r[2],
                    text=r[4],
                    score=float(r[7]),
                    rank=rank,
                    section=r[5],
                    metadata=r[6]
                ))
            return results
        else:
            # In-memory cosine similarity fallback for SQLite/tests
            query_obj = db.query(Chunk)
            if document_id:
                query_obj = query_obj.filter(Chunk.document_id == document_id)
            chunks = query_obj.all()

            if not chunks:
                return []

            q_vec = np.array(query_vec, dtype=np.float32)
            scored_chunks = []
            for c in chunks:
                if c.embedding:
                    c_vec = np.array(c.embedding, dtype=np.float32)
                    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
                    scored_chunks.append((sim, c))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            results = []
            for rank, (sim, c) in enumerate(scored_chunks[:top_k], start=1):
                results.append(RetrievalResult(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    page_number=c.page_number,
                    text=c.text,
                    score=sim,
                    rank=rank,
                    section=c.section,
                    metadata=c.metadata_json
                ))
            return results
