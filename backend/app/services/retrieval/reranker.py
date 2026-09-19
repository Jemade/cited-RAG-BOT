from typing import List, Dict, Any, Optional
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.services.retrieval.dense import RetrievalResult

class CrossEncoderReranker:
    _instance = None
    _model = None

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._init_model()

    def _init_model(self):
        if CrossEncoderReranker._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model '{self.model_name}'...")
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                logger.info("CrossEncoder model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ('{e}'). Using lexical overlap reranker fallback.")
                CrossEncoderReranker._model = "fallback"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = settings.RERANK_TOP_N
    ) -> List[RetrievalResult]:
        """Rerank candidates using Cross-Encoder and return top_n."""
        if not candidates:
            return []

        if CrossEncoderReranker._model != "fallback" and CrossEncoderReranker._model is not None:
            pairs = [[query, c.text] for c in candidates]
            scores = CrossEncoderReranker._model.predict(pairs)
            scores = [float(s) for s in scores]
        else:
            # Fallback lexical matching score
            import re
            q_terms = set(re.findall(r"\b\w+\b", query.lower()))
            scores = []
            for c in candidates:
                c_terms = set(re.findall(r"\b\w+\b", c.text.lower()))
                overlap = len(q_terms.intersection(c_terms)) / (len(q_terms) + 1e-6)
                # Map to logit-like scale around [-5.0, +5.0]
                base_logit = (overlap * 10.0) - 5.0
                scores.append(float(base_logit))

        # Attach rerank score to candidates and sort
        scored_candidates = []
        for score, cand in zip(scores, candidates):
            meta = dict(cand.metadata)
            meta["rerank_score"] = round(score, 4)
            cand_copy = RetrievalResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                page_number=cand.page_number,
                text=cand.text,
                score=score,
                rank=0,
                section=cand.section,
                metadata=meta
            )
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        # Assign new ranks to top_n
        reranked_top: List[RetrievalResult] = []
        for rank, cand in enumerate(scored_candidates[:top_n], start=1):
            cand.rank = rank
            cand.metadata["rerank_rank"] = rank
            reranked_top.append(cand)

        logger.info(f"Reranked {len(candidates)} candidates down to {len(reranked_top)}.")
        return reranked_top
