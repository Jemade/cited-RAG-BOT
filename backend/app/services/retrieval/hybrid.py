from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.retrieval.dense import RetrievalResult

class HybridRetriever:
    def __init__(self, rrf_k: int = settings.RRF_K):
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int = settings.DENSE_TOP_K
    ) -> List[RetrievalResult]:
        """Combine dense and BM25 results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}
        retrieval_meta: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense rankings
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            chunk_map[cid] = res
            rrf_val = 1.0 / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_val
            retrieval_meta[cid] = {
                "dense_rank": rank,
                "dense_score": res.score,
                "bm25_rank": None,
                "bm25_score": None,
            }

        # 2. Process BM25 rankings
        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_val = 1.0 / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_val

            if cid not in retrieval_meta:
                retrieval_meta[cid] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": res.score,
                }
            else:
                retrieval_meta[cid]["bm25_rank"] = rank
                retrieval_meta[cid]["bm25_score"] = res.score

        # 3. Sort by RRF score descending
        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        fused_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            base_res = chunk_map[cid]
            meta = dict(base_res.metadata)
            meta.update(retrieval_meta[cid])
            meta["rrf_score"] = scores[cid]

            fused_results.append(RetrievalResult(
                chunk_id=base_res.chunk_id,
                document_id=base_res.document_id,
                page_number=base_res.page_number,
                text=base_res.text,
                score=scores[cid],
                rank=rank,
                section=base_res.section,
                metadata=meta
            ))

        return fused_results

    def weighted_score_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        alpha: float = 0.5,
        top_k: int = settings.DENSE_TOP_K
    ) -> List[RetrievalResult]:
        """Combine dense and BM25 results via normalized weighted fusion."""
        chunk_map: Dict[str, RetrievalResult] = {}
        dense_scores: Dict[str, float] = {r.chunk_id: r.score for r in dense_results}
        bm25_scores: Dict[str, float] = {r.chunk_id: r.score for r in bm25_results}

        for r in dense_results:
            chunk_map[r.chunk_id] = r
        for r in bm25_results:
            if r.chunk_id not in chunk_map:
                chunk_map[r.chunk_id] = r

        # Min-max normalization helper
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v - min_v < 1e-9:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

        norm_dense = normalize(dense_scores)
        norm_bm25 = normalize(bm25_scores)

        combined_scores: Dict[str, float] = {}
        for cid in chunk_map:
            d_val = norm_dense.get(cid, 0.0)
            b_val = norm_bm25.get(cid, 0.0)
            combined_scores[cid] = (alpha * d_val) + ((1.0 - alpha) * b_val)

        sorted_cids = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)

        results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            base_res = chunk_map[cid]
            meta = dict(base_res.metadata)
            meta["weighted_score"] = combined_scores[cid]
            results.append(RetrievalResult(
                chunk_id=base_res.chunk_id,
                document_id=base_res.document_id,
                page_number=base_res.page_number,
                text=base_res.text,
                score=combined_scores[cid],
                rank=rank,
                section=base_res.section,
                metadata=meta
            ))

        return results
