"""Hybrid retrieval: dense (semantic) + sparse (BM25), fused with Reciprocal
Rank Fusion, then optionally diversified with Maximal Marginal Relevance
and refined with a lightweight lexical-overlap reranker.

This is the "smart" part of Smart RAG -- each stage patches a known failure
mode of the previous one:

  dense search   -> misses exact keywords / rare terms
  + BM25         -> misses paraphrases / topical similarity
  + RRF fusion   -> combines both rankings without needing score calibration
  + MMR          -> avoids returning 5 near-duplicate chunks
  + rerank       -> cheap final polish using query-chunk term overlap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .embeddings import EmbeddingBackend
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


def reciprocal_rank_fusion(
    dense_ranking: List[Chunk], sparse_ranking: List[Chunk], k: int = 60
) -> List[RetrievalResult]:
    """Merge two rankings without needing their raw scores to be comparable.

    RRF score for a doc = sum over rankings it appears in of 1 / (k + rank).
    This is the standard, simple, and surprisingly strong fusion method used
    across hybrid-search literature.
    """
    scores: dict = {}
    dense_rank_map, sparse_rank_map = {}, {}

    for rank, chunk in enumerate(dense_ranking):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        dense_rank_map[chunk.chunk_id] = rank

    chunk_lookup = {c.chunk_id: c for c in dense_ranking}
    for rank, chunk in enumerate(sparse_ranking):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        sparse_rank_map[chunk.chunk_id] = rank
        chunk_lookup.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievalResult(
            chunk=chunk_lookup[cid],
            score=scores[cid],
            dense_rank=dense_rank_map.get(cid),
            sparse_rank=sparse_rank_map.get(cid),
        )
        for cid in ordered_ids
    ]


def mmr_diversify(
    results: List[RetrievalResult],
    vectors_by_chunk_id: dict,
    top_k: int,
    lambda_mult: float = 0.7,
) -> List[RetrievalResult]:
    """Maximal Marginal Relevance re-selection to reduce redundancy.

    Greedily picks the next chunk that is relevant to the query *and*
    dissimilar to what's already selected, so results aren't 5 near-copies
    of the same paragraph.
    """
    if not results:
        return []
    candidates = list(results)
    selected: List[RetrievalResult] = [candidates.pop(0)]

    while candidates and len(selected) < top_k:
        best_idx, best_val = 0, -1e9
        for i, cand in enumerate(candidates):
            cand_vec = vectors_by_chunk_id.get(cand.chunk.chunk_id)
            if cand_vec is None:
                sim_to_selected = 0.0
            else:
                sims = [
                    float(cand_vec @ vectors_by_chunk_id[s.chunk.chunk_id])
                    for s in selected
                    if vectors_by_chunk_id.get(s.chunk.chunk_id) is not None
                ]
                sim_to_selected = max(sims) if sims else 0.0
            mmr_score = lambda_mult * cand.score - (1 - lambda_mult) * sim_to_selected
            if mmr_score > best_val:
                best_val, best_idx = mmr_score, i
        selected.append(candidates.pop(best_idx))

    return selected[:top_k]


def lexical_rerank(query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Cheap final rerank pass using normalized query/chunk token overlap
    (Jaccard-style) as a tie-breaker signal blended with the fusion score.
    Approximates what a cross-encoder reranker would do, without needing
    one downloaded.
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return results
    rescored = []
    for r in results:
        c_tokens = set(tokenize(r.chunk.text))
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        blended = 0.7 * r.score + 0.3 * overlap
        rescored.append(RetrievalResult(r.chunk, blended, r.dense_rank, r.sparse_rank))
    return sorted(rescored, key=lambda r: r.score, reverse=True)


class HybridRetriever:
    """Ties together embeddings, the vector store, and BM25 into one
    retrieve() call that does fusion + diversification + reranking.
    """

    def __init__(self, embedder: EmbeddingBackend, fetch_k: int = 20):
        self.embedder = embedder
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self.fetch_k = fetch_k  # how many candidates each retriever pulls before fusion

    def index(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        if not self.embedder.is_fitted:
            self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.vector_store.add(chunks, vectors)
        self.bm25.fit(self.vector_store.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
        use_rerank: bool = True,
        mmr_lambda: float = 0.7,
    ) -> List[RetrievalResult]:
        if len(self.vector_store) == 0:
            return []

        query_vec = self.embedder.encode([query])[0]
        dense_hits = self.vector_store.search(query_vec, top_k=self.fetch_k)
        dense_ranking = [c for c, _ in dense_hits]

        sparse_hits = self.bm25.search(query, top_k=self.fetch_k)
        sparse_ranking = [c for c, _ in sparse_hits]

        fused = reciprocal_rank_fusion(dense_ranking, sparse_ranking)

        if use_rerank:
            fused = lexical_rerank(query, fused)

        if use_mmr:
            id_to_vec = {}
            for chunk, vec in zip(self.vector_store.chunks, self.vector_store.vectors):
                id_to_vec[chunk.chunk_id] = vec
            fused = mmr_diversify(fused, id_to_vec, top_k=top_k, lambda_mult=mmr_lambda)
        else:
            fused = fused[:top_k]

        return fused
