"""Retrieval quality metrics.

Given a query and the set of chunk/doc ids that are actually relevant to it
(ground truth), these compute the standard IR metrics so you can quantify
whether a chunking strategy, embedder, or retrieval config is actually an
improvement -- rather than eyeballing results.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .retriever import RetrievalResult


def _matched_id(r: RetrievalResult, relevant: set) -> str | None:
    """Return whichever relevant identifier this result satisfies (chunk id,
    doc id, or doc#position), or None. Matching against several granularities
    lets ground truth be labeled at whatever level is convenient (a whole
    source document, or one specific chunk).
    """
    cid = r.chunk.chunk_id
    doc_ref = f"{r.chunk.doc_id}#{r.chunk.position}"
    if cid in relevant:
        return cid
    if doc_ref in relevant:
        return doc_ref
    if r.chunk.doc_id in relevant:
        return r.chunk.doc_id
    return None


def _hit_flags(results: Sequence[RetrievalResult], relevant_ids: Iterable[str]) -> List[bool]:
    """Per-position hit/miss, used for precision and MRR (order matters,
    duplicates against the same relevant id are fine here)."""
    relevant = set(relevant_ids)
    return [_matched_id(r, relevant) is not None for r in results]


def precision_at_k(results: Sequence[RetrievalResult], relevant_ids: Iterable[str], k: int) -> float:
    top = results[:k]
    if not top:
        return 0.0
    hits = _hit_flags(top, relevant_ids)
    return sum(hits) / len(top)


def recall_at_k(results: Sequence[RetrievalResult], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of *distinct* relevant ids covered by the top-k results.

    Uses distinct matches (not raw hit count) so that retrieving several
    chunks from the same relevant document can't push recall above 1.0.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = results[:k]
    covered = {m for r in top if (m := _matched_id(r, relevant)) is not None}
    return len(covered) / len(relevant)


def mean_reciprocal_rank(results: Sequence[RetrievalResult], relevant_ids: Iterable[str]) -> float:
    hits = _hit_flags(results, relevant_ids)
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: Sequence[RetrievalResult], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    top = results[:k]
    seen: set = set()
    dcg = 0.0
    for i, r in enumerate(top):
        match = _matched_id(r, relevant)
        # Only the first chunk that satisfies a given relevant id contributes
        # gain -- otherwise retrieving the same relevant doc twice would
        # inflate nDCG past 1.0.
        if match is not None and match not in seen:
            seen.add(match)
            dcg += 1.0 / math.log2(i + 2)
    n_relevant = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_queries(
    retriever_fn,
    queries_with_relevant: List[dict],
    k: int = 5,
) -> dict:
    """Aggregate metrics across a labeled evaluation set.

    `queries_with_relevant`: list of {"query": str, "relevant_ids": [...]}
    `retriever_fn`: callable(query, top_k) -> List[RetrievalResult]
    """
    totals = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    n = len(queries_with_relevant)
    if n == 0:
        return {**totals, "n_queries": 0}

    per_query = []
    for item in queries_with_relevant:
        results = retriever_fn(item["query"], k)
        relevant = item["relevant_ids"]
        p = precision_at_k(results, relevant, k)
        r = recall_at_k(results, relevant, k)
        mrr = mean_reciprocal_rank(results, relevant)
        ndcg = ndcg_at_k(results, relevant, k)
        totals["precision"] += p
        totals["recall"] += r
        totals["mrr"] += mrr
        totals["ndcg"] += ndcg
        per_query.append(
            {"query": item["query"], "precision": p, "recall": r, "mrr": mrr, "ndcg": ndcg}
        )

    return {
        "precision_at_k": totals["precision"] / n,
        "recall_at_k": totals["recall"] / n,
        "mrr": totals["mrr"] / n,
        "ndcg_at_k": totals["ndcg"] / n,
        "k": k,
        "n_queries": n,
        "per_query": per_query,
    }
