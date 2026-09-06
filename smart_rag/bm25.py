"""Okapi BM25 sparse (keyword) retrieval, implemented from scratch (no extra deps).

Dense embeddings are great at topical/semantic similarity but can miss exact
identifiers, jargon, or rare terms (error codes, proper nouns, model names).
BM25 is the classic complement -- combining the two (see retriever.py) tends
to beat either alone.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len: float = 0.0
        self._n_docs: int = 0

    def fit(self, chunks: List[Chunk]) -> "BM25Index":
        self.chunks = list(chunks)
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        self._n_docs = len(self._doc_tokens)
        self._avg_doc_len = (
            sum(len(d) for d in self._doc_tokens) / self._n_docs if self._n_docs else 0.0
        )
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        return self

    def _idf(self, term: str) -> float:
        n = self._n_docs
        df = self._doc_freq.get(term, 0)
        # BM25+ style smoothing, always non-negative
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._n_docs == 0:
            return []
        q_terms = tokenize(query)
        scores = [0.0] * self._n_docs
        for i, doc_tokens in enumerate(self._doc_tokens):
            if not doc_tokens:
                continue
            term_counts = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for term in q_terms:
                if term not in term_counts:
                    continue
                f = term_counts[term]
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avg_doc_len)
                score += idf * (f * (self.k1 + 1)) / denom
            scores[i] = score

        ranked = sorted(range(self._n_docs), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in ranked if scores[i] > 0]
