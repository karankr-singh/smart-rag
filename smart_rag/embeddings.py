"""Embedding backends.

`EmbeddingBackend` is the interface every embedder implements, so the rest
of the pipeline (vector store, retriever) never cares which one is in use.

`TfidfSvdEmbedder` is the default: it fits TF-IDF + Truncated SVD (i.e. a
Latent Semantic Analysis model) directly on your corpus, entirely offline
with no model download and no API calls -- good for experimentation and
for environments without network access to model hubs.

For production-quality semantic embeddings, swap in a real dense model by
implementing `EmbeddingBackend` around sentence-transformers, OpenAI, or
the Anthropic API's embedding-capable partners. See `EchoBackendExample`
docstring at the bottom for the shape to follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class EmbeddingBackend(ABC):
    """Interface for turning text into fixed-size vectors."""

    @abstractmethod
    def fit(self, corpus: List[str]) -> "EmbeddingBackend":
        """Fit any corpus-dependent statistics (vocabulary, etc.)."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of L2-normalized embeddings."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        ...


class TfidfSvdEmbedder(EmbeddingBackend):
    """Local semantic embedder: TF-IDF followed by Truncated SVD (LSA).

    Plain TF-IDF captures only exact lexical overlap. Projecting it through
    SVD collapses co-occurring terms into shared latent dimensions, so
    queries and documents that share *topic* rather than *exact words* end
    up close together in vector space -- a cheap, fully local proxy for
    semantic embeddings.
    """

    def __init__(
        self,
        n_components: int = 128,
        ngram_range: tuple = (1, 2),
        max_features: Optional[int] = 20000,
        min_df: int = 1,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd: Optional[TruncatedSVD] = None
        self._fitted = False
        self.random_state = random_state

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: List[str]) -> "TfidfSvdEmbedder":
        if not corpus:
            raise ValueError("Cannot fit embedder on an empty corpus")
        tfidf = self.vectorizer.fit_transform(corpus)
        # n_components must be < n_features and < n_samples
        n_comp = max(2, min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("Embedder must be fit() on a corpus before encode()")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense).astype(np.float32)


# ---------------------------------------------------------------------------
# Shape to follow if you want to plug in a real dense-embedding API/model.
# Not used by default (keeps this project offline-runnable), but shown here
# as documentation for extending the system.
#
# class SentenceTransformerEmbedder(EmbeddingBackend):
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(model_name)
#         self._fitted = True  # pretrained, no corpus fitting needed
#
#     def fit(self, corpus):
#         return self  # no-op: model is pretrained
#
#     def encode(self, texts):
#         import numpy as np
#         vecs = self.model.encode(texts, normalize_embeddings=True)
#         return np.asarray(vecs, dtype=np.float32)
#
#     @property
#     def is_fitted(self):
#         return self._fitted
# ---------------------------------------------------------------------------
