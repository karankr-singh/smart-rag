"""In-memory vector store with cosine-similarity search and disk persistence."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .chunking import Chunk


class VectorStore:
    """A minimal but complete vector index.

    Not FAISS-scale, but exact and dependency-free -- exactly right for
    experimentation on corpora up to a few hundred thousand chunks.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray | None = None  # (n, dim), L2-normalized

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of vectors")
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, cosine_similarity) pairs for a query vector."""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        query_vector = query_vector.reshape(1, -1)
        # Vectors are pre-normalized, so dot product == cosine similarity.
        scores = (self.vectors @ query_vector.T).ravel()
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {"n_chunks": len(self.chunks), "dim": int(self.vectors.shape[1]) if self.vectors is not None else 0}
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        path = Path(path)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
