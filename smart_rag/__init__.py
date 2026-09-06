"""
Smart RAG
=========

A lightweight, dependency-minimal Retrieval-Augmented Generation toolkit
focused on improving semantic retrieval quality and contextual responses.

Core pieces:
- chunking:    document splitting strategies (fixed, sentence, recursive)
- embeddings:  pluggable text embedding backends (local TF-IDF+SVD by
               default; drop-in slots for OpenAI/Anthropic/sentence-transformers)
- vector_store: in-memory cosine-similarity vector index with persistence
- bm25:        classic keyword (sparse) retrieval
- retriever:   hybrid dense+sparse retrieval with Reciprocal Rank Fusion,
               MMR diversification, and lightweight reranking
- pipeline:    end-to-end RAG orchestration (retrieve -> compress -> prompt -> generate)
- llm:         pluggable generation backends (mock + Anthropic API)
- evaluation:  retrieval quality metrics (Precision@k, Recall@k, MRR, NDCG)
"""

from .pipeline import SmartRAG
from .chunking import chunk_fixed, chunk_sentences, chunk_recursive
from .embeddings import TfidfSvdEmbedder, EmbeddingBackend
from .vector_store import VectorStore
from .retriever import HybridRetriever

__all__ = [
    "SmartRAG",
    "chunk_fixed",
    "chunk_sentences",
    "chunk_recursive",
    "TfidfSvdEmbedder",
    "EmbeddingBackend",
    "VectorStore",
    "HybridRetriever",
]

__version__ = "0.1.0"
