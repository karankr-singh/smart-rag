"""End-to-end RAG orchestration: index -> retrieve -> compress -> prompt -> generate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .bm25 import tokenize
from .chunking import Chunk, chunk_recursive
from .embeddings import EmbeddingBackend, TfidfSvdEmbedder
from .llm import LLMBackend, MockLLM
from .retriever import HybridRetriever, RetrievalResult

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. Answer the user's question using "
    "ONLY the information in the provided context. If the context does not "
    "contain the answer, say so plainly instead of guessing. Cite which "
    "source(s) you used by their [source_id]."
)


@dataclass
class RAGResponse:
    answer: str
    query: str
    contexts: List[RetrievalResult] = field(default_factory=list)

    def sources(self) -> List[str]:
        return [f"{r.chunk.doc_id}#{r.chunk.position}" for r in self.contexts]

    def __repr__(self) -> str:  # pragma: no cover
        return f"RAGResponse(query={self.query!r}, n_sources={len(self.contexts)})"


def compress_chunk_to_query(chunk_text: str, query: str, max_sentences: int = 3) -> str:
    """Contextual compression: keep only the sentences most relevant to the
    query instead of feeding the whole chunk to the LLM. Cuts token usage
    and reduces the chance the model latches onto irrelevant nearby text.
    """
    from .chunking import _split_sentences  # local import to avoid circularity concerns

    sentences = _split_sentences(chunk_text)
    if len(sentences) <= max_sentences:
        return chunk_text

    q_tokens = set(tokenize(query))
    scored = []
    for s in sentences:
        s_tokens = set(tokenize(s))
        overlap = len(q_tokens & s_tokens)
        scored.append((overlap, s))

    # Keep top-scoring sentences but preserve their original order for readability.
    top_sentences = {s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]}
    ordered = [s for s in sentences if s in top_sentences]
    return " ".join(ordered) if ordered else chunk_text


class SmartRAG:
    """The main entry point: add documents, ask questions, get grounded answers."""

    def __init__(
        self,
        embedder: Optional[EmbeddingBackend] = None,
        llm: Optional[LLMBackend] = None,
        fetch_k: int = 20,
    ):
        self.embedder = embedder or TfidfSvdEmbedder()
        self.retriever = HybridRetriever(self.embedder, fetch_k=fetch_k)
        self.llm = llm or MockLLM()
        self._all_texts: List[str] = []  # kept for re-fitting embedder if needed

    def add_document(
        self,
        text: str,
        doc_id: str,
        chunk_size: int = 500,
        overlap: int = 50,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        chunks = chunk_recursive(text, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap, metadata=metadata)
        self._all_texts.append(text)
        self.retriever.index(chunks)
        return chunks

    def add_documents(self, docs: dict, **chunk_kwargs) -> None:
        """Convenience: docs is a {doc_id: text} mapping."""
        for doc_id, text in docs.items():
            self.add_document(text, doc_id=doc_id, **chunk_kwargs)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        return self.retriever.retrieve(query, top_k=top_k, **kwargs)

    def build_prompt(
        self,
        query: str,
        contexts: List[RetrievalResult],
        compress: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        blocks = []
        for r in contexts:
            text = compress_chunk_to_query(r.chunk.text, query) if compress else r.chunk.text
            source_id = f"{r.chunk.doc_id}#{r.chunk.position}"
            blocks.append(f"[source_id: {source_id}]\n{text}")
        context_block = "\n\n---\n\n".join(blocks) if blocks else "(no relevant context found)"

        return (
            f"{system_prompt}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
        compress: bool = True,
        max_tokens: int = 500,
        **retrieve_kwargs,
    ) -> RAGResponse:
        contexts = self.retrieve(query, top_k=top_k, **retrieve_kwargs)
        prompt = self.build_prompt(query, contexts, compress=compress)
        answer = self.llm.generate(prompt, max_tokens=max_tokens)
        return RAGResponse(answer=answer, query=query, contexts=contexts)
