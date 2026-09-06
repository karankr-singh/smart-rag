import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_rag import SmartRAG, chunk_fixed, chunk_sentences, chunk_recursive
from smart_rag.bm25 import BM25Index
from smart_rag.evaluation import precision_at_k, recall_at_k, mean_reciprocal_rank


SAMPLE_TEXT = (
    "The cat sat on the mat. It was a sunny day outside. "
    "Dogs are loyal companions and enjoy playing fetch. "
    "Cats, unlike dogs, are often more independent. "
    "The weather forecast predicts rain tomorrow evening."
)


def test_chunk_fixed_respects_size_and_overlap():
    chunks = chunk_fixed(SAMPLE_TEXT, doc_id="d1", chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_chunk_sentences_never_splits_mid_sentence():
    chunks = chunk_sentences(SAMPLE_TEXT, doc_id="d1", sentences_per_chunk=2, overlap_sentences=1)
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))


def test_chunk_recursive_produces_nonempty_chunks():
    chunks = chunk_recursive(SAMPLE_TEXT, doc_id="d1", chunk_size=60, overlap=10)
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)


def test_bm25_ranks_exact_keyword_match_higher():
    chunks = chunk_sentences(SAMPLE_TEXT, doc_id="d1", sentences_per_chunk=1, overlap_sentences=0)
    idx = BM25Index().fit(chunks)
    results = idx.search("dogs fetch", top_k=3)
    assert results, "BM25 should return at least one hit"
    assert "dogs" in results[0][0].text.lower() or "fetch" in results[0][0].text.lower()


def test_smart_rag_end_to_end_retrieval():
    rag = SmartRAG()
    rag.add_document(SAMPLE_TEXT, doc_id="animals", chunk_size=80, overlap=10)
    results = rag.retrieve("Tell me about dogs and their loyalty", top_k=2)
    assert len(results) > 0
    combined = " ".join(r.chunk.text.lower() for r in results)
    assert "dog" in combined


def test_smart_rag_query_returns_answer_and_sources():
    rag = SmartRAG()
    rag.add_document(SAMPLE_TEXT, doc_id="animals", chunk_size=80, overlap=10)
    response = rag.query("What does the forecast say?", top_k=2)
    assert response.answer
    assert isinstance(response.sources(), list)


def test_evaluation_metrics_perfect_case():
    from smart_rag.retriever import RetrievalResult
    from smart_rag.chunking import Chunk

    c1 = Chunk(text="relevant chunk", doc_id="docA", position=0, chunk_id="c1")
    c2 = Chunk(text="irrelevant chunk", doc_id="docB", position=0, chunk_id="c2")
    results = [RetrievalResult(c1, 1.0), RetrievalResult(c2, 0.5)]

    assert precision_at_k(results, ["docA"], k=1) == 1.0
    assert recall_at_k(results, ["docA"], k=2) == 1.0
    assert mean_reciprocal_rank(results, ["docA"]) == 1.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
