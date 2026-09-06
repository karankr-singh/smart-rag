"""
Smart RAG demo
==============

Indexes the sample corpus in data/sample_docs/, runs a few queries through
the hybrid retriever, prints grounded answers with sources, and reports
retrieval quality metrics against a small hand-labeled evaluation set.

Run with:
    python demo.py

To use real generation instead of the offline mock, set ANTHROPIC_API_KEY
and pass llm=AnthropicLLM() when constructing SmartRAG below.
"""
from pathlib import Path

from smart_rag import SmartRAG
from smart_rag.evaluation import evaluate_queries

DOCS_DIR = Path(__file__).parent / "data" / "sample_docs"


def load_corpus() -> dict:
    return {p.stem: p.read_text() for p in sorted(DOCS_DIR.glob("*.txt"))}


def main():
    print("=" * 70)
    print("SMART RAG DEMO")
    print("=" * 70)

    rag = SmartRAG()  # default: TF-IDF+SVD embedder, offline MockLLM

    corpus = load_corpus()
    print(f"\nIndexing {len(corpus)} documents: {list(corpus.keys())}")
    rag.add_documents(corpus, chunk_size=400, overlap=60)
    print(f"Indexed {len(rag.retriever.vector_store)} chunks total.\n")

    queries = [
        "How does approximate nearest neighbor search work?",
        "Why do RAG systems reduce hallucination?",
        "What is the difference between TF-IDF and neural embeddings?",
    ]

    for q in queries:
        print("-" * 70)
        print(f"QUERY: {q}")
        response = rag.query(q, top_k=3)
        print(f"\nRetrieved sources: {response.sources()}")
        for r in response.contexts:
            print(f"  [{r.chunk.doc_id}#{r.chunk.position}] score={r.score:.4f}")
            print(f"    {r.chunk.text[:150].strip()}...")
        print(f"\nANSWER:\n{response.answer}\n")

    # --- Retrieval quality evaluation against hand-labeled relevance ---
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    eval_set = [
        {
            "query": "How does approximate nearest neighbor search work?",
            "relevant_ids": ["vector_databases"],
        },
        {
            "query": "Why do RAG systems reduce hallucination?",
            "relevant_ids": ["rag_overview"],
        },
        {
            "query": "What is the difference between TF-IDF and neural embeddings?",
            "relevant_ids": ["embeddings_explained"],
        },
        {
            "query": "What are the three stages of a typical RAG pipeline?",
            "relevant_ids": ["rag_overview"],
        },
    ]

    metrics = evaluate_queries(
        retriever_fn=lambda q, k: rag.retrieve(q, top_k=k),
        queries_with_relevant=eval_set,
        k=3,
    )

    print(f"\nOver {metrics['n_queries']} queries @k={metrics['k']}:")
    print(f"  Precision@k : {metrics['precision_at_k']:.3f}")
    print(f"  Recall@k    : {metrics['recall_at_k']:.3f}")
    print(f"  MRR         : {metrics['mrr']:.3f}")
    print(f"  NDCG@k      : {metrics['ndcg_at_k']:.3f}")


if __name__ == "__main__":
    main()
