# Smart RAG

Retrieval-augmented generation experimentation focused on improving semantic
retrieval and contextual responses.

**Python • RAG • Embeddings • Vector Search**

A small, dependency-light toolkit for experimenting with what actually
moves the needle in RAG systems: chunking strategy, hybrid dense+keyword
retrieval, reranking/diversification, and contextual compression — plus
metrics to measure whether a change actually helped.

Runs fully offline out of the box (no API keys, no model downloads): the
default embedder is a local TF-IDF + SVD (LSA) model fit on your own
corpus, and the default "LLM" is a mock generator so you can validate the
retrieval pipeline in isolation. Swap in a real embedding model or the
Anthropic API for production-quality generation whenever you're ready.

## Why these design choices

| Problem | What this project does about it |
|---|---|
| Dense search misses exact keywords/rare terms | Hybrid retrieval: dense (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion |
| Top-k results are near-duplicates of each other | Maximal Marginal Relevance (MMR) re-selection for diversity |
| Long chunks dilute relevance / waste tokens | Contextual compression trims each chunk to its most query-relevant sentences before prompting |
| Chunks cut mid-sentence or mid-paragraph | Three chunking strategies: fixed-size, sentence-aware, and structure-aware recursive splitting |
| "It looks better" isn't a metric | Built-in Precision@k, Recall@k, MRR, and NDCG@k evaluation against labeled relevance |
| Coupling to one embedding/LLM vendor | Both are pluggable interfaces (`EmbeddingBackend`, `LLMBackend`) |

## Project structure

```
smart-rag/
├── smart_rag/
│   ├── chunking.py      # fixed / sentence / recursive document chunking
│   ├── embeddings.py    # pluggable embedding backends (TF-IDF+SVD by default)
│   ├── vector_store.py  # in-memory cosine-similarity vector index + persistence
│   ├── bm25.py          # from-scratch Okapi BM25 keyword retrieval
│   ├── retriever.py     # hybrid retrieval: RRF fusion + MMR + lexical rerank
│   ├── llm.py           # pluggable generation backends (Mock + Anthropic API)
│   ├── pipeline.py       # SmartRAG: the end-to-end orchestrator
│   └── evaluation.py    # Precision@k / Recall@k / MRR / NDCG@k
├── data/sample_docs/    # sample corpus (RAG, embeddings, vector DB explainers)
├── tests/test_rag.py    # pytest unit tests
├── demo.py              # runnable end-to-end walkthrough
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from smart_rag import SmartRAG

rag = SmartRAG()

rag.add_document(
    "Retrieval-Augmented Generation combines a retrieval system with a "
    "generative model, grounding answers in retrieved context instead of "
    "relying solely on parametric memory.",
    doc_id="rag_intro",
)

response = rag.query("How does RAG reduce hallucination?")
print(response.answer)
print(response.sources())   # -> ['rag_intro#0']
```

Run the full walkthrough (indexes the sample corpus, runs several queries,
prints retrieval scores + grounded answers, and reports evaluation metrics):

```bash
python demo.py
```

Run the tests:

```bash
pytest tests/ -v
```

## Using real generation

By default `SmartRAG()` uses `MockLLM`, an offline stand-in that just shows
the retrieved context flowing into the prompt — useful for judging
retrieval quality independent of generation quality.

For real answers, install the `anthropic` package, set `ANTHROPIC_API_KEY`,
and pass an `AnthropicLLM`:

```python
from smart_rag import SmartRAG
from smart_rag.llm import AnthropicLLM

rag = SmartRAG(llm=AnthropicLLM(model="claude-sonnet-4-6"))
```

## Using a stronger embedder

The default `TfidfSvdEmbedder` is fully local (no downloads), which keeps
the project runnable anywhere. If you have access to a dense embedding
model (sentence-transformers, an API, etc.), implement the small
`EmbeddingBackend` interface — `embeddings.py` includes a commented example
for `sentence-transformers` — and pass it in:

```python
rag = SmartRAG(embedder=MyDenseEmbedder())
```

Everything downstream (vector store, hybrid retrieval, MMR, reranking,
evaluation) works unchanged regardless of which embedder is plugged in.

## Evaluating retrieval quality

```python
from smart_rag.evaluation import evaluate_queries

eval_set = [
    {"query": "What is hybrid search?", "relevant_ids": ["vector_databases"]},
    # ground truth can be a doc_id, "doc_id#position", or a specific chunk_id
]

metrics = evaluate_queries(
    retriever_fn=lambda q, k: rag.retrieve(q, top_k=k),
    queries_with_relevant=eval_set,
    k=5,
)
print(metrics["precision_at_k"], metrics["recall_at_k"], metrics["mrr"], metrics["ndcg_at_k"])
```

Use this to A/B test chunking strategies, `fetch_k`, MMR's `lambda_mult`,
or a swapped-in embedder against a fixed labeled query set, instead of
judging changes by eye.

## Extending

- **New chunking strategy**: add a function to `chunking.py` returning `List[Chunk]`.
- **New embedder**: implement `EmbeddingBackend` (`fit`, `encode`, `is_fitted`).
- **New generator**: implement `LLMBackend` (`generate`).
- **Persist an index**: `rag.retriever.vector_store.save("path/")` /
  `VectorStore.load("path/")`.
