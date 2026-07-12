# RAG_and_Knowledge

Retrieval-Augmented Generation and the building blocks it's made of. Five templates, each in its own directory, going from the smallest mechanic (embedding) up through a full pipeline (RAG) to two ways of scaling or improving it (hybrid search, coarse-to-fine retrieval).

None of these use a real embeddings API or vector database — every template implements a small, dependency-free stand-in (a hashing-based "embedding" and, in some cases, BM25) purely in Python, so nothing here needs anything beyond `ANTHROPIC_API_KEY`. Each template is explicit in its comments about what it's simplifying and why — see `embedding/README.md` for the fullest explanation of that trade-off.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`embedding/`](embedding/README.md) | Converting text into a vector and ranking by cosine similarity — the core mechanic everything else here builds on |
| 2 | [`chunking/`](chunking/README.md) | Splitting documents into smaller pieces before they're embedded, and the size trade-offs involved — pure text processing, no API calls |
| 3 | [`rag/`](rag/README.md) | The full pipeline: chunk → embed → retrieve → generate, wired together as classic (non-tool-based) RAG |
| 4 | [`hybrid_rag/`](hybrid_rag/README.md) | Combining dense (embedding) search with sparse (BM25) keyword search to cover each other's blind spots |
| 5 | [`coarse_to_fine_retrieval/`](coarse_to_fine_retrieval/README.md) | A two-stage search funnel — broad document-level retrieval first, then precise section-level retrieval only within the shortlist |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 RAG_and_Knowledge/rag/basic_rag.py
```

`chunking/chunking_strategies.py` is the one exception — it's pure text processing with no API calls, so it runs without `ANTHROPIC_API_KEY` set.

## How this relates to `../Task_and_State_Management/context_management/`

`../Task_and_State_Management/context_management/retrieval.py` already covers the *idea* of retrieval (pulling in only relevant content on demand) using simple keyword overlap. This directory goes deeper into the mechanics that power real RAG systems — vector embeddings, chunking strategies, hybrid dense+sparse search, and multi-stage retrieval — the pieces `../Task_and_State_Management/context_management/retrieval.py` deliberately kept simple.
