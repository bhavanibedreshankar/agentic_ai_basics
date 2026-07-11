# hybrid_rag

Hybrid RAG — combining dense vector retrieval (semantic similarity via embeddings) with sparse keyword search (BM25) so each covers the other's blind spot.

## hybrid_search.py

A support assistant over a knowledge base that deliberately includes rare, exact terms (error codes) alongside general how-to content. Type `exit` to end the conversation — each query prints a side-by-side comparison of dense-only, sparse-only, and hybrid rankings before generating an answer.

### Concepts covered

- **Dense search's blind spot** — good at matching meaning, but can under-rank rare, specific terms (an error code, a product SKU) because they get diluted among common words in a fixed-size vector.
- **Sparse search's blind spot** — BM25 is excellent at exact term matching (the rarer a term across the corpus, the more it weights a matching document) but has no notion of meaning, so paraphrased queries score poorly.
- **`_bm25_score`** — a from-scratch implementation of the BM25 ranking formula: term frequency (with diminishing returns via `K1`), inverse document frequency (rarer terms score higher via `_idf`), and document-length normalization (via `B`), commented line by line.
- **`_normalize`** — dense (cosine similarity) and sparse (BM25) scores live on completely different numeric scales; min-max normalizing both to `[0, 1]` is what makes combining them meaningful instead of letting whichever metric produces bigger raw numbers dominate.
- **`hybrid_search`** — blends the two normalized rankings via a tunable `alpha` weight. In testing, a query like *"my card got rejected, what happened"* causes dense search alone to rank the wrong document first — hybrid search recovers the correct answer via the BM25 signal.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/hybrid_rag/hybrid_search.py
```

Try a query that benefits from exact matching:

```
You: I'm seeing error E-4521, what does it mean?

  --- ranking comparison ---
  dense only  : error-e4521 (1.000), error-e1032 (0.969), password-reset (0.413)
  sparse only : error-e4521 (1.000), error-e1032 (0.377), password-reset (0.000)
  hybrid      : error-e4521 (1.000), error-e1032 (0.673), password-reset (0.206)
```

Notice dense search alone barely distinguishes the two error codes (0.969 vs. 1.0); BM25 discriminates far more confidently via the exact rare term.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `DOCUMENTS` — the knowledge base indexed at startup
- `K1` / `B` — BM25 tuning parameters (term-frequency saturation and length normalization)
- `hybrid_search`'s `alpha` — the dense/sparse blend weight (`1.0` = pure dense, `0.0` = pure sparse, `0.5` = equal)

### See also

- `../embedding/README.md` — the dense half of this template's search, on its own
- `../rag/README.md` — the dense-only pipeline this template adds sparse search to
