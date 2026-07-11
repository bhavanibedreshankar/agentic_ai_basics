# coarse_to_fine_retrieval

Coarse-to-fine retrieval — first retrieving broadly to narrow down a large corpus, then retrieving again within just that shortlist at a finer grain to pinpoint the specific relevant section.

## coarse_to_fine_retrieval.py

A company-docs assistant over three longer, multi-topic handbooks (employee, engineering, security). Type `exit` to end the conversation — each query prints both retrieval stages before generating an answer.

### Concepts covered

- **Why two stages** — at small scale (a handful of documents), searching every fine-grained chunk directly (like `../rag/basic_rag.py` does) is simplest and there's no real cost problem. Coarse-to-fine earns its complexity at *larger* scale: embedding and scoring every chunk of a huge corpus against every query is expensive, so a cheap first pass narrows the field before the expensive, precise pass runs.
- **Stage 1 — `coarse_search`** — embeds each *whole document* as a single vector and ranks all documents. Imprecise (it averages together every topic a document covers) but cheap — exactly right for "is this document even in the right ballpark."
- **Stage 2 — `fine_search`** — splits *only* the shortlisted documents into paragraph-level chunks and ranks those individually. More expensive per item, but only ever run on a small subset instead of the whole corpus.
- **The funnel in action** — `answer_query` prints how many documents were shortlisted at stage 1 and how many candidate sections were ranked at stage 2, making the narrowing effect visible.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/coarse_to_fine_retrieval/coarse_to_fine_retrieval.py
```

Try prompts like:

```
You: how often are passwords rotated

  [stage 1 - coarse: shortlisted 1 of 3 documents]
    security-policy: 0.406
  [stage 2 - fine: ranked 3 candidate sections, kept top 2]
    [security-policy] (similarity: 0.447) Passwords must be at least 14 characters...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `DOCUMENTS` — the handbooks indexed at startup (each with multiple `\n\n`-separated sections)
- `answer_query`'s `top_n_docs` — how many documents survive stage 1
- `answer_query`'s `top_k_chunks` — how many sections survive stage 2

### See also

- `../embedding/README.md` — the embedding mechanic both stages reuse
- `../chunking/README.md` — the paragraph-splitting idea `fine_search` applies at stage 2
- `../rag/README.md` — the single-stage alternative this template's two-stage funnel is built to scale beyond
