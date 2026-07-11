# rag

RAG (Retrieval-Augmented Generation) — dynamically fetching relevant documents from a knowledge base and injecting them into the prompt *before* generating a response.

## basic_rag.py

An internal docs Q&A assistant over a small company-policy knowledge base (PTO, expenses, remote work, onboarding). Implements the pipeline end to end: chunk → embed → retrieve → generate. Type `exit` to end the conversation.

### Concepts covered

- **Classic RAG vs. tool-based retrieval** — this template always retrieves before generating, on every query, with no model decision involved. Contrast with `../embedding/embedding_search.py` and `../../context_management/retrieval.py`, where retrieval is a *tool* the model decides whether and when to call. Classic RAG is simpler and deterministic; tool-based retrieval is more flexible but relies on the model deciding correctly.
- **The full pipeline wired together** — `chunk_text` (same idea as `../chunking/chunking_strategies.py`) splits each document, `embed`/`cosine_similarity` (same mechanic as `../embedding/embedding_search.py`) index and rank the chunks, and `answer_query` injects the top matches into a single direct API call — no `tools=` parameter, no agentic loop, because retrieval already happened in Python before Claude is ever called.
- **Retrieval transparency** — `answer_query` prints which chunks were retrieved and their similarity scores before generating, so you can see exactly what context Claude is grounded in.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/rag/basic_rag.py
```

Try prompts like:

```
You: how many days can I work remotely per week
  [retrieved 3 chunks]
    [remote-work-guidelines] (similarity: 0.445) Employees may work remotely up to 3 days per week...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `DOCUMENTS` — the knowledge base indexed at startup
- `chunk_text`'s `max_chunk_size` — how finely documents are split before embedding
- `retrieve`'s `top_k` — how many chunks are injected as context per query

### See also

- `../chunking/README.md` — the chunking strategies this template's `chunk_text` is a simplified version of
- `../embedding/README.md` — the embedding mechanic this template's `embed`/`cosine_similarity` reuses, with the fuller explanation of its limitations
- `../hybrid_rag/README.md` — adding sparse keyword search alongside this template's dense-only retrieval
