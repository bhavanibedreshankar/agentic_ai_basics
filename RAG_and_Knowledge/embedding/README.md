# embedding

Converting text into a vector representation for semantic similarity search — the core mechanic every other template in `RAG_and_Knowledge/` builds on.

## embedding_search.py

A documentation assistant, same knowledge base shape as `../../Task_and_State_Management/context_management/retrieval.py`, but backed by vector search instead of keyword overlap. Type `exit` to end the conversation.

### Concepts covered

- **`embed(text)`** — converts text into a fixed-length list of floats using "feature hashing": each word hashes into one of 64 buckets, which increments a count. This is a **dependency-free stand-in** for a real embedding model (Anthropic recommends Voyage AI; OpenAI's `text-embedding-3` and local `sentence-transformers` models are common alternatives — the Claude API itself doesn't produce embeddings). It captures word *overlap*, not true meaning.
- **`cosine_similarity(a, b)`** — the actual comparison operation every vector database is built around, just applied here to 5 documents instead of millions.
- **Building the index once** — vectors are computed up front for every document (`INDEX`), not recomputed per query. Embedding is the expensive step in a real system (an API call per document); do it once.
- **Being honest about limitations** — this technique only matches literal word overlap. "Purchase" and "buy" hash to unrelated buckets even though they mean the same thing; a real embedding model is trained specifically to avoid that failure. The docstring and comments call this out explicitly rather than hiding it — a query like *"can I work from home"* won't match a document that only says *"remotely"*, while *"how many days can I work remotely"* will.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/embedding/embedding_search.py
```

Try prompts like:

```
You: how do I commit changes with git
  [vector search: top 2 of 5 documents]
    git-basics: 0.509
    python-typing: 0.080
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `EMBEDDING_DIM` — vector length (default: `64`); higher reduces hash collisions at the cost of more computation
- `KNOWLEDGE_BASE` — the documents indexed at startup

### See also

- `../chunking/README.md` — splitting documents before they're embedded, which this template's knowledge base is small enough to skip
- `../rag/README.md` — this same embedding mechanic, wired into a full chunk → embed → retrieve → generate pipeline
