# external_memory

External memory / vector store — long-term facts stored outside the conversation, retrieved by semantic similarity rather than always being resent in full.

## external_memory.py

An assistant that can save new facts to a persistent, writable vector store at runtime, and later retrieve only what's relevant to the current query. Type `exit` to end the conversation — saved facts persist in `external_memory.json` and are still there next run.

### Concepts covered

- **Persistence + semantic retrieval combined** — `../memory_management/basic_agentic_memory.py` persists facts but injects ALL of them wholesale on every call; `../../RAG_and_Knowledge/embedding/` retrieves selectively by similarity but is read-only. This template combines both: `save_memory` writes a new fact (embedded once, at write time), `search_memory` retrieves only the top-matching facts for the current query.
- **The shape of a real vector database integration** — a `save`/`search` tool pair backed by embed-and-compare is exactly what a Pinecone, Chroma, or Weaviate integration looks like from the application's side; this template's flat JSON file with a hand-rolled `embed()`/`cosine_similarity()` pair is the same mechanic as `../../RAG_and_Knowledge/embedding/embedding_search.py`, without needing an actual vector database.
- **Write-at-runtime, not just read-at-startup** — the store starts empty and grows as the conversation adds facts to it, unlike `../../RAG_and_Knowledge/embedding/`'s fixed knowledge base.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/external_memory/external_memory.py
```

Try:

```
You: Remember that I prefer dark mode in all my apps.
  [tool] save_memory({'fact': 'The user prefers dark mode in all apps'})
  [result] Saved to external memory: The user prefers dark mode in all apps
...
You: What do you know about my UI preferences?
  [tool] search_memory({'query': 'UI preferences'})
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `EMBEDDING_DIM` — vector length (default: `64`); see `../../RAG_and_Knowledge/embedding/README.md` for the trade-offs
- `STORE_FILE` — where facts are persisted (default: `external_memory.json` next to the script)
- `search_memory`'s `top_k` — how many facts are retrieved per query

### See also

- `../../RAG_and_Knowledge/embedding/README.md` — the embedding mechanic this template's store is built on, including its documented limitations (literal word overlap, not true meaning)
- `../semantic_memory/README.md` — a structured, categorized alternative for facts that need explicit updates rather than similarity-ranked recall
