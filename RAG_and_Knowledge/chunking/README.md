# chunking

Splitting documents into smaller pieces for efficient vector indexing and retrieval — the preprocessing step that happens *before* embedding.

## chunking_strategies.py

Pure text processing — no API calls, no `ANTHROPIC_API_KEY` needed. Run it directly to see three chunking strategies applied to the same sample document, side by side.

### Concepts covered

- **Why chunk at all** — embedding models and context windows have limits on how much text fits meaningfully in one vector, and returning a whole document for a query about one small part of it wastes context on everything irrelevant.
- **The size trade-off** — chunks too small lose surrounding context; chunks too large dilute relevance and cost more tokens once retrieved. There's no universally correct chunk size.
- **`chunk_fixed_size`** — the simplest strategy: cut every N characters, no regard for word or sentence boundaries. Fast, but visibly slices words in half at the cut point.
- **`chunk_by_sentences`** — split into sentences first, then greedily pack them into chunks up to a max size. Never cuts a sentence, at the cost of uneven chunk sizes.
- **`chunk_with_overlap`** — fixed-size chunking where each chunk repeats the tail end of the previous one, so content sitting right at a boundary appears complete in at least one chunk. The cost: more total characters get embedded.

### Run

From the repo root:

```bash
python3 RAG_and_Knowledge/chunking/chunking_strategies.py
```

Output shows all three strategies side by side with chunk counts, average sizes, and previews — including where fixed-size chunking visibly cuts a word in half (e.g. `"langua..."`) and where sentence-based chunking doesn't.

### Configuration

- `SAMPLE_DOCUMENT` — the text being chunked; swap in your own to see how the strategies handle it
- `chunk_size` / `max_chunk_size` / `overlap` — passed as arguments to each strategy function; tune per use case

### See also

- `../rag/README.md` and `../hybrid_rag/README.md` — what happens to chunks after this step: they get embedded and indexed for retrieval
