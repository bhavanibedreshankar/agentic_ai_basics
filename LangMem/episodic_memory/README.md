# episodic_memory

Recalling a specific past interaction relevant to the current situation, not just an extracted fact.

## episodic_memory.py

A support agent that recalls the closest-matching past episode for a customer before responding to a new report, reusing the customer domain from [`../semantic_memory/`](../semantic_memory/README.md). Type `exit` to end the session.

### Concepts covered

- **`Episode(situation, resolution)`** — a Pydantic schema capturing a whole past interaction, not a standalone fact; the same `create_memory_store_manager` mechanism as `../semantic_memory/`, aimed at a different schema.
- **`IndexConfig(fields=["content.situation"])`** — indexes only the situation half of each stored episode, so a new ticket's text matches past SITUATIONS rather than past resolutions.
- **`store.search(..., query=...)`** — LangGraph's semantic search over the store's indexed field, ranking past episodes by similarity to the current situation rather than exact key lookup.
- Reuses **`HashEmbeddings`** from [`../../LangChain/retrieval_augmented_generation/retrieval_augmented_generation.py`](../../LangChain/retrieval_augmented_generation/README.md) as the store's embedding function.
- Contrast with [`../../Memory/episodic_memory/README.md`](../../Memory/episodic_memory/README.md), which hand-rolls a JSON file of episodes and a bag-of-words scoring tool the agent calls itself.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langchain langgraph langmem
export ANTHROPIC_API_KEY=your-key-here
python3 LangMem/episodic_memory/episodic_memory.py
```

Try:

```
Customer id (or 'exit'): alex
(c)lose a past ticket or (n)ew ticket? c
Text: Customer said they were charged twice; we refunded the duplicate.
  [episode recorded]

Customer id (or 'exit'): alex
(c)lose a past ticket or (n)ew ticket? n
Text: I was charged twice again this month
  [similar past episode found] situation: 'Duplicate charge on invoice'
  [it was resolved by] Refunded the duplicate charge
```

### Configuration

- `MODEL` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `IndexConfig["dims"]` in `build_store` — vector size for the hash-based embedding

### See also

- [`../semantic_memory/README.md`](../semantic_memory/README.md) — the same extraction mechanism, aimed at standalone facts instead of whole episodes
- [`../../Memory/episodic_memory/README.md`](../../Memory/episodic_memory/README.md) — the hand-built version of this exact concept
