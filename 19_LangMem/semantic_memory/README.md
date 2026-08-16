# semantic_memory

Durable facts about a user/entity, extracted from conversation into a structured, schema-validated store.

## semantic_memory.py

A support agent that extracts durable facts about each customer (preferences, role, product) into a per-customer store, reusing the customer/support domain from [`../../17_LangChain/prompt_templates/`](../../17_LangChain/prompt_templates/README.md) and [`../../17_LangChain/memory/`](../../17_LangChain/memory/README.md). Type `exit` to end the session.

### Concepts covered

- **`UserFact`** — a Pydantic schema defining what a "fact" looks like; `create_memory_store_manager` extracts validated instances of it, not free-text notes.
- **`create_memory_store_manager(llm, schemas=[UserFact], namespace=..., store=...)`** — given conversation messages, decides on its own which facts are worth keeping and writes them into a `BaseStore`, replacing the hand-written `save_fact` tool + JSON file in [`../../05_Memory/semantic_memory/README.md`](../../05_Memory/semantic_memory/README.md).
- **`namespace=("memories", "{customer_id}")`** — a templated namespace resolved per call from `config={"configurable": {"customer_id": ...}}`, giving every customer an isolated fact store from ONE manager instance — the same per-session isolation idea as `../../17_LangChain/memory/memory.py`'s `session_id`.
- **`store.search(...)`** — reads facts back for a given customer, independent of any in-memory conversation state.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langchain langgraph langmem
export ANTHROPIC_API_KEY=your-key-here
python3 19_LangMem/semantic_memory/semantic_memory.py
```

Try:

```
Customer id (or 'exit'): alex
[alex] Message: I always prefer email, never call me
[alex] Facts on file: ['Prefers email over phone calls.']
```

### Configuration

- `MODEL` — see [`../../01_Core_Architecture/basics/README.md`](../../01_Core_Architecture/basics/README.md)
- `INSTRUCTIONS` — the extraction guidance passed to `create_memory_store_manager`; edit to change what counts as worth remembering

### See also

- [`../../05_Memory/semantic_memory/README.md`](../../05_Memory/semantic_memory/README.md) — the same concept, hand-built with a raw tool call and a JSON file
- [`../episodic_memory/README.md`](../episodic_memory/README.md) — the same extraction mechanism aimed at a different schema (whole past interactions, not standalone facts)
