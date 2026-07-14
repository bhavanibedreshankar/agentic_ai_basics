# LangMem

Long-term agent memory — durable across sessions, unlike [`../LangChain/memory/`](../LangChain/memory/README.md)'s per-session conversational history. Five ways an agent's memory can differ: what it captures (a standalone fact vs. a whole past interaction vs. its own instructions), who decides when it updates (surrounding code vs. the model itself), and when the update actually runs (blocking the current response vs. deferred to the background).

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`semantic_memory/`](semantic_memory/README.md) | Extracting durable FACTS from conversation into a schema-validated store (`create_memory_store_manager`) |
| 2 | [`episodic_memory/`](episodic_memory/README.md) | The same mechanism aimed at whole past INTERACTIONS, retrieved by semantic similarity to the current situation |
| 3 | [`procedural_memory/`](procedural_memory/README.md) | The agent's own INSTRUCTIONS improving from feedback, persisted across runs (`create_prompt_optimizer`) |
| 4 | [`memory_management_tools/`](memory_management_tools/README.md) | The MODEL deciding when to save/search memory, via tools, instead of code deciding after every turn |
| 5 | [`background_memory_consolidation/`](background_memory_consolidation/README.md) | Running extraction OFF the response path, debounced across a burst (`ReflectionExecutor`) |

## Setup

Same as the rest of the repo, plus the LangMem/LangGraph packages:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
pip install langchain langchain-core langchain-anthropic langgraph langmem pydantic
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 LangMem/semantic_memory/semantic_memory.py
```

## How these relate to each other

| | Captures | Who decides when it updates | Runs |
|---|---|---|---|
| `semantic_memory/` | A standalone fact | Code (after every turn) | Synchronously, blocking |
| `episodic_memory/` | A whole past interaction (situation + resolution) | Code (when a ticket closes) | Synchronously, blocking |
| `procedural_memory/` | The agent's own system prompt | Code (when feedback is given) | Synchronously, blocking |
| `memory_management_tools/` | A fact (same schema as `semantic_memory/`) | The MODEL, via tool calls | Synchronously, blocking |
| `background_memory_consolidation/` | A standalone fact (same schema again) | Code, but debounced | Asynchronously, off the response path |

`semantic_memory/` and `episodic_memory/` share one extraction mechanism (`create_memory_store_manager`) and differ only in schema and retrieval mode — a fact looked up by namespace vs. an episode ranked by similarity to the current situation. `procedural_memory/` targets a different kind of state entirely (the agent's own prompt, via `create_prompt_optimizer`) but is triggered the same code-decides, blocking way as the first two. `memory_management_tools/` takes `semantic_memory/`'s exact schema and hands the update DECISION to the model instead of the code. `background_memory_consolidation/` takes it back from the model but changes WHEN the update runs — deferred and debounced, so a burst of triggers costs one extraction instead of one per trigger.
