# Memory

Six distinct kinds of memory an agent can have, and what actually distinguishes one from another — scope (task vs. session vs. forever), persistence (in-process vs. on-disk), structure (flat list vs. categorized key-value vs. event log), and who manages it (the host program vs. the model itself via tools).

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`in_context_memory/`](in_context_memory/README.md) | The baseline: memory that's just the `messages` list — ephemeral and token-limited, with a tiny artificial limit so eviction is actually observable |
| 2 | [`working_memory/`](working_memory/README.md) | A task-scoped scratchpad the model manages itself via tools, cleared between tasks |
| 3 | [`episodic_memory/`](episodic_memory/README.md) | An automatically-logged history of past interactions, recalled by the model when a new request resembles an old one |
| 4 | [`memory_management/`](memory_management/README.md) | The simplest possible long-term memory: a flat, append-only list of facts, always injected wholesale into the system prompt |
| 5 | [`semantic_memory/`](semantic_memory/README.md) | `memory_management/`'s facts, restructured: categorized, keyed, and updatable rather than a flat growing list |
| 6 | [`external_memory/`](external_memory/README.md) | Persistent facts retrieved by semantic similarity, the shape of a real vector database integration (Pinecone, Chroma, Weaviate) |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Memory/semantic_memory/semantic_memory.py
```

## How these relate to each other

| | Scope | Persists to disk? | Structure | Who manages it |
|---|---|---|---|---|
| `in_context_memory/` | Current conversation | No | Flat message list | The host program (evicts oldest turns) |
| `working_memory/` | Current task | No | Flat key-value dict | The model (via `write_scratchpad`/`read_scratchpad`) |
| `episodic_memory/` | Forever | Yes | Timestamped event log | The host program logs; the model recalls |
| `memory_management/` | Forever | Yes | Flat, append-only list | The model (via `save_memory`), injected wholesale |
| `semantic_memory/` | Forever | Yes | Categorized, updatable key-value | The model (via `remember`/`forget`/`recall`) |
| `external_memory/` | Forever | Yes | Vector-embedded records | The model (via `save_memory`/`search_memory`) |

`../RAG_and_Knowledge/embedding/` is what `external_memory/`'s embedding mechanic is built on, applied there to a fixed read-only knowledge base instead of a writable store.
