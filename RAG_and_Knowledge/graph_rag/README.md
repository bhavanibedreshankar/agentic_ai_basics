# graph_rag

Graph RAG — retrieving from a graph of a codebase's actual structure (modules, classes, functions, and the calls/contains/inherits/imports edges between them) instead of ranking flat text chunks by similarity.

## graph_construction.py

Builds a queryable graph "database" out of a small in-memory toy codebase (a four-file order-processing mini-app) using Python's `ast` module, prints a summary, and persists it to `code_graph.json`. This is the "how is the database created" half of Graph RAG; run it first if you want the other two scripts to load a persisted graph instead of rebuilding one in memory.

### Concepts covered

- **Nodes and edges instead of chunks** — a module/class/function becomes a node; a `contains`/`inherits`/`imports`/`calls` relationship becomes an edge. That structure is exactly what flat chunking (`../chunking/chunking_strategies.py`) throws away — chunking only knows "these characters are near each other," never "this function calls that one."
- **`_FileVisitor`** — an `ast.NodeVisitor` that walks one file's AST, adding nodes and same-file `contains` edges directly, and queuing anything cross-file (imports, base classes, calls) into `pending` for later resolution.
- **Name-based call-edge resolution** — `pending` calls are resolved by matching the callee's bare name against every function in the graph, not by real type inference. The `visit_FunctionDef` CONCEPT comment spells out exactly what this trades away (a real code-intelligence tool like Sourcegraph or CodeQL resolves `order.total()` via `order`'s inferred type, so it never confuses `Order.total` with an unrelated `total()` elsewhere).
- **`build_code_graph`** — parses every file, then calls `_resolve_pending` once all node ids are known, since a call/import/inheritance target might be defined in a file visited later.
- **`code_graph.json`** — the persisted graph, JSON-serializable, that `graph_retrieval.py` and `graph_rag_agent.py` prefer to load over rebuilding from scratch (the realistic "build once, query many times" shape of a real graph database).

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 RAG_and_Knowledge/graph_rag/graph_construction.py
```

No `ANTHROPIC_API_KEY` needed — this script is pure static analysis, no API calls.

```
Built a code graph from 4 source files.

18 nodes: 4 module, 3 class, 11 function
31 edges: 14 contains, 1 inherits, 13 calls, 3 imports
...
models.py::PriorityOrder --inherits--> models.py::Order
orders.py::create_order --calls--> notifications.py::notify_order_confirmed
```

### Configuration

- `SOURCE_FILES` — the toy codebase, defined as strings rather than real files on disk so the template needs no filesystem setup
- `GRAPH_FILE` — where the graph is persisted (`code_graph.json`, gitignored — a runtime artifact, not source)

### See also

- `./graph_retrieval.py` — the "how is it queried" half of this same graph
- `../chunking/chunking_strategies.py` — the flat-text alternative this template's graph structure is contrasted against

## graph_retrieval.py

A fixed retrieval pipeline: keyword-match a couple of SEED nodes from the query, then traverse the code graph outward along its real edges for a bounded number of hops, and hand the resulting connected subgraph to Claude as context. Type `exit` to end the conversation.

### Concepts covered

- **Seed-then-traverse vs. independently-scored top-k** — `../rag/basic_rag.py` and `../coarse_to_fine_retrieval/coarse_to_fine_retrieval.py` rank each chunk independently by similarity. Here, a node can end up in the retrieved context because it's structurally *connected* to something relevant, even if it shares no words with the query at all.
- **`find_seed_nodes`** — the same keyword-overlap scoring idea as `../../Task_and_State_Management/context_management/retrieval.py`'s `search_notes`, just scored against a graph node's name/qualname/docstring instead of a flat document.
- **`retrieve_subgraph`** — a breadth-first walk outward from the seeds, bounded by `max_hops` and `max_nodes` for the same reason vector search caps `top_k`: without a budget, "follow every edge" on a real codebase would pull in most of the graph.
- **The headline example** — the query *"how is an order confirmed"* keyword-matches `notify_order_confirmed` (shares "confirmed"). Flat retrieval would stop there; this template follows its `calls` edge one more hop and pulls in `send_email` too, a function whose name and docstring share zero words with the query.
- **`load_or_build_graph`** — prefers the `code_graph.json` persisted by `graph_construction.py` if present, falls back to building the same graph in memory so this file still runs completely on its own.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/graph_rag/graph_retrieval.py
```

Try the query the docstring calls out:

```
You: how is an order confirmed

  [seeds: ['models.py::PriorityOrder', 'notifications.py::notify_order_confirmed']]
  [retrieved 8 of 18 nodes, 11 edges]

Claude: An order is confirmed by notify_order_confirmed, which sends a
confirmation email via send_email...
```

Notice `send_email` shows up in the retrieved node count even though it never appears in the query — it arrived via a 2-hop `calls` edge from the seed.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `retrieve_subgraph`'s `seed_top_n` / `max_hops` / `max_nodes` — how many seeds to start from and how far/wide the traversal is allowed to go
- `retrieve_subgraph`'s `edge_types` — restrict traversal to specific relationship types (e.g. only `calls`) for a given question

### See also

- `./graph_construction.py` — builds the graph this file traverses
- `./graph_rag_agent.py` — the same graph, but queried via Claude-driven tool calls instead of a fixed pipeline
- `../../Task_and_State_Management/context_management/retrieval.py` — the flat-text `search_notes` this template's seed-matching is modeled on

## graph_rag_agent.py

Instead of a fixed pipeline that always seed-matches-then-traverses on every question, this gives Claude two tools onto the same code graph and lets it decide, per question, whether and how to query. Type `exit` to end the conversation.

### Concepts covered

- **Fixed pipeline vs. tool-calling**, the same contrast already drawn between `../rag/basic_rag.py` and `../../Task_and_State_Management/context_management/retrieval.py`'s `search_notes` tool — applied here to a graph instead of flat documents.
- **`search_code_graph`** — the fuzzy tool: same seed-then-traverse mechanic as `./graph_retrieval.py`, for open-ended questions like "how does order confirmation work?"
- **`trace_call_chain`** — the precise tool: follows only `calls` edges from a named function, in one direction (`callees` or `callers`), with no keyword matching. Nothing analogous exists for flat text, since flat text has no notion of "callers" to trace.
- **Honest ambiguity, not silent guessing** — when a bare function name matches more than one node (e.g. `total` matches both `Order.total` and `PriorityOrder.total`), `trace_call_chain` reports the chain for *every* match rather than silently picking one, surfacing the same name-based ambiguity `graph_construction.py` warns about when building `calls` edges in the first place.
- **Claude picks the tool itself** — based only on each tool's `description` in `TOOLS`, which is what makes this agentic rather than a fixed pipeline. Try asking a structural question ("what calls charge_card?") vs. an open-ended one and compare which tool gets called.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 RAG_and_Knowledge/graph_rag/graph_rag_agent.py
```

Try both tools in the same session:

```
You: what calls charge_card?

  [tool] trace_call_chain({'function_name': 'charge_card', 'direction': 'callers'})

Claude: charge_card is called by create_order.

You: what does create_order do end to end?

  [tool] search_code_graph({'query': 'what does create_order do end to end'})

Claude: create_order builds an Order (or PriorityOrder), charges the card
via charge_card, and sends a confirmation email via notify_order_confirmed...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `TOOLS` — the two tool schemas Claude chooses between; edit each `description` to see how it changes Claude's tool choice
- `trace_call_chain`'s `direction` (`callees`/`callers`) and `max_hops` — how far and which way to follow `calls` edges

### See also

- `./graph_retrieval.py` — the same graph queried via a fixed pipeline instead of Claude-driven tool calls
- `../../Core_Architecture/tool_use/basic_agentic_tools.py` — the tool-calling loop shape (`TOOLS` + `execute_tool` + `run_turn`) this template follows
