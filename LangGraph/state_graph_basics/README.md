# state_graph_basics

One level deeper than the LangChain topic's lightweight intro: how LangGraph merges partial state updates from multiple nodes.

## state_graph_basics.py

A 3-node order fulfillment pipeline (validate, charge, draft confirmation) where every node appends one line to a shared, accumulating audit log. Type `exit` to end the session.

### Concepts covered

- **`START` / `END`** — the singleton markers every graph begins and ends at; `add_edge(START, "validate_order")` is what `../../LangChain/langgraph_workflows/langgraph_workflows.py`'s `set_entry_point()` wraps.
- **`Annotated[list[str], operator.add]`** — a reducer: each node returns only the ONE new log line it wants to add, and LangGraph concatenates it onto the existing list automatically, rather than replacing it.
- Contrast with [`../../Memory/working_memory/README.md`](../../Memory/working_memory/README.md)'s scratchpad, where a plain Python dict a function mutates is what accumulates state — here the graph framework does the accumulating, driven by a type annotation.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/state_graph_basics/state_graph_basics.py
```

Try:

```
Order id (or 'exit'): A100
Item: widget

[audit log]
  - validate_order: 'widget' is in stock (12 left)
  - charge_payment: charged order A100
  - draft_confirmation: wrote customer confirmation

Your widget order A100 has shipped — thanks for your business!
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `_INVENTORY` — edit to change which items are in/out of stock

### See also

- [`../../LangChain/langgraph_workflows/README.md`](../../LangChain/langgraph_workflows/README.md) — the entry point for this topic, a simpler 3-node graph with no reducer
- [`../conditional_routing/README.md`](../conditional_routing/README.md) — the next primitive: edges that can cycle, not just merge state
