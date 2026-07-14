# langgraph_workflows

A lightweight intro to using LangGraph's graph-based orchestration from within LangChain.

## langgraph_workflows.py

A three-node ticket-triage graph: `classify` always runs first, then a conditional edge routes to either `auto_respond` (routine tickets) or `escalate` (urgent ones). Type `exit` to end the session.

### Concepts covered

- **`StateGraph(TicketState)`** — a `TypedDict`-backed shared state every node reads from and writes back to, unlike a chain's data flowing linearly through `|`.
- **Nodes as functions** — `(state) -> partial state update`; `classify`, `auto_respond`, and `escalate` each update only the keys they care about.
- **`add_conditional_edges`** — the destination node is computed at runtime from current state (`route_after_classify`), making branching a property of the graph's edges rather than a Runnable's internal logic.
- **`.compile()` / `.invoke()`** — turns the graph definition into a runnable object, the same "build once, invoke many times" shape as `chain.invoke()` elsewhere in this topic.
- Contrast with [`../chains/chains.py`](../chains/README.md)'s `RunnableBranch`, which also branches but keeps the pipeline shape a fixed DAG; a `StateGraph`'s edges can point back at a node that already ran (a cycle), which is what an agent loop needs — not used in this lightweight intro, but see `../../LangGraph/` for where that shows up.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/langgraph_workflows/langgraph_workflows.py
```

Try:

```
Ticket text: The entire production site is down for all customers
  [priority: urgent]

[escalated to a human agent] The entire production site is down for all customers
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `TicketState` — add fields to carry more data between nodes

### See also

- [`../../LangGraph/README.md`](../../LangGraph/README.md) — the full topic: cycles, persistence, human-in-the-loop, streaming, multi-agent subgraphs
- [`../chains/README.md`](../chains/README.md) — the LCEL-level equivalent of conditional routing, without a graph
