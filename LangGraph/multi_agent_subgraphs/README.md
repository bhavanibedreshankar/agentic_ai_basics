# multi_agent_subgraphs

Composing a fully compiled `StateGraph` as a node inside a larger graph.

## multi_agent_subgraphs.py

A support ticket router: `classify` picks a category, then a conditional edge hands off to whichever specialist subgraph matches (`billing` or `technical`), each internally multi-step. Type `exit` to end the session.

### Concepts covered

- **`graph.add_node("billing", billing_subgraph)`** — a compiled `StateGraph` added directly as a node; LangGraph invokes it exactly like a plain function node, because a compiled graph is itself invokable the same way.
- **Shared state schema** — `billing_subgraph` and `technical_subgraph` both operate on the same `TicketState` as the outer `router_graph`, so a specialist's nodes read and write the same state object with no translation layer at the boundary.
- HONESTY NOTE: an `Annotated[list[str], operator.add]` reducer (the pattern in [`../state_graph_basics/README.md`](../state_graph_basics/README.md)) was tried first for `steps_taken` here and DOUBLE-APPLIED the parent's most recent log entries across the subgraph boundary — verified directly with a minimal repro. `steps_taken` is accumulated manually instead (`state["steps_taken"] + [...]`) — see the module docstring for the exact reproduction and why.
- Contrast with [`../../Multi_Agent_Systems/orchestrator/README.md`](../../Multi_Agent_Systems/orchestrator/README.md), where each specialist is a TOOL — one opaque call out and back, with the specialist's internal steps invisible from the orchestrator's side. Here, both specialists' internal nodes are real graph structure, visible in `steps_taken`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/multi_agent_subgraphs/multi_agent_subgraphs.py
```

Try:

```
Ticket text: I was charged twice for my subscription this month
  [category: billing]
  [step] router: classified as billing
  [step] billing: verified account status
  [step] billing: drafted resolution

We've refunded the duplicate charge to your original payment method.
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `classify`'s keyword list — edit to change routing between `billing` and `technical`

### See also

- [`../../Multi_Agent_Systems/orchestrator/README.md`](../../Multi_Agent_Systems/orchestrator/README.md) — the same "route to a specialist" idea, with specialists as opaque tool calls instead of subgraphs
- [`../state_graph_basics/README.md`](../state_graph_basics/README.md) — the reducer mechanic that turns out NOT to be safe across the subgraph boundary this template hits
