# conditional_routing

Branching AND cycling: `add_conditional_edges` used to loop a node back on itself.

## conditional_routing.py

Revises a one-paragraph product pitch until it passes a plain-code validator (at least 3 sentences, mentions the product name), capped at `MAX_ATTEMPTS` so a stuck pipeline can't loop forever. Type `exit` to end the session.

### Concepts covered

- **Cycles** — `add_conditional_edges("check_draft", route_after_check, {"write_draft": "write_draft", "end": END})` lets `check_draft` route back to `write_draft`, a node that already ran — something no LCEL chain (a fixed DAG built at compose time) can express.
- **`route_after_check`** — a plain function of state returning the next node's name; doing both jobs at once (looping on failure, stopping at the cap) as graph structure instead of a `while` loop.
- Contrast with [`../../Execution_Loops/max_iterations/README.md`](../../Execution_Loops/max_iterations/README.md) (a hand-rolled iteration cap) and [`../../Execution_Loops/interrupts_breakpoints/README.md`](../../Execution_Loops/interrupts_breakpoints/README.md) (stops early on a condition) — this template's cycle does both at once.
- **`check_draft`** — a plain-code gate with no model call, the same spirit as [`../../Agent_Frameworks_and_Patterns/prompt_chaining/README.md`](../../Agent_Frameworks_and_Patterns/prompt_chaining/README.md)'s `validate_outline`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/conditional_routing/conditional_routing.py
```

Try:

```
Product (or 'exit'): NoiseAway headphones

[2 attempt(s), passed validation]
NoiseAway headphones block outside noise instantly. They're comfortable for all-day wear. You'll never want to take them off.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `MAX_ATTEMPTS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md); `MAX_ATTEMPTS` caps the cycle

### See also

- [`../state_graph_basics/README.md`](../state_graph_basics/README.md) — the state-merging mechanic this template's `attempts` counter and `feedback` field build on
- [`../../Execution_Loops/max_iterations/README.md`](../../Execution_Loops/max_iterations/README.md) — the same safety cap, hand-rolled as a `while` loop instead of graph structure
