# state_machine

State Machine — a formal structure defining an agent's valid states and transitions, preventing invalid actions before they happen.

## state_machine.py

An order-processing agent whose orders move through `pending → paid → shipped → delivered`, with a `cancelled` branch off `pending` or `paid`. Type `exit` to quit.

### Concepts covered

- **A single source of truth for legality** — `TRANSITIONS` is a plain `{state: {allowed next states}}` dict. Every legality check in the file reads from this one table; nothing hardcodes "you can't ship an unpaid order" anywhere else.
- **Enforcement happens before mutation, not after** — `transition_order` checks `order.can_transition_to(new_state)` *before* calling `order.transition_to()`. An illegal request never touches the order's actual state, not even briefly — verified in testing that a rejected transition leaves `order.state` completely unchanged.
- **Contrast with `../../Multi_Agent_Systems/agent_handoff/agent_handoff.py`** — that template also tracks a current-state string (`current_agent_name`), but *any* transfer tool can fire from *any* agent; there's no table of which transitions are even legal. This template is what handoff would need to add real state-legality enforcement.
- **Terminal states** — `delivered` and `cancelled` map to an empty set of allowed transitions, so *everything* attempted from them is rejected — verified directly, including that a `cancelled` attempt from `delivered` fails with a clear error rather than silently doing nothing.
- **Tool errors carry the rejection, not a crash** — an illegal transition returns a normal `tool_result` with `is_error: true` and a specific explanation of what *was* allowed, so Claude can recover and try a legal action instead of the conversation breaking.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Task_and_State_Management/state_machine/state_machine.py
```

Try:

```
You: Ship order A123.
  [tool] transition_order({'order_id': 'A123', 'new_state': 'shipped'})
  [result  (REJECTED)] Error: cannot transition order A123 from 'pending' to 'shipped'. Allowed from 'pending': ['cancelled', 'paid'].

Claude: Order A123 hasn't been paid for yet — I'll need to mark it paid first...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `TRANSITIONS` — the state machine definition itself; add a state or edge here and both tools automatically respect it

### See also

- `../../Multi_Agent_Systems/agent_handoff/README.md` — an unenforced current-state pattern this template's transition table formalizes
- `../checkpointing/README.md` — persisting progress *within* a state, rather than controlling which states are reachable
