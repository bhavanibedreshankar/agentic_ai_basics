# human_in_the_loop

Pausing a graph mid-execution for a human decision via `interrupt()`, then resuming via `Command(resume=...)`.

## human_in_the_loop.py

The expense-approval scenario from [`../../LangChain/agents_and_tools/agents_and_tools.py`](../../LangChain/agents_and_tools/README.md)'s tool set — anything over `APPROVAL_THRESHOLD` pauses for a human decision instead of an agent freely calling `approve_expense`. Type `exit` to end the session.

### Concepts covered

- **`interrupt(payload)`** — called inside `request_approval`, this raises a signal that unwinds the graph run and returns control to the caller, with `payload` attached under `result["__interrupt__"]`; the graph isn't blocked on anything, it's stopped, with its state saved by the checkpointer.
- **`Command(resume=decision)`** — a separate, later `.invoke()` call against the same `thread_id` that continues execution from exactly where `interrupt()` paused, with `decision` becoming that call's return value.
- **Requires a checkpointer** — see [`../persistence_and_checkpointing/README.md`](../persistence_and_checkpointing/README.md); the paused state has to be saved somewhere for the resume call to find it.
- Contrast with [`../../Execution_Loops/human_in_the_loop/README.md`](../../Execution_Loops/human_in_the_loop/README.md), where approval is a plain `input()` call blocking the Python process itself — kill that process while it's waiting and the pending approval is gone; here the graph is genuinely paused and resumable, even from a different process (given a durable checkpointer).

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/human_in_the_loop/human_in_the_loop.py
```

Try:

```
Expense description (or 'exit'): New laptop
Amount: $1500

  [paused for approval] New laptop — $1500.00
  Approve? (yes/no): yes

Approved: New laptop ($1500.00)
```

### Configuration

- `APPROVAL_THRESHOLD` — expenses at or below this amount auto-approve with no interrupt at all

### See also

- [`../persistence_and_checkpointing/README.md`](../persistence_and_checkpointing/README.md) — the checkpointer this template depends on
- [`../../Execution_Loops/human_in_the_loop/README.md`](../../Execution_Loops/human_in_the_loop/README.md) — the same concept, hand-rolled as a blocking `input()` call
