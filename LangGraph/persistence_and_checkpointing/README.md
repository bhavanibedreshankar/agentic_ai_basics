# persistence_and_checkpointing

Attaching a `checkpointer` at compile time so a graph's full state survives across separate `.invoke()` calls.

## persistence_and_checkpointing.py

A running notes list per conversation thread: add a note, come back later with the same `thread_id` and the list is still there; switch `thread_id` and it's empty again. Type `exit` to end the session.

### Concepts covered

- **`graph.compile(checkpointer=InMemorySaver())`** — one keyword argument: every `.invoke()` against a given `thread_id` loads that thread's last saved state first, then saves the new state after.
- **Partial input across calls** — `add()` only ever passes `{"new_note": note}`; the checkpointer supplies the rest (`notes`) from the last checkpoint, and the `operator.add` reducer (same mechanic as [`../state_graph_basics/README.md`](../state_graph_basics/README.md)) appends to it.
- **Thread isolation** — a different `thread_id` starts from a blank slate, same guarantee [`../../LangChain/memory/README.md`](../../LangChain/memory/README.md) makes for `session_id`.
- This file is the concrete answer to `../../LangChain/memory/memory.py`'s HONESTY NOTE, which names checkpointing as the recommended replacement for `RunnableWithMessageHistory` — see that file's README for the "persists ONE thing vs. persists the ENTIRE state" distinction.
- Contrast with [`../../Task_and_State_Management/checkpointing/README.md`](../../Task_and_State_Management/checkpointing/README.md), which gets real cross-process durability today by writing its own JSON file by hand — `InMemorySaver` here doesn't survive a restart (see the module's HONESTY NOTE).

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/persistence_and_checkpointing/persistence_and_checkpointing.py
```

Try:

```
Thread id (or 'exit'): work
[work] New note: Finish the report
[work] All notes so far: ['Finish the report']

Thread id (or 'exit'): personal
[personal] New note: Buy milk
[personal] All notes so far: ['Buy milk']

Thread id (or 'exit'): work
[work] New note: Email the client
[work] All notes so far: ['Finish the report', 'Email the client']
```

### Configuration

- `InMemorySaver()` in `build_graph` — swap for a persistent backend (`langgraph-checkpoint-sqlite`, `-postgres`) for durability across restarts; nothing else in the file changes

### See also

- [`../../LangChain/memory/README.md`](../../LangChain/memory/README.md) — the deprecated mechanism this file replaces
- [`../human_in_the_loop/README.md`](../human_in_the_loop/README.md) — requires this exact checkpointing mechanism to pause and resume a graph
