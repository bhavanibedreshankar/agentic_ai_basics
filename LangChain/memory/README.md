# memory

Persisting conversation state across turns so a chain/agent can hold context.

## memory.py

A multi-session support chat: several customers can be served concurrently in the same running process, each with an isolated history keyed by `session_id`. Type `exit` to end the session.

### Concepts covered

- **`RunnableWithMessageHistory`** — wraps a chain so a session's prior messages are fetched, injected into the prompt, and appended back automatically on every call, with no state held by the chain itself.
- **`MessagesPlaceholder("history")`** — a template slot filled with a *list* of prior messages rather than a single string.
- **`InMemoryChatMessageHistory`** / **`get_session_history`** — the per-session store; swapping this for a database-backed implementation is the only change needed to persist across process restarts.
- **Session isolation** — `session_id` lives in `config`, not the input dict, so it's routing metadata rather than something sent to the model.
- Contrast with [`../../Memory/in_context_memory/in_context_memory.py`](../../Memory/in_context_memory/README.md), which has exactly one `messages` list for the whole program; this template supports many independent conversations at once.
- Honesty note: `RunnableWithMessageHistory` is deprecated in LangChain 1.x in favor of a LangGraph checkpointer (see `../../LangGraph/persistence_and_checkpointing/`) — kept here because it isolates the memory mechanic on its own, without a graph or agent loop around it.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/memory/memory.py
```

Try:

```
Session id (or 'exit'): alex
[alex] You: My order number is 4471
[alex] Claude: Got it, I've noted order 4471.

Session id (or 'exit'): jamie
[jamie] You: What's my order number?
[jamie] Claude: I don't have an order number on file for this session yet.

Session id (or 'exit'): alex
[alex] You: Remind me what my order number was
[alex] Claude: It's 4471, as you mentioned earlier.
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `_SESSION_STORE` — in-memory only; replace with a persistent `BaseChatMessageHistory` for real deployments

### See also

- [`../../Memory/README.md`](../../Memory/README.md) — six other kinds of agent memory, none LangChain-specific
- [`../../LangGraph/persistence_and_checkpointing/README.md`](../../LangGraph/persistence_and_checkpointing/README.md) — the currently-recommended replacement mechanism
