# memory_management_tools

Giving the agent itself the ability to decide when to save or search its own memories, as ordinary tool calls.

## memory_management_tools.py

A support agent with `manage_memory`/`search_memory` tools scoped to a per-customer namespace, reusing the customer domain from [`../semantic_memory/`](../semantic_memory/README.md) and [`../episodic_memory/`](../episodic_memory/README.md). Type `exit` to end the session.

### Concepts covered

- **`create_manage_memory_tool(namespace, store=...)`** — a tool the model calls itself to create/update/delete a memory, bound on `create_agent` the same way [`../../LangChain/agents_and_tools/agents_and_tools.py`](../../LangChain/agents_and_tools/README.md) binds its expense tools.
- **`create_search_memory_tool(namespace, store=...)`** — a tool the model calls itself to semantically search memories before answering.
- **Who decides** — `../semantic_memory/` and `../episodic_memory/` both extract memory AFTER every turn, unconditionally, because the CODE decides. Here the MODEL decides, mid-conversation, whether a message is worth saving and whether a question needs a search first.
- **Namespace-scoped tool construction** — each `build_agent` call scopes both tools to one customer's namespace, so the model has no way to reach a different customer's memories even if it wanted to.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langchain langgraph langmem
export ANTHROPIC_API_KEY=your-key-here
python3 LangMem/memory_management_tools/memory_management_tools.py
```

Try:

```
Customer id (or 'exit'): alex
[alex] You: Please always email me, never call
  [tool] manage_memory({'content': 'Prefers email over phone.', 'action': 'create'})
  [result] created memory ...
Claude: Got it, I'll always email you.

Customer id (or 'exit'): alex
[alex] You: How should we contact you?
  [tool] search_memory({'query': 'contact preference'})
  [result] [{"value":{"content":"Prefers email over phone."}, ...}]
Claude: You told me before you prefer email over phone.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `SYSTEM_PROMPT` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)

### See also

- [`../../LangChain/agents_and_tools/README.md`](../../LangChain/agents_and_tools/README.md) — the same `create_agent` + tool-binding pattern, over ordinary business tools instead of memory
- [`../semantic_memory/README.md`](../semantic_memory/README.md) — the code-decides counterpart to this model-decides template
