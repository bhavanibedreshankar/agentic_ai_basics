# tool_use

Tool use (function calling) — giving Claude the ability to take actions, not just talk.

## basic_agentic_tools.py

A task manager agent that uses tools to add, list, and complete tasks. Tasks persist to `tasks.json` in this folder. Type `exit` to end the conversation.

### Concepts covered

- **JSON Schema tool definitions** — each tool declares its inputs (required/optional params, enums) so Claude knows how to call it.
- **`tool_choice`** — defaults to `"auto"`, letting Claude decide when a tool is needed. Alternatives: force a specific tool, force any tool, or disable tools entirely.
- **Parallel tool calls** — if a request needs multiple tools at once (e.g. "add a task and show me my list"), all calls are executed and their results returned together in a single message.
- **Tool result handling, including errors** — a failed tool call (e.g. completing a task ID that doesn't exist) is returned with `is_error: true` so Claude can react appropriately instead of the program crashing.
- **The manual agentic loop** — two nested loops: an *outer* loop (in `main()`) that keeps the conversation going across turns, and an *inner* loop (in `run_turn()`) that resolves any tool calls within a single turn before returning.
- **Stateful tools** — tools have real side effects (reading/writing `tasks.json`), not just pure functions.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 tool_use/basic_agentic_tools.py
```

Try prompts like:

```
You: Add a task to buy groceries, due tomorrow
You: What tasks do I have open?
You: Mark that task as done
```

### Configuration

Edit the constants at the top of `basic_agentic_tools.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- `TOOLS` — the JSON Schema tool definitions Claude can call
- `TASKS_FILE` — where tasks are persisted (default: `tasks.json` next to the script)

### See also

- `../agentic_loop/basic_agentic_loop.py` — the outer chat loop this template's `main()` follows
- `../tool_registry/basic_tool_registry.py` — a scalable alternative to this file's hand-maintained `TOOLS` list + if/elif dispatch
