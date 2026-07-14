# agents_and_tools

An LLM-driven loop that selects and invokes external tools to accomplish a task.

## agents_and_tools.py

An expense reimbursement assistant with three tools (`submit_expense`, `list_expenses`, `approve_expense`), run through LangChain's prebuilt agent instead of a hand-written tool-calling loop. Type `exit` to end the conversation.

### Concepts covered

- **`@tool`** — derives a tool's name, JSON Schema, and description from a Python function's type hints and docstring, replacing hand-written `input_schema` dicts.
- **`create_agent(llm, tools, system_prompt=...)`** — builds the entire request → tool-call → tool-result → request loop internally (on top of a LangGraph state graph) and returns a single invokable agent.
- **`run_turn`** — invokes the agent once per turn and reads the resulting `messages` list back to print every tool call and result, proving the loop ran even though this file never writes it explicitly.
- Contrast with [`../../Core_Architecture/tool_use/basic_agentic_tools.py`](../../Core_Architecture/tool_use/README.md), which hand-writes the identical loop: a `while True`, a `stop_reason == "tool_use"` check, and manual dispatch by tool name.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/agents_and_tools/agents_and_tools.py
```

Try:

```
You: Submit a $42.50 taxi expense, then approve it

  [tool] submit_expense({'description': 'Taxi', 'amount': 42.5})
  [result] Submitted expense a1b2c3d4: Taxi ($42.50), status=pending
  [tool] approve_expense({'expense_id': 'a1b2c3d4'})
  [result] Approved expense a1b2c3d4.

Claude: Submitted your $42.50 taxi expense and approved it right away.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `SYSTEM_PROMPT` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `TOOLS` — add more `@tool`-decorated functions to extend what the agent can do

### See also

- [`../../Core_Architecture/tool_use/README.md`](../../Core_Architecture/tool_use/README.md) — the same tool-calling loop, written by hand
- [`../../LangGraph/README.md`](../../LangGraph/README.md) — the graph layer `create_agent` is built on, explored directly
