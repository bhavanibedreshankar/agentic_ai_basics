# code_interpreter

Code Interpreter — a sandboxed Python/shell execution environment the agent can use to run code, hosted entirely by Anthropic.

## code_interpreter.py

A data analysis assistant that runs real Python (via the native `code_execution_20260521` server tool) to answer questions exactly rather than estimating. Type `exit` to end the conversation.

### Concepts covered

- **Server-side execution, not client-side** — every custom tool elsewhere in this repo (`../../Core_Architecture/tool_use/`, `../../Agent_Frameworks_and_Patterns/tool_registry/`) is client-executed: Claude requests a call, *your code* runs it, you send a result back. Code execution is the opposite: declare the tool and Claude runs the code itself, in an Anthropic-hosted container. There's no `execute_tool()` function in this file at all — that's the actual mechanic, not a simplification.
- **Reading the response's extra block types** — `print_response_content` distinguishes `server_tool_use` (the code Claude wrote) from `bash_code_execution_tool_result` (its stdout/stderr), so both the code and its output are visible, not just Claude's final summary.
- **Container reuse** — `run_turn` passes the response's `container` id back on the next call, so a later question in the same session can reference a variable or file a previous turn's code created, instead of starting a fresh sandbox every time.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Tools_and_Actions/code_interpreter/code_interpreter.py
```

Try:

```
You: Calculate the standard deviation of [4, 8, 15, 16, 23, 42] and tell me which values are more than 1 std dev from the mean.

  [running code]
import statistics
data = [4, 8, 15, 16, 23, 42]
...
  [stdout] ...

Claude: The standard deviation is approximately 12.9...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `TOOLS` — just the one server-side tool declaration; no schema to write

### See also

- `../web_search/README.md` — the other native server-side tool, same "no execute_tool()" shape
- `../../Core_Architecture/tool_use/README.md` — the client-side tool-calling loop this template deliberately doesn't need
