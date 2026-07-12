# Tools_and_Actions

Seven kinds of tools an agent can use to affect the world beyond generating text — some run on Anthropic's servers, some run on yours, and the difference matters for how you build with them.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`tool_use/`](tool_use/README.md) | The foundational pattern: JSON Schema tool definitions, parallel calls, error handling, the client-side agentic tool-calling loop |
| 2 | [`tool_registry/`](tool_registry/README.md) | A catalog-driven alternative to `tool_use/`'s hand-maintained list — one registry drives both tool definitions and dispatch |
| 3 | [`code_interpreter/`](code_interpreter/README.md) | Claude's real, native server-side Python sandbox — no execute_tool() needed at all |
| 4 | [`web_search/`](web_search/README.md) | Claude's real, native live web search — same server-side shape as code execution |
| 5 | [`file_io_tools/`](file_io_tools/README.md) | The real Anthropic-defined text editor tool, client-executed, with a hard security boundary around a sandbox directory |
| 6 | [`api_connectors_mcp/`](api_connectors_mcp/README.md) | Wrapping a real external REST API as a custom tool, plus the MCP connector shape for comparison |
| 7 | [`browser_computer_use/`](browser_computer_use/README.md) | Autonomous UI control via structured page state, contrasted with Anthropic's pixel-based computer use tool |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Tools_and_Actions/code_interpreter/code_interpreter.py
```

## Server-side vs. client-side — the split that runs through this whole directory

| | Who executes it | execute_tool() needed? | Templates |
|---|---|---|---|
| **Server-side** | Anthropic's infrastructure | No — the tool's output just appears in the response | `code_interpreter/`, `web_search/` |
| **Client-side (Anthropic-defined)** | Your code, but the schema is built into the model | Yes | `file_io_tools/` |
| **Client-side (custom)** | Your code, and you define the schema too | Yes | `tool_use/`, `tool_registry/`, `api_connectors_mcp/`'s `get_weather`, `browser_computer_use/`'s page tools |

`tool_use/` and `tool_registry/` are the pattern every custom tool elsewhere in this repo follows too (`../Memory/external_memory/`, `../RAG_and_Knowledge/embedding/`, and so on) — this directory is where the *other* two categories, server-side and Anthropic-defined client-side, show up for the first time. `api_connectors_mcp/` also touches a fourth option in its reference-only closing section: MCP, where a *server you don't control* executes the tool, over the network, with no `execute_tool()` on your side either.
