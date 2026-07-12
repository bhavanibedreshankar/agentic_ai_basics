# tool_registry

The skill/tool registry pattern — a single catalog of available tools, with descriptions, that both advertises tools to the LLM and dispatches calls to them.

## basic_tool_registry.py

A utility agent (calculator, word counter, text case converter, text reverser, clock) built around a `TOOL_REGISTRY` catalog, instead of the hand-maintained list + if/elif chain used in `../../Tools_and_Actions/tool_use/basic_agentic_tools.py`. Type `exit` to end the conversation.

### Concepts covered

- **One record per tool** — the `register_tool` decorator stores a tool's name, description, JSON Schema, and handler function together in `TOOL_REGISTRY`, right next to where the tool is defined. This avoids the "two places to update" bug where a tool's definition (what the LLM sees) drifts out of sync with its dispatch logic (what actually runs).
- **Catalog-driven tool definitions** — `build_tool_catalog()` generates the `tools` list sent to the API directly from the registry, so a newly registered tool appears automatically with no other code to touch.
- **Registry-driven dispatch** — `execute_tool()` looks up the handler by name (a dict lookup), instead of growing an if/elif chain with every new tool.
- **A note on scale** — with a large registry, sending every tool's full schema on every request gets expensive; comments in the file point to Anthropic's tool search feature as the natural next step, letting Claude discover only the relevant tools from a much bigger catalog.
- **Safe tool implementation** — `calculate` parses expressions with `ast` instead of calling `eval()` directly, since tool input (even when it comes from the model rather than the user) shouldn't be trusted blindly.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Tools_and_Actions/tool_registry/basic_tool_registry.py
```

Try prompts like:

```
You: What's 12% of 340?
You: How many words are in "the quick brown fox jumps over the lazy dog"?
You: What time is it in Tokyo?
```

### Configuration

Edit the constants at the top of `basic_tool_registry.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- Add a new tool by writing a function and decorating it with `@register_tool(name=..., description=..., input_schema=...)` — no other code needs to change

### See also

- `../../Tools_and_Actions/tool_use/basic_agentic_tools.py` — the simpler hand-maintained approach this pattern replaces
