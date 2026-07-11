# Agentic AI Basics

Basic Python templates for building with the Claude API.

## basic.py

Prompts you for a description of Python code you want, sends it to Claude, and prints the generated code.

### Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

Don't have a key? Create one at [platform.claude.com](https://platform.claude.com).

### Run

```bash
python3 basic.py
```

You'll be prompted:

```
What Python code would you like me to generate?
>
```

Type a description (e.g. `a function that reverses a string`) and press Enter. Claude's generated code prints to the terminal.

### Configuration

Edit the constants at the top of `basic.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape how Claude generates code

## basic_agentic_loop.py

A continuous chat agent. Keeps asking you questions and answering on any topic until you type `exit`. Maintains full conversation history across turns.

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_agentic_loop.py
```

You'll see:

```
Chat with Claude. Type 'exit' to end the conversation.

You:
```

Type a message and press Enter to get a response. Keep chatting, or type `exit` to end the conversation.

### Configuration

Edit the constants at the top of `basic_agentic_loop.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant

## basic_agentic_tools.py

A task manager agent that uses tools (function calling) to add, list, and complete tasks. Tasks persist to `tasks.json` in this folder. Type `exit` to end the conversation.

This template showcases the core tool-use features used in agentic AI development:

- **JSON Schema tool definitions** — each tool declares its inputs (required/optional params, enums) so Claude knows how to call it
- **`tool_choice`** — defaults to `"auto"`, letting Claude decide when a tool is needed
- **Parallel tool calls** — if a request needs multiple tools at once (e.g. "add a task and show me my list"), all calls are executed and their results returned together
- **Tool result handling, including errors** — a failed tool call (e.g. completing a task ID that doesn't exist) is returned with `is_error: true` so Claude can react appropriately
- **The manual agentic loop** — the script loops on the API response's `stop_reason` until Claude stops requesting tools and gives a final answer
- **Stateful tools** — tools have real side effects (reading/writing `tasks.json`), not just pure functions

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_agentic_tools.py
```

Try prompts like:

```
You: Add a task to buy groceries, due tomorrow
You: What tasks do I have open?
You: Mark that task as done
```

### Configuration

Edit the constants at the top of `basic_agentic_tools.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- `TOOLS` — the JSON Schema tool definitions Claude can call
- `TASKS_FILE` — where tasks are persisted (default: `tasks.json` next to the script)

## basic_agentic_memory.py

A personal assistant that remembers facts about you across separate runs of the program. Long-term memories persist to `memory.json`; type `exit` to end the conversation (memories are kept — only the in-session chat history is lost).

This template showcases memory management and optimization techniques for agents:

- **Short-term memory** — the `messages` list, same as `basic_agentic_loop.py`. Exists only for the current run.
- **Long-term memory** — facts persisted to `memory.json` via a `save_memory` tool. Survives between runs: close the program, reopen it, and Claude still "remembers" you.
- **Context injection** — saved memories are read at startup and injected into the system prompt (`build_system_prompt`), which is how long-term memory actually reaches the model — Claude has no memory of its own between API calls.
- **Sliding-window trimming** — `trim_history` bounds the in-session conversation to the most recent `MAX_HISTORY_TURNS` messages, so token cost and latency don't grow unbounded as a chat gets longer.
- **Measuring context size** — `print_context_size` uses the `count_tokens` endpoint to show the real token cost of the conversation after each turn, instead of guessing from message or character counts.

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_agentic_memory.py
```

Try prompts like:

```
You: My name is Alex and I'm learning to build AI agents
You: exit
```

Then run it again — Claude will recall what it saved:

```
(Recalling 1 saved memories from past sessions)

You: What do you remember about me?
```

### Configuration

Edit the constants at the top of `basic_agentic_memory.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `MEMORY_FILE` — where long-term memories are persisted (default: `memory.json` next to the script)
- `MAX_HISTORY_TURNS` — how many recent messages to keep before trimming (default: `12`)

## basic_token_tracking.py

A chat agent (same shape as `basic_agentic_loop.py`) that prints token usage and estimated cost after every turn, plus a running session summary when you type `exit`.

This template showcases token tracking techniques for agents:

- **Reading usage from a response** — `response.usage.input_tokens` / `.output_tokens`, the ground truth for what a call actually cost.
- **Pre-flight token counting** — `preview_input_tokens` uses the `count_tokens` endpoint to check a request's size *before* sending it, at no generation cost.
- **Cumulative tracking** — the `SessionUsage` class accumulates totals across every turn, since a multi-turn chat resends its full history on every call (see `basic_agentic_loop.py`), so cost compounds as the conversation grows. This template doesn't trim history, so you can watch that growth happen — pair it with `basic_agentic_memory.py`'s `trim_history` if you want to bound it.
- **Cache fields** — `cache_creation_input_tokens` / `cache_read_input_tokens` are tracked too, with a comment on why they matter (cache reads are billed at a fraction of normal input cost) even though this template doesn't use prompt caching itself.
- **Cost estimation** — `estimate_cost` converts token counts into an approximate dollar figure using per-token pricing.

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_token_tracking.py
```

Each turn prints usage inline:

```
You: What's the capital of France?
  [pre-flight estimate: ~24 input tokens]
  [usage: 24 in / 8 out, ~$0.0002 this turn]

Claude: The capital of France is Paris.
```

Type `exit` to see the session summary:

```
--- Session summary ---
Turns:          3
Input tokens:   412
Output tokens:  187
Cache reads:    0 (billed ~10x cheaper than input)
Cache writes:   0
Estimated cost: $0.0040
```

### Configuration

Edit the constants at the top of `basic_token_tracking.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- `PRICE_PER_MILLION_INPUT` / `PRICE_PER_MILLION_OUTPUT` — approximate pricing used for cost estimates; check [platform.claude.com/pricing](https://platform.claude.com/pricing) for current rates

## basic_prompt_chaining.py

A blog post generator that breaks the task into a chain of three focused LLM calls — outline, draft, edit — with a programmatic validation gate between the outline and the draft. Type `exit` at the topic prompt to quit.

This template showcases the prompt chaining technique:

- **Decomposing a task into steps** — `generate_outline`, `write_draft`, and `edit_draft` are each a separate, stateless call with its own narrow system prompt, instead of one call trying to do everything at once.
- **Chaining outputs into inputs** — `write_draft` builds its prompt directly from `generate_outline`'s output; `edit_draft` consumes `write_draft`'s output. That data flow *is* the chain.
- **A programmatic gate** — `validate_outline` is plain Python (no LLM call) that checks the outline has enough sections before continuing, so the chain fails fast instead of wasting a draft call on a broken outline.
- **Fixed control flow, model-generated content** — `run_chain` calls the three steps in a hardcoded sequence decided by your code, unlike `basic_agentic_tools.py`'s agentic loop, where the *model* decides what happens next and how many steps it takes.
- **Narrow, single-purpose context per step** — unlike the chat templates, no shared conversation history accumulates across steps; each call only sees the specific input it needs for its one job.

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_prompt_chaining.py
```

Try a topic:

```
Topic: why sourdough bread takes so long to rise

[1/3] Generating outline...
- Introduction to sourdough fermentation
- The role of wild yeast and bacteria
- Temperature and its effect on rise time
- Tips for speeding up (or slowing down) fermentation
[gate] Outline has 4 sections — looks good.

[2/3] Writing draft from outline...
...

[3/3] Editing draft...

=== Final post ===
...
```

### Configuration

Edit the constants at the top of `basic_prompt_chaining.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length per step
- `EFFORT` — thinking/response depth (default: `medium`)
- `OUTLINE_SYSTEM_PROMPT` / `DRAFT_SYSTEM_PROMPT` / `EDIT_SYSTEM_PROMPT` — the focused instructions for each step
- `validate_outline` — the gate logic; edit the section-count threshold or add your own checks

## basic_tool_registry.py

A utility agent (calculator, word counter, text case converter, text reverser, clock) built around a **tool registry** — a single catalog that both advertises tools to Claude and dispatches calls to them, instead of maintaining that information in two separate places. Type `exit` to end the conversation.

This template showcases the skill/tool registry pattern:

- **One record per tool** — the `register_tool` decorator stores a tool's name, description, JSON Schema, and handler function together in `TOOL_REGISTRY`, right next to where the tool is defined. Compare this to `basic_agentic_tools.py`, which hand-maintains a `TOOLS` list and a separate `execute_tool` if/elif chain — two places that can drift out of sync.
- **Catalog-driven tool definitions** — `build_tool_catalog()` generates the `tools` list sent to the API directly from the registry, so a newly registered tool appears automatically with no other code to touch.
- **Registry-driven dispatch** — `execute_tool()` looks up the handler by name (a dict lookup), instead of growing an if/elif chain with every new tool.
- **A note on scale** — with a large registry, sending every tool's full schema on every request gets expensive; the comments point to Anthropic's tool search feature as the natural next step, letting Claude discover only the relevant tools from a much bigger catalog.
- **Safe tool implementation** — `calculate` parses expressions with `ast` instead of calling `eval()` directly, since tool input (even when it comes from the model rather than the user) shouldn't be trusted blindly.

### Setup

Same as above — install dependencies and set `ANTHROPIC_API_KEY`.

### Run

```bash
python3 basic_tool_registry.py
```

Try prompts like:

```
You: What's 12% of 340?
You: How many words are in "the quick brown fox jumps over the lazy dog"?
You: What time is it in Tokyo?
```

### Configuration

Edit the constants at the top of `basic_tool_registry.py` to change behavior:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- Add a new tool by writing a function and decorating it with `@register_tool(name=..., description=..., input_schema=...)` — no other code needs to change
