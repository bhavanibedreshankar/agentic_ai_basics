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
