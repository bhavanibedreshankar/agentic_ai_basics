# memory_management

Giving an agent both short-term memory (the current conversation) and long-term memory (facts that persist across separate runs of the program), and keeping both from growing unbounded.

## basic_agentic_memory.py

A personal assistant that remembers facts about you across separate runs of the program. Long-term memories persist to `memory.json`; type `exit` to end the conversation (memories are kept — only the in-session chat history is lost).

### Concepts covered

- **Short-term memory** — the `messages` list, same as `../../agentic_loop/basic_agentic_loop.py`. Exists only for the current run.
- **Long-term memory** — facts persisted to `memory.json` via a `save_memory` tool. Survives between runs: close the program, reopen it, and Claude still "remembers" you.
- **Context injection** — saved memories are read at startup and injected into the system prompt (`build_system_prompt`). This is *how* long-term memory actually reaches the model — Claude has no memory of its own between API calls; everything is fed back in as text.
- **Sliding-window trimming** — `trim_history` bounds the in-session conversation to the most recent `MAX_HISTORY_TURNS` messages, so token cost and latency don't grow unbounded as a chat gets longer. (For a more surgical pruning technique, see `../../context_management/pruning.py`.)
- **Measuring context size** — `print_context_size` uses the `count_tokens` endpoint to show the real token cost of the conversation after each turn, instead of guessing from message or character counts.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/memory_management/basic_agentic_memory.py
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

Edit the constants at the top of `basic_agentic_memory.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `MEMORY_FILE` — where long-term memories are persisted (default: `memory.json` next to the script)
- `MAX_HISTORY_TURNS` — how many recent messages to keep before trimming (default: `12`)

### See also

- `../../context_management/` — deeper dives into pruning, summarization, and retrieval as distinct context management strategies
- `../../token_tracking/basic_token_tracking.py` — a closer look at measuring and reasoning about token usage
- `../semantic_memory/` — this template's flat, append-only fact list restructured into categorized, updatable key-value memory
- `../` — the other five memory types in this directory, including a proper comparison table
