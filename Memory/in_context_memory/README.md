# in_context_memory

In-context memory — information held within the active context window: ephemeral (gone when the process exits) and limited by token size.

## in_context_memory.py

A chat agent with a deliberately tiny artificial context limit (500 tokens, vs. the ~1M tokens real context windows actually have), so you can watch eviction happen after just a few short turns. Type `exit` to quit.

### Concepts covered

- **Ephemeral** — nothing here is written to disk. This is the memory every chat template in the repo (starting with `../../Execution_Loops/agentic_loop/`) has been relying on; this template makes explicit what "ephemeral" actually means by never persisting anything at all.
- **Limited by token size** — `count_tokens` (the real endpoint, same as `../../token_tracking/`) measures the conversation before every call against `MAX_CONTEXT_TOKENS`.
- **`evict_oldest_pair`** — the simplest possible eviction policy: drop the oldest user/assistant pair, unconditionally, once the limit is exceeded. Deliberately dumber than `../../Task_and_State_Management/context_management/pruning.py` (which prunes selectively) or `../../Task_and_State_Management/context_management/summarization.py` (which compresses rather than discards) — this template exists to show the raw problem those two solve, not to solve it well itself.
- **Genuine, unrecoverable forgetting** — ask about something early on, keep chatting past the eviction point, then ask about it again: the agent has no way to recover it. There's no fallback here, unlike every other template in `../`.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/in_context_memory/in_context_memory.py
```

Try mentioning something early, then asking enough follow-up questions to push past the 500-token limit, then asking about the first thing again:

```
You: My favorite programming language is Rust.
  [context: 42/500 tokens, 2 messages]
...
  [evicted oldest turn to stay under 500 tokens: "My favorite programming language is Rust..."]
You: What's my favorite programming language?
Claude: I don't have that information...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `MAX_CONTEXT_TOKENS` — the artificial limit (default: `500`); raise it to see eviction happen less often, or lower it to see it happen almost immediately

### See also

- `../../Task_and_State_Management/context_management/README.md` — smarter eviction strategies (pruning, summarization) that avoid this template's total, unrecoverable loss
- `../working_memory/README.md`, `../episodic_memory/README.md`, `../semantic_memory/README.md`, `../external_memory/README.md` — the four memory types in this directory that exist specifically to survive past what in-context memory alone can hold
