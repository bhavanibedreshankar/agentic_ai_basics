# context_management

Strategies for keeping the context window relevant as a conversation grows: pruning, summarization, and retrieval. Each file demonstrates one distinct technique.

All three scripts are set up the same way — install dependencies, set `ANTHROPIC_API_KEY`, run, and type `exit` to quit.

```bash
pip install -r ../../requirements.txt   # from this directory, or use the root requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

## pruning.py

A research assistant that repeatedly calls a (mocked) `search_web` tool. Once a tool result is superseded by a newer one, its raw content is replaced with a short placeholder — the surrounding conversation (questions, Claude's answers) is left untouched.

- **`prune_old_tool_results`** — selectively replaces stale `tool_result` blocks with a placeholder, in place. More surgical than `../../Memory/memory_management/basic_agentic_memory.py`'s `trim_history`, which drops whole old *turns* wholesale.
- Prints a before/after character count each time pruning runs, so the effect is visible.

```bash
python3 pruning.py
```

## summarization.py

A general chat agent that, once the conversation passes `SUMMARIZE_AFTER_TURNS` messages, uses a separate, focused API call to summarize everything except the most recent `KEEP_RECENT_TURNS` messages, then continues from the summary.

- **`summarize`** — a narrow, single-purpose LLM call (same idea as `../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py`'s steps) whose only job is compressing older history into a short paragraph.
- **`maybe_summarize`** — splices the summary in as a single leading message, collapsing many old messages into one.
- A hand-built version of the same idea behind Anthropic's built-in "compaction" feature — useful for understanding the mechanism, even though production systems would typically use the server-side feature instead of reimplementing it.

```bash
python3 summarization.py
```

## retrieval.py

A documentation assistant with a small local knowledge base (5 short notes on unrelated topics). Instead of injecting every note into the system prompt, Claude calls a `search_notes` tool and only the most relevant notes (by keyword overlap) are pulled into context for that turn.

- **`search_notes`** — ranks notes by how many words they share with the query and returns only the top matches. A dependency-free stand-in for the embeddings + vector database a real RAG system would use — same pattern, simpler math.
- Contrast with `../../Memory/memory_management/basic_agentic_memory.py`: memory always injects saved facts into every system prompt; retrieval pulls in only what's relevant to the *current* question, on demand.

```bash
python3 retrieval.py
```

## Configuration

Each file has its own constants at the top (`MODEL`, `MAX_TOKENS`, `EFFORT`, plus technique-specific knobs like `MAX_HISTORY_TURNS`-style thresholds, `SUMMARIZE_AFTER_TURNS`/`KEEP_RECENT_TURNS`, or `top_k`) — see the comments in each file for what they control.

## See also

- `../task_decomposition/README.md` — a different kind of state to manage: the shape of the work itself, not the conversation's context window
- `../checkpointing/README.md` — persisting a task's *progress* across a crash, versus these three templates' focus on keeping the live conversation lean
