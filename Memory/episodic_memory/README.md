# episodic_memory

Episodic memory — records of past interactions or task runs, used to inform future behavior. Memory about events, not facts.

## episodic_memory.py

A task assistant that automatically logs every completed interaction as a timestamped episode, and can recall similar past episodes before tackling a new request. Type `exit` to end the conversation — episodes persist in `episodes.json` and are available again next run.

### Concepts covered

- **Events, not facts** — the key distinction from `../semantic_memory/` and `../external_memory/`: an episode is "the user asked me to summarize a PDF on this date, and here's what I did", not a timeless statement like "the user prefers concise summaries". This template deliberately does *not* extract facts from episodes — that would cross into semantic memory territory, out of scope here.
- **Automatic logging, deliberate recall** — `log_episode` runs unconditionally after every completed turn (the program's job, not a choice); `recall_episodes` is a tool the *model* chooses to use when a new request seems similar to something handled before.
- **A different retrieval technique on purpose** — `recall_episodes` uses simple keyword overlap rather than `../external_memory/`'s embeddings, showing that episodic recall (matching a new task description against old ones) often doesn't need the heavier machinery — task descriptions tend to share literal words.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/episodic_memory/episodic_memory.py
```

Try:

```
You: Summarize this PDF report for me.
Claude: [does the task]
  [logged episode at 2026-07-11T23:59:23+00:00]

You: Can you summarize another PDF document?
  [tool] recall_episodes({'query': 'summarize a PDF document'})
  [result] [2026-07-11T23:59:23+00:00] Request: Summarize this PDF report for me. ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `EPISODES_FILE` — where episodes are persisted (default: `episodes.json` next to the script)
- `recall_episodes`'s `top_k` — how many past episodes are surfaced per query

### See also

- `../semantic_memory/README.md` — the kind of durable fact you might *extract* from repeated episodes (not done automatically here)
- `../../Task_and_State_Management/context_management/retrieval.py` — the same keyword-overlap technique this template's recall uses, applied there to a fixed knowledge base instead of a growing event log
