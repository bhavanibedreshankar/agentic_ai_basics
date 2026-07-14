# streaming

Watching a graph's progress node by node as it runs, instead of only getting the final result once everything finishes.

## streaming.py

A document analysis pipeline (summarize -> extract keywords -> classify sentiment), run once with each of two `stream_mode` values so the difference in what gets yielded is directly visible. Type `exit` to end the session.

### Concepts covered

- **`.stream(..., stream_mode="updates")`** — yields only what each node just changed, as `{node_name: {changed_fields}}`; good for a progress indicator.
- **`.stream(..., stream_mode="values")`** — yields the full accumulated state after each node, including the initial snapshot before any node has run.
- HONESTY NOTE: there's no raw-`anthropic`-SDK streaming template elsewhere in this repo to contrast against — every other template calls `client.messages.create(...)` and waits for the complete response. The contrast here is entirely within this file: `.invoke()` (used everywhere else in [`../`](../README.md)) blocks until the whole graph finishes; `.stream()` is a generator over the same graph.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langgraph
export ANTHROPIC_API_KEY=your-key-here
python3 LangGraph/streaming/streaming.py
```

Try:

```
Text (or 'exit'): Our new feature launch exceeded every target we set for the quarter.

-- stream_mode='updates' --
  [updates] summarize -> {'summary': 'The new feature launch surpassed all quarterly targets.'}
  [updates] extract_keywords -> {'keywords': 'feature launch, targets, quarter, success'}
  [updates] classify_sentiment -> {'sentiment': 'positive'}

-- stream_mode='values' --
  [values] {'text': '...', 'summary': '', 'keywords': '', 'sentiment': ''}
  [values] {'text': '...', 'summary': 'The new feature launch surpassed all quarterly targets.', 'keywords': '', 'sentiment': ''}
  [values] {'text': '...', 'summary': '...', 'keywords': 'feature launch, targets, quarter, success', 'sentiment': ''}
  [values] {'text': '...', 'summary': '...', 'keywords': '...', 'sentiment': 'positive'}
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)

### See also

- [`../state_graph_basics/README.md`](../state_graph_basics/README.md) — the same kind of pipeline, only ever observed via `.invoke()`
