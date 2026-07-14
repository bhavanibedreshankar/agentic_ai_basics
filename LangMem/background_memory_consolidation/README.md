# background_memory_consolidation

The hot-path vs. background distinction: extracting memory synchronously and blocking, vs. asynchronously and debounced.

## background_memory_consolidation.py

A burst of three quick messages in one support conversation, submitted for background consolidation as they arrive — only the LAST submission's reflection actually runs. Run directly; not interactive (the timing/debounce behavior is easiest to see against a fixed sequence).

### Concepts covered

- **`ReflectionExecutor(manager, store=store)`** — wraps a synchronous extraction `Runnable` (the same kind `../semantic_memory/` calls directly) so it's submitted rather than invoked, and runs in a real background `threading.Thread`.
- **`executor.submit(payload, config=..., after_seconds=..., thread_id=...)`** — schedules a reflection to run after a delay; submitting again with the SAME `thread_id` before it fires CANCELS the pending one and replaces it — verified directly: 2 of 3 submissions in the burst came back `.cancelled() == True`, and the underlying model was called exactly once, not three times.
- **Hot-path vs. background** — `../semantic_memory/` and `../memory_management_tools/` both pay the extraction cost (a blocking model call) on every triggering message; this template pays it once, after a burst settles, off the response path entirely.

### Run

From the repo root:

```bash
pip install -r requirements.txt
pip install langchain langgraph langmem
export ANTHROPIC_API_KEY=your-key-here
python3 LangMem/background_memory_consolidation/background_memory_consolidation.py
```

Output:

```
[1/3] message arrives: 'I prefer email.' — submitting for background consolidation
[2/3] message arrives: 'Actually, email or text is fine.' — submitting for background consolidation
[3/3] message arrives: 'On second thought, just always email me, never call.' — submitting for background consolidation

Of 3 submissions, 2 were cancelled by a later one.
Waiting for the final (uncancelled) submission to actually run...

Consolidation ran once, on the FINAL message only: [...]

Compare: ../semantic_memory/'s hot-path equivalent would have called the model once PER message above
(3 calls), blocking the reply each time, instead of once in the background after the burst settled.
```

### Configuration

- `MODEL` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `CONSOLIDATION_DELAY_SECONDS` — how long a submission waits before running; increase to tolerate a longer burst window

### See also

- [`../semantic_memory/README.md`](../semantic_memory/README.md) — the synchronous, hot-path equivalent this template avoids running repeatedly
- [`../memory_management_tools/README.md`](../memory_management_tools/README.md) — another hot-path trigger, this time model-decided instead of code-decided
