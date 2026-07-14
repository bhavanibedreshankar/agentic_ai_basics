# callbacks_and_tracing

Hooks for logging, monitoring, and debugging chain/agent execution.

## callbacks_and_tracing.py

A two-step draft-then-critique pipeline, traced end to end by a custom callback handler that writes a JSONL log — the same kind of record [`../../Safety_and_Control/audit_trail/audit_trail.py`](../../Safety_and_Control/audit_trail/README.md) produces, but attached from outside the chain instead of written into it. Type `exit` to end the conversation.

### Concepts covered

- **`BaseCallbackHandler`** — implement only the lifecycle methods you care about (`on_chain_start`, `on_chat_model_start`, `on_llm_end`); the rest default to no-ops.
- **`config={"callbacks": [handler]}`** — attaches the tracer to any `.invoke()` call without changing `DRAFT_PROMPT`, `CRITIQUE_PROMPT`, or the chains built from them.
- **`run_id`** — uniquely identifies one step's execution, matching a call's start time to its end time even when steps run concurrently (e.g. [`../chains/chains.py`](../chains/README.md)'s `RunnableParallel` branches).
- **`print_trace`** — replays the JSONL log from disk only, proving it's an independent record of what happened, same proof-of-independence idea as `audit_trail.py`'s `print_audit_log`.
- Contrast with `audit_trail.py`, where `log.record(...)` calls are written directly into `run_turn()` — logging is welded to business logic there; here the same chain code in `chains.py` never mentions logging at all.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 LangChain/callbacks_and_tracing/callbacks_and_tracing.py
```

Try:

```
Question: What is 2+2?
  [trace] llm_start: 'Write one sentence answering: What is 2+2?'
  [trace] llm_end: 812.3ms -> 'The answer is 4.'
  [trace] llm_start: 'Critique this answer in one short sentence...'
  [trace] llm_end: 640.1ms -> 'Accurate and complete.'

Draft: The answer is 4.
Critique: Accurate and complete.
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see [`../../Core_Architecture/basics/README.md`](../../Core_Architecture/basics/README.md)
- `TRACE_LOG_PATH` — where the JSONL trace is written (gitignored; runtime data, not source)

### See also

- [`../../Safety_and_Control/audit_trail/README.md`](../../Safety_and_Control/audit_trail/README.md) — the same structured-log idea, written by hand into the chain's own code
- [`../chains/README.md`](../chains/README.md) — the pipeline this template traces
