# selective_context_retention

Scoring each incoming piece of information for importance the moment it arrives, and only letting the important pieces into the agent's persisted context — everything else is discarded immediately, not stored anywhere for later.

## selective_context_retention.py

An on-call incident assistant ingesting a simulated stream of raw log lines (heartbeats and debug noise mixed with errors, a config change, and a deploy). Each line is scored 0–10 for importance; only lines at or above `IMPORTANCE_THRESHOLD` are folded into the incident's persistent context. Once ingestion finishes, ask questions about the incident — answers come only from retained facts. Type `exit` to end the Q&A session.

### Concepts covered

- **`score_importance`** — a small, cheap classification call (`max_tokens=8`, `effort="low"`) whose only job is judging one line's importance. The model's output is untrusted text, parsed defensively with a regex and clamped to range rather than trusted as a well-formed integer — and unparseable output fails *closed* (treated as unimportant), not open.
- **`ingest`** — the retain/discard gate: importance at or above `IMPORTANCE_THRESHOLD` folds a line into `retained_context`; anything below is dropped. The demo keeps a `discarded_log` purely so you can see what was thrown away — a real agent wouldn't keep that list at all.
- **`build_system_prompt`** — injects *only* retained facts. Contrast with `../../Memory/memory_management/README.md`'s `basic_agentic_memory.py`, which injects every saved fact unconditionally.
- **Proactive vs. reactive discarding** — contrast with `../../Task_and_State_Management/context_management/README.md`'s `pruning.py`, which is reactive (content enters context, then gets removed once superseded) and `retrieval.py`, which keeps the full pool around externally forever. This template is proactive and genuinely destructive: low-importance input never becomes part of any store in the first place.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Caching/selective_context_retention/selective_context_retention.py
```

Try:

```
Ingesting a simulated incident log stream...

  [discarded score= 1] INFO heartbeat: worker-3 alive, queue_depth=12
  [retained  score= 9] ERROR checkout-service: payment gateway timeout after 30s, 214 requests affected
  [retained  score= 8] DEPLOY: checkout-service rolled back from v2.14.1 to v2.14.0 at 14:32 UTC
  ...

Done: 5 retained / 7 discarded out of 12 lines.

You: What changed right before the incident?
Claude: A config change raised the payment gateway timeout from 10s to 30s...

You: What was worker-2's queue depth?
Claude: I don't have that information in the retained incident facts.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `IMPORTANCE_THRESHOLD` — minimum score (0–10) to retain a line (default: `6`)
- `LOG_STREAM` — the simulated incoming stream fed through `ingest`

### See also

- `../../Task_and_State_Management/context_management/README.md` — three other techniques for the same underlying problem (context that grows unbounded), each with a different trade-off
- `../cache_eviction_policies/README.md` — a complementary technique once *something* is being retained: bounding how much of it sticks around
