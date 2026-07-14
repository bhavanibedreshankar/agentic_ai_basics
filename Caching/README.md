# Caching

Four ways an agent keeps only what's worth paying to keep: caching a stable prompt prefix server-side, deciding what's important enough to retain in the first place, bounding a retained cache so it doesn't grow forever, and memoizing an expensive tool call's result.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`context_caching/`](context_caching/README.md) | Server-side prompt caching (`cache_control`) — reusing a large, stable system prompt and tool definitions across turns at a fraction of full input price |
| 2 | [`selective_context_retention/`](selective_context_retention/README.md) | Deciding, per incoming item, whether it's important enough to enter the agent's context at all — the rest is discarded outright, not stored anywhere |
| 3 | [`cache_eviction_policies/`](cache_eviction_policies/README.md) | Bounding a client-side cache of retained items with an explicit LRU (size) and TTL (time) eviction policy |
| 4 | [`tool_result_caching/`](tool_result_caching/README.md) | Memoizing an expensive, idempotent tool call's return value client-side — the unbounded cache that (3) goes on to bound |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Caching/context_caching/context_caching.py
```

## How these relate to each other

| | What's being cached | Where it lives | What decides eviction |
|---|---|---|---|
| `context_caching/` | The model's own read of a prompt prefix (system + tools + history) | Server-side, inside Anthropic's infrastructure | A 5-minute (or 1-hour) TTL set by the API, invisible to your code |
| `selective_context_retention/` | Nothing yet — this is the *gate* deciding what's worth caching/retaining at all | N/A (a decision point, not a store) | An importance score computed per item, checked once at ingestion |
| `cache_eviction_policies/` | Whatever a tool call already returned | Client-side, an in-process `BoundedCache` | An explicit LRU + TTL policy you write and control |
| `tool_result_caching/` | Same as above, without a bound | Client-side, a plain dict | Nothing — entries live for the whole process |

`context_caching/` and `tool_result_caching/`/`cache_eviction_policies/` operate on two entirely different layers — one caches how the *model* processes a prefix, the other caches the *return value of your own code* — and a real agent typically uses both at once. `selective_context_retention/` is the odd one out: it's not a cache implementation but the filter that decides what's worth feeding into any of the others in the first place, directly reflecting the "keep the important, discard the unwanted" idea this whole topic is built around.
