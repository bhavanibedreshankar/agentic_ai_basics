# tool_result_caching

Memoizing an expensive, idempotent tool call's return value in a plain client-side dictionary, so repeating the same call returns instantly instead of redoing the underlying work.

## tool_result_caching.py

A travel-planning agent with a `get_weather(city)` tool that simulates slow network latency. Asking about the same city twice in one session hits the network only once. Type `exit` to end the conversation and see cache hit/miss totals.

### Concepts covered

- **`_weather_cache`** — an unbounded, permanent-for-the-process dict, keyed on lowercased city name. The simplest possible cache: no size limit, no expiration.
- **`get_weather`** — the memoization wrapper: check the cache before doing expensive work, populate it after. The same shape as `functools.lru_cache`, written out by hand so hit/miss bookkeeping is visible.
- **Client-side vs. server-side caching** — contrast with `../context_caching/README.md`, which caches how the *model* processed a prompt prefix inside Anthropic's infrastructure via `cache_control`. This template never touches that mechanism at all — it intercepts the call to *our own* tool function before it reaches the simulated network.
- **What this template deliberately skips** — no eviction, no expiration, no bound on key-space growth. `../cache_eviction_policies/README.md` picks up exactly here.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Caching/tool_result_caching/tool_result_caching.py
```

Try:

```
You: What's the weather in Tokyo?
    [cache miss for 'Tokyo' -- fetching...]

Claude: It's 72F and partly cloudy in Tokyo right now.

You: Should I pack a coat for Tokyo?
    [cache hit  for 'Tokyo' -- skipping network call]

Claude: Given it's 72F and partly cloudy, you probably won't need a coat.

You: exit

--- Cache summary: 1 hits / 1 misses ---
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `_CONDITIONS` — the fixed, canned "weather data" this template simulates instead of calling a real API

### See also

- `../cache_eviction_policies/README.md` — the same idea with a size and time bound, so the cache doesn't grow forever
- `../context_caching/README.md` — a different layer of caching entirely: the model's read of a prompt prefix, not a tool's return value
