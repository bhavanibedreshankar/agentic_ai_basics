# cache_eviction_policies

Bounding a client-side cache with an explicit eviction policy — a size limit enforced by LRU (least-recently-used) eviction, and a time limit enforced by TTL (time-to-live) expiration — instead of letting it grow forever.

## cache_eviction_policies.py

The same simulated `get_weather(city)` tool as `../tool_result_caching/tool_result_caching.py`, now wrapped in a `BoundedCache` with `maxsize=3` and `ttl_seconds=10.0`. Asking about a 4th distinct city forces a visible eviction; re-asking about an evicted city — or waiting past the TTL — forces a fresh fetch instead of a stale hit. Type `exit` to end the conversation and see cache stats.

### Concepts covered

- **`BoundedCache.get` / `.put`** — an `OrderedDict`-backed cache where every read or write moves an entry to the back (most-recently-used end) via `move_to_end`, so the front of the dict is always the least-recently-used entry.
- **LRU eviction** — `put` calls `popitem(last=False)` to evict the front entry once the cache is at `maxsize`, printing which key got evicted and why.
- **TTL expiration** — `get` checks `time.time() - written_at >= ttl_seconds` and treats an expired entry as a miss even though the key is technically still present — stale data being worse than no data, for something like weather that goes stale on its own regardless of access pattern.
- **What this builds on** — contrast with `../tool_result_caching/README.md`'s `_weather_cache`, a plain dict with neither limit; this template is that same idea made safe for a long-running process.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Caching/cache_eviction_policies/cache_eviction_policies.py
```

Try asking about `tokyo`, `lisbon`, `reykjavik`, then `nairobi` in a row (four distinct cities against a `maxsize=3` cache) to see an eviction, then re-ask about `tokyo` to see it re-fetched instead of served stale:

```
You: What's the weather in tokyo, lisbon, reykjavik, and nairobi?
    [cache miss for 'tokyo' -- fetching...]
    [cache miss for 'lisbon' -- fetching...]
    [cache miss for 'reykjavik' -- fetching...]
    [evicted 'tokyo' to make room -- cache at maxsize=3]
    [cache miss for 'nairobi' -- fetching...]

You: And tokyo again?
    [cache miss for 'tokyo' -- fetching...]

You: exit

--- Cache summary: hits=0 misses=5 evictions=1 expirations=0 current size=3/3 ---
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `BoundedCache(maxsize, ttl_seconds)` — instantiated as `weather_cache = BoundedCache(maxsize=3, ttl_seconds=10.0)`; both knobs are deliberately small so eviction and expiration are easy to trigger interactively

### See also

- `../tool_result_caching/README.md` — the unbounded version this template replaces the cache in
- `../context_caching/README.md` — a server-side cache with its own eviction rule (a 5-minute or 1-hour TTL set by Anthropic's infrastructure), for contrast with this hand-rolled client-side one
