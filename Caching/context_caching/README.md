# context_caching

Using Anthropic's server-side prompt caching (`cache_control`) so a large, stable prefix — a policy handbook, tool definitions, earlier conversation turns — gets billed at a fraction of full price on every request after the first.

## context_caching.py

A Nimbus Cloud Storage support agent whose system prompt is a multi-thousand-token policy handbook that never changes between turns, plus a small `look_up_account_tier` tool. The first turn pays a cache-write premium to store the handbook; every turn after that reads it back cheaply instead of reprocessing it. Type `exit` to end the conversation and see a cache economics summary.

### Concepts covered

- **`build_system_blocks`** — places a `cache_control` marker on the last (and only) system content block. Because render order is `tools -> system -> messages`, this one marker caches the tool definitions *and* the handbook together — no separate marker needed on `TOOLS`.
- **`add_cache_breakpoint`** — moves the breakpoint to the last content block of the most recently appended turn on every call, so the cached prefix grows to cover the whole conversation so far, not just the system prompt. Returns a deep copy so the stored `messages` list itself stays marker-free.
- **`CacheStats`** — reads `cache_creation_input_tokens` / `cache_read_input_tokens` / `input_tokens` off each response and estimates the session's actual cost against what it would have cost with no caching at all. Builds on `../../Core_Architecture/token_tracking/basic_token_tracking.py`'s `SessionUsage`, which prints these same fields but never sets a `cache_control` marker to make them non-zero in the first place.
- **The `POLICY_HANDBOOK` constant** — deliberately large (~1,500+ tokens) to clear the model-dependent minimum cacheable prefix, and deliberately free of anything that changes per-request (no timestamps, no session IDs) — the single most common way prompt caching silently breaks.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Caching/context_caching/context_caching.py
```

Try:

```
You: What's your refund policy?
  [cache: 0 read / 1543 written / 12 at full price]

Claude: Monthly subscriptions are refundable within 14 days...

You: What about annual plans?
  [cache: 1543 read / 28 written / 15 at full price]

Claude: Annual subscriptions are refundable within 30 days...

You: exit

--- Cache economics summary ---
Turns:                   2
...
Savings from caching:    41.2%
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `POLICY_HANDBOOK` — the cached, stable system prompt content
- `CacheStats.PRICE_PER_MILLION_INPUT` — approximate Sonnet pricing used for the cost comparison; check platform.claude.com/pricing for current rates

### See also

- `../tool_result_caching/README.md` — a completely different layer: caching a tool's *return value* client-side, rather than the model's read of a prompt prefix server-side
- `../../Core_Architecture/token_tracking/README.md` — a closer look at reading `usage` in general, without the caching opt-in
