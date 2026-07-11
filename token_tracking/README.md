# token_tracking

How to measure and monitor token usage — and therefore cost — in a Claude API application.

## basic_token_tracking.py

A chat agent (same shape as `../agentic_loop/basic_agentic_loop.py`) that prints token usage and estimated cost after every turn, plus a running session summary when you type `exit`.

### Concepts covered

- **Reading usage from a response** — `response.usage.input_tokens` / `.output_tokens`, and the cache-related fields that report cost savings when prompt caching is in play.
- **Pre-flight token counting** — `preview_input_tokens` uses the `count_tokens` endpoint to see a request's size *before* sending it, at no generation cost — useful for warning on an oversized request or deciding whether to trim history first.
- **Cumulative tracking** — the `SessionUsage` class accumulates totals across every turn, since a multi-turn chat resends its full history on every call, so cost compounds as the conversation grows. This template doesn't trim history, so you can watch that growth happen — pair it with `../memory_management/basic_agentic_memory.py`'s `trim_history` if you want to bound it.
- **Cache fields** — `cache_creation_input_tokens` / `cache_read_input_tokens` are tracked too, with a comment on why they matter (cache reads are billed at a fraction of normal input cost) even though this template doesn't use prompt caching itself.
- **Cost estimation** — `estimate_cost` converts token counts into an approximate dollar figure using per-token pricing.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 token_tracking/basic_token_tracking.py
```

Each turn prints usage inline:

```
You: What's the capital of France?
  [pre-flight estimate: ~24 input tokens]
  [usage: 24 in / 8 out, ~$0.0002 this turn]

Claude: The capital of France is Paris.
```

Type `exit` to see the session summary:

```
--- Session summary ---
Turns:          3
Input tokens:   412
Output tokens:  187
Cache reads:    0 (billed ~10x cheaper than input)
Cache writes:   0
Estimated cost: $0.0040
```

### Configuration

Edit the constants at the top of `basic_token_tracking.py`:

- `MODEL` — the Claude model used (default: `claude-sonnet-5`)
- `MAX_TOKENS` — max response length
- `EFFORT` — thinking/response depth (default: `medium`)
- `SYSTEM_PROMPT` — instructions that shape Claude's behavior as the assistant
- `PRICE_PER_MILLION_INPUT` / `PRICE_PER_MILLION_OUTPUT` — approximate pricing used for cost estimates; check [platform.claude.com/pricing](https://platform.claude.com/pricing) for current rates

### See also

- `../memory_management/basic_agentic_memory.py` — bounding context growth (and therefore cost) with a sliding window
