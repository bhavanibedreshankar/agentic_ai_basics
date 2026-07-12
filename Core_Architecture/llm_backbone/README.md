# llm_backbone

LLM (Backbone) — the large language model that serves as the reasoning engine inside an agent, swappable independently of the agent's architecture.

## llm_backbone.py

Runs one fixed harness — same system prompt, same tool, same loop shape as `../tool_use/basic_agentic_tools.py` — against the identical task under two different models (`claude-haiku-4-5` and `claude-sonnet-5`), printing each one's answer, token usage, and time taken side by side. Type your own task, or press enter for the built-in default.

### Concepts covered

- **The model is a parameter, not a constant** — every other template in this repo (starting with `../basics/basic.py`) fixes `MODEL` once at module level. Here, `run_with_backbone(model, task)` takes it as an argument on purpose, because the whole point is running the *same* harness against *different* backbones in the same process.
- **The harness stays fixed; only the backbone changes** — `run_with_backbone` is called twice with nothing different except the `model` string. Any difference in the two runs' answers, tone, token counts, or latency is attributable entirely to the backbone, because everything else was held constant.
- **Usage is per-backbone, not per-harness** — `response.usage` (see `../token_tracking/basic_token_tracking.py` for a dedicated deep dive) is accumulated separately for each model, making the cost/quality tradeoff between backbones directly comparable rather than theoretical.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Core_Architecture/llm_backbone/llm_backbone.py
```

Try:

```
Task to give both backbones (blank for a default demo):
> A store is offering 30% off a $85 jacket. What's the final price?

Running the same harness with 2 different backbones on:
  A store is offering 30% off a $85 jacket. What's the final price?

--- backbone: claude-haiku-4-5 ---
answer: The final price is $59.50.
tokens: 412 in / 18 out | time: 1.1s

--- backbone: claude-sonnet-5 ---
answer: The final price is $59.50, after a $25.50 discount.
tokens: 486 in / 27 out | time: 1.8s
```

### Configuration

- `BACKBONES` — the list of model IDs to compare (default: `["claude-haiku-4-5", "claude-sonnet-5"]`)
- `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../basics/README.md`; these stay fixed across every backbone in the comparison, unlike `model`
- `TOOLS` — the single `calculate` tool available to every backbone under test

### See also

- `../basics/README.md` — where `MODEL` is a fixed constant instead of a swappable parameter
- `../token_tracking/README.md` — a focused look at the `usage` object this file compares across backbones
- `../agent/README.md` — the same "one harness, autonomous loop" shape, focused on goal autonomy rather than backbone comparison
