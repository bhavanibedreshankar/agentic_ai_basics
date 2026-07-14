# task_classifier_router

Classifying how demanding a request is with one cheap, upfront call, then routing to the model tier that classification calls for.

## task_classifier_router.py

A general assistant that classifies every request into `simple` / `moderate` / `complex` via a structured-output call, then answers it on Haiku, Sonnet, or Opus respectively. Type `exit` to quit.

### Concepts covered

- **`classify_complexity`** — a structured-output classification call (`json_schema` format, closed `enum` of tiers) run on the cheapest model in `MODEL_FOR_TIER`, since judging "is this simple?" is itself a simple task. Same fixed-schema pattern as `../../Agent_Frameworks_and_Patterns/router_agent/router_agent.py`'s `CLASSIFY_SCHEMA`.
- **`MODEL_FOR_TIER`** — the routing table itself: a plain dict mapping each complexity tier to an actual model ID.
- **`route`** — classify, then dispatch: two separate, sequential API calls, not one tool-calling loop, same shape as `router_agent.py`'s `route`.
- **Domain routing vs. tier routing** — `../../Agent_Frameworks_and_Patterns/router_agent/README.md` classifies request *domain* (billing/technical/sales) and always uses the same model; this template classifies request *complexity* and always uses the same system prompt. Its docstring literally calls out that the classifier is "a good candidate for a cheaper model" without doing it — this template is that idea, followed through.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Model_Routing/task_classifier_router/task_classifier_router.py
```

Try:

```
You: What year did WW2 end?
  [router] classified as 'simple' -> routing to claude-haiku-4-5-20251001

Claude (simple -> claude-haiku-4-5-20251001): World War 2 ended in 1945.

You: Design a zero-downtime migration plan for a sharded Postgres database
  [router] classified as 'complex' -> routing to claude-opus-4-8

Claude (complex -> claude-opus-4-8): ...
```

### Configuration

- `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `MODEL_FOR_TIER` — which model handles each tier
- `PRICE_PER_MILLION_INPUT` — approximate rates used for cost context; check platform.claude.com/pricing
- `CLASSIFY_SYSTEM_PROMPT` — the classifier's instructions, including its deliberate bias toward the *higher* tier when unsure

### See also

- `../../Agent_Frameworks_and_Patterns/router_agent/README.md` — the same classify-then-dispatch shape, routing on domain instead of complexity
- `../cost_aware_model_selection/README.md` — the same goal (pick a cheap-enough tier) without spending an extra API call on classification
