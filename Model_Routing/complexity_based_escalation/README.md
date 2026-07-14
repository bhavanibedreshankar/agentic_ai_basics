# complexity_based_escalation

Always trying the cheapest model first, and escalating up a ladder of stronger models only when the current tier reports it wasn't actually confident in its own answer.

## complexity_based_escalation.py

A general assistant with a Haiku -> Sonnet -> Opus escalation ladder. Every question starts at Haiku; if Haiku (via structured output) reports low confidence, the same question goes to Sonnet, and if Sonnet is also unconfident, on to Opus. Type `exit` to quit.

### Concepts covered

- **`ask_tier`** — one tier's attempt, returned as structured output (`{"answer": ..., "confident": ...}`) rather than a string marker to search for in free text — same "never free text to parse" principle as `../../Agent_Frameworks_and_Patterns/router_agent/router_agent.py`'s `CLASSIFY_SCHEMA`, applied to a confidence flag instead of a category.
- **`ask_with_escalation`** — walks `ESCALATION_LADDER` in order, stopping at the first confident tier (or the last tier regardless, so it always terminates and returns *something*).
- **Adaptive vs. upfront routing** — contrast with `../task_classifier_router/README.md` and `../cost_aware_model_selection/README.md`, both of which decide a tier *before* the first real answer is generated. This template decides *after seeing an actual answer* whether it was good enough, so the common case (a cheap model nails it) costs exactly one call, same as always using the cheap model — the expensive tiers are paid for only on the request slice that genuinely needs them.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Model_Routing/complexity_based_escalation/complexity_based_escalation.py
```

Try:

```
You: What color is the sky?

Claude (claude-haiku-4-5-20251001, 1 tier(s) tried): The sky is blue...

You: Prove there are infinitely many prime numbers
  [escalation] claude-haiku-4-5-20251001 not confident -- escalating to next tier
  [escalation] settled on tier 2/3: claude-sonnet-5

Claude (claude-sonnet-5, 2 tier(s) tried): Euclid's proof by contradiction...
```

### Configuration

- `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `ESCALATION_LADDER` — the ordered list of models to try, cheapest first
- `SYSTEM_PROMPT` — instructs every tier to self-report confidence honestly rather than bluff

### See also

- `../task_classifier_router/README.md` — decides the tier *before* answering, via a separate classification call
- `../multi_model_fallback/README.md` — also tries multiple models in sequence, but triggered by an actual API *error*, not by the model's own reported confidence
