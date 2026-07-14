# cost_aware_model_selection

Picking a model tier from free, local heuristics on the request text — no extra API call — plus a running session budget that forces the cheapest tier once spending gets close to a cap.

## cost_aware_model_selection.py

A budget-capped chat assistant. Each request is scored `cheap` or `standard` by plain string heuristics (length, complexity keywords, multi-part questions); a `SessionBudget` tracks estimated spend and overrides the heuristic to force `cheap` once the session cap is reached. Type `exit` to end the conversation and see the spending summary.

### Concepts covered

- **`score_complexity`** — zero-cost heuristic scoring: length, `COMPLEXITY_KEYWORDS`, and multiple `?`s all bump a request to `standard`. No API call happens to make this decision, unlike `../task_classifier_router/task_classifier_router.py`'s `classify_complexity`.
- **`SessionBudget.choose_tier`** — where the heuristic's choice can get overridden: once `spent_usd >= cap_usd`, every remaining request is forced to `cheap` regardless of what the heuristic wanted.
- **`SessionBudget.record`** — converts a response's real `usage` into an estimated dollar cost and accumulates it, the same idea as `../../Benchmarking/latency_cost_benchmarking/README.md`'s cost tracking, applied per-tier here.
- **Free heuristic vs. paid classification** — contrast with `../task_classifier_router/README.md`: that template pays for a small extra call to get a more reliable complexity judgment; this one accepts a blunter signal for zero marginal cost per request.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Model_Routing/cost_aware_model_selection/cost_aware_model_selection.py
```

Try a few short questions, then a longer one with "design" or "step by step" in it, and keep going until the budget forces everything cheap:

```
You: What's 2+2?
  [router] heuristic tier='cheap' -> claude-haiku-4-5-20251001

Claude (cheap): 4

You: Can you design a caching strategy and explain the trade-offs step by step?
  [router] heuristic tier='standard' -> claude-sonnet-5
...
You: exit

--- Estimated session spend: $0.0031 (cap: $0.0200) ---
```

### Configuration

- `MODEL_FOR_TIER`, `PRICE_PER_MILLION_INPUT` / `_OUTPUT` — see the file; approximate rates, check platform.claude.com/pricing
- `BUDGET_CAP_USD` — session spending cap before every request is forced cheap (default: `0.02`)
- `COMPLEXITY_KEYWORDS`, `LONG_REQUEST_CHARS` — the heuristic's tunable signals

### See also

- `../task_classifier_router/README.md` — the same routing goal via a real (cheap) classification call instead of free heuristics
- `../../Core_Architecture/token_tracking/README.md` — general per-turn cost tracking, without a tier-selection decision attached
