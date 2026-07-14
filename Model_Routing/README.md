# Model_Routing

Four ways an agent decides which model actually answers a request, so a simple question never pays Opus prices and a hard one never gets stuck with an answer too weak for it: classify-then-route, free heuristics plus a budget, adaptive escalation on low confidence, and failover on real errors.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`task_classifier_router/`](task_classifier_router/README.md) | A cheap upfront classification call judges request complexity, then dispatches to the model tier that complexity calls for |
| 2 | [`cost_aware_model_selection/`](cost_aware_model_selection/README.md) | The same goal with no classification call at all — free local heuristics, plus a running session budget that forces the cheapest tier once spending is capped |
| 3 | [`complexity_based_escalation/`](complexity_based_escalation/README.md) | No upfront judgment either way — always try the cheapest model, and escalate up a ladder only when a tier admits it wasn't confident |
| 4 | [`multi_model_fallback/`](multi_model_fallback/README.md) | Routing driven by reliability, not cost or quality — fail over to another model only when the primary one is actually unavailable |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Model_Routing/task_classifier_router/task_classifier_router.py
```

## How these relate to each other

| | Routing decided by | Extra API cost to decide | When it triggers |
|---|---|---|---|
| `task_classifier_router/` | An LLM classifying the request | One small extra call per request | Always, before the real answer |
| `cost_aware_model_selection/` | Local string heuristics + a running budget | None | Always, before the real answer |
| `complexity_based_escalation/` | The current tier's own self-reported confidence | None beyond the answer itself | After seeing an actual (weak) answer |
| `multi_model_fallback/` | Whether the call raised a retryable API error | None | Only on an actual failure |

The first three all optimize the same thing — don't pay for more model than a request needs — from three different angles: pay a little to ask an LLM, pay nothing and guess from text features, or pay nothing upfront and let the cheap model's own performance decide. `multi_model_fallback/` is a different axis entirely: it doesn't care about cost or difficulty at all, only about keeping the agent answering when a specific model is temporarily down. A production system plausibly runs more than one of these at once — cost-aware or classifier-based routing to pick the ideal tier, with a fallback chain underneath in case that tier's provider has a bad moment.
