# Agent_Frameworks_and_Patterns

Five compositional patterns for structuring how an agent (or several) turns a request into a result — from a single agent's tool catalog, through fixed and dynamic multi-call pipelines, to a standalone reusable scorer.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`tool_registry/`](tool_registry/README.md) | A catalog of available tools, with descriptions, that both advertises tools to the LLM and dispatches calls to them — moved here from `Tools_and_Actions/` |
| 2 | [`prompt_chaining/`](prompt_chaining/README.md) | Multiple LLM calls in a fixed sequence, output of one feeding the next — moved here from the repo root |
| 3 | [`router_agent/`](router_agent/README.md) | Classifying input and dispatching to the right specialist as a single upfront decision |
| 4 | [`evaluator_agent/`](evaluator_agent/README.md) | A standalone, reusable LLM call that scores any output against any caller-supplied rubric |
| 5 | [`../Planning_and_Reasoning/self_reflection/`](../Planning_and_Reasoning/self_reflection/README.md) | **Reflection Loop** — generate, critique, revise, repeat until quality passes. Stays under `Planning_and_Reasoning/`, where it's already integrated into that topic's own comparison table; linked here rather than duplicated |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Agent_Frameworks_and_Patterns/router_agent/router_agent.py
```

## How these relate to each other

| | Number of calls | Sequence decided by | Reusable across tasks? |
|---|---|---|---|
| `tool_registry/` | One call, N possible tool round-trips | The model, per request | The *catalog* is reusable; each call is task-specific |
| `prompt_chaining/` | A fixed N calls | The developer, hardcoded in advance | The *shape* is reusable; each step is written for one pipeline |
| `router_agent/` | Exactly 2 (classify, then respond) | The classifier's own output | The classifier is reusable across any request in the same domain |
| `evaluator_agent/` | 1 per evaluation | N/A — no sequence, a single utility call | **Yes, by design** — task/output/rubric are all arguments, not hardcoded |
| `self_reflection/` (external) | Variable, up to `MAX_ROUNDS` | Whether the critic approves | The loop shape is reusable; the critic's criteria are task-specific |

`tool_registry/` and `prompt_chaining/` are the two foundational shapes everything else in this directory is a variation of: tool_registry lets the *model* decide what happens next within one call; prompt_chaining fixes the sequence in code ahead of time. `router_agent/` sits between them — one decision point, made by a dedicated classification call rather than either a tool choice or a hardcoded sequence. `evaluator_agent/` and `self_reflection/` both involve judging output, but `evaluator_agent/` is a general-purpose scorer you call from anywhere, while `self_reflection/` is a complete generate-critique-revise loop built around one task.
