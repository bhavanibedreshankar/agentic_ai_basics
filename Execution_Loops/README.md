# Execution_Loops

The core loop every agent runs, and three ways to keep it under control — pausing on specific actions, pausing on specific conditions, and an unconditional backstop that doesn't need either to have been anticipated.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`agentic_loop/`](agentic_loop/README.md) | The core observe → think → act → observe cycle, in its simplest form: a multi-turn chat loop, with an explicit mapping of the code onto each phase |
| 2 | [`human_in_the_loop/`](human_in_the_loop/README.md) | A human approves, rejects, or edits specific tool calls before they execute |
| 3 | [`interrupts_breakpoints/`](interrupts_breakpoints/README.md) | Execution pauses when a predefined *condition* is met — cost, a flagged action, a detected stall — not on every call to a given tool |
| 4 | [`max_iterations/`](max_iterations/README.md) | An unconditional cap on loop cycles, catching the runaway case no condition-based check was written to anticipate |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Execution_Loops/human_in_the_loop/human_in_the_loop.py
```

## How these relate to each other

| | What triggers a pause | Fires on | Can it be silently bypassed by an unanticipated situation? |
|---|---|---|---|
| `agentic_loop/` | Nothing — the baseline cycle these other three constrain | N/A | N/A |
| `human_in_the_loop/` | A specific **tool name** | Every single call to that tool, unconditionally | Yes — any tool not on the approval list runs freely |
| `interrupts_breakpoints/` | A **condition** over running state (cost, action, stall) | Only when that specific condition is met | Yes — a condition nobody thought to write won't fire |
| `max_iterations/` | An **iteration count** | Always, once the count is reached, regardless of cause | **No** — it has no condition to miss; it's the backstop for everything the other two might not catch |

`human_in_the_loop/` and `interrupts_breakpoints/` are both about STOPPING for a reason a human can act on; `max_iterations/` is about stopping for *no* reason beyond "this has gone on long enough" — cheap insurance against exactly the failure modes the other two can't anticipate in advance. A production agent typically layers all of them: approval gates on genuinely risky actions, breakpoints on cost/stall conditions, and a max-iteration cap underneath both as the last line of defense.
