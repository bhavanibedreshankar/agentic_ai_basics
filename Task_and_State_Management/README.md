# Task_and_State_Management

Four ways an agent manages the shape and progress of its own work — breaking a goal down, constraining what states it can be in, keeping its live context lean, and surviving a crash partway through.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`task_decomposition/`](task_decomposition/README.md) | Breaking a goal into a structured, multi-level tree of subtasks — as its own inspectable artifact, before any execution happens |
| 2 | [`state_machine/`](state_machine/README.md) | A formal table of legal states and transitions, enforced in code before a state-changing action is ever allowed to run |
| 3 | [`context_management/`](context_management/README.md) | Keeping the context window relevant as a conversation grows: pruning, summarization, and retrieval |
| 4 | [`checkpointing/`](checkpointing/README.md) | Saving progress after every step so a long-running, multi-step task can resume exactly where it left off after a crash |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Task_and_State_Management/state_machine/state_machine.py
```

## How these relate to each other

| | What "state" means here | When it's checked/saved | Survives a crash? |
|---|---|---|---|
| `task_decomposition/` | The shape of the work — a tree of subtasks | Once, up front, before execution starts | N/A — nothing has run yet |
| `state_machine/` | Which of a fixed set of states an entity (an order) is currently in | Before every attempted transition | No — in-memory only, but see `checkpointing/` for the durable version of this idea |
| `context_management/` | What's currently in the live conversation | Every turn, as the conversation grows | No — by design; it manages what's live, not what's durable |
| `checkpointing/` | Progress through a fixed multi-step pipeline | After every completed step | **Yes** — the whole point |

`task_decomposition/` produces the step list; `checkpointing/` is what makes running that list durable across a crash — see `checkpointing/README.md`'s note on swapping in a decomposed step list instead of its own fixed `STEPS`. `state_machine/` and `checkpointing/` both track "how far along is this," but `state_machine/` is about which transitions are *legal* (an order can't skip from pending to shipped), while `checkpointing/` is about which steps are *already done* (nothing here stops you from re-running a step — it just avoids re-running one that already succeeded).
