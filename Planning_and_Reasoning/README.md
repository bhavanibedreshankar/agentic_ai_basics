# Planning_and_Reasoning

Five techniques for getting Claude to reason more reliably before acting or answering — from a single interleaved reason/act loop up through exploring multiple reasoning paths and critiquing its own output.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`chain_of_thought/`](chain_of_thought/README.md) | The foundation: reasoning step-by-step before answering, compared three ways — direct, prompted, and native extended thinking |
| 2 | [`react/`](react/README.md) | Interleaving reasoning with tool calls — a labeled Thought → Action → Observation cycle |
| 3 | [`tree_of_thought/`](tree_of_thought/README.md) | Extending chain-of-thought into several parallel reasoning branches, evaluated and ranked against each other |
| 4 | [`plan_and_execute/`](plan_and_execute/README.md) | Generating a dynamic, task-specific plan first, then executing each step in sequence |
| 5 | [`self_reflection/`](self_reflection/README.md) | Iteratively critiquing and revising a single output until it's good enough |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Planning_and_Reasoning/react/react_agent.py
```

## How these relate to each other

- **`chain_of_thought/`** reasons once, linearly, before answering.
- **`react/`** interleaves that reasoning with tool calls, one action at a time, narrating why before each one.
- **`tree_of_thought/`** runs chain-of-thought-style reasoning multiple times with different strategies, then picks the best result — width instead of a single path.
- **`plan_and_execute/`** decides the *shape* of the work up front (a model-generated plan), then works through it step by step — contrast with `../prompt_chaining/`, where the sequence of steps is fixed by the developer rather than generated per task.
- **`self_reflection/`** takes a single output and improves it in place through repeated critique — contrast with `tree_of_thought/`, which compares several independent outputs rather than revising one.
