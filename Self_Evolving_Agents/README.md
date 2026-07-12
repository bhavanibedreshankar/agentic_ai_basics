# Self_Evolving_Agents

Self-evolving agents are AI systems that autonomously update their own capabilities — such as prompts, tools, memory, and reasoning workflows — through continuous, closed-loop feedback. This directory currently has one template, focused on the most tractable version of that loop: an agent that rewrites its own system prompt based on feedback on its answers, and keeps the result across runs. (Evolving tools, memory schemas, or entire reasoning workflows are the same closed-loop idea applied to a different capability — see the notes in the template's docstring for where those live elsewhere in this repo.)

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`self_evolving_agents/`](self_evolving_agents/README.md) | A coding assistant that turns negative feedback into a persistent rule appended to its own system prompt, changing its behavior on the very next call |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run the template from the repo root:

```bash
python3 Self_Evolving_Agents/self_evolving_agents/self_evolving_agents.py
```

## What makes this different from "the agent got better"

Plenty of templates elsewhere in this repo produce better output through iteration or memory. What distinguishes *self-evolution* specifically:

| | Persists past one conversation? | What changes | Where the change lives |
|---|---|---|---|
| `self_evolving_agents/` | Yes — across process restarts | The agent's own system prompt | `evolved_rules.json` on disk |
| `../Planning_and_Reasoning/self_reflection/` | No | One draft, in place | Local variable, gone at exit |
| `../Memory/episodic_memory/` | Yes | Nothing — a log the agent can *choose* to consult | `episodes.json` on disk |

The load-bearing distinction is the last row versus the first: episodic memory gives the agent facts to reason over each time; self-evolution changes what the agent is instructed to do, automatically, without re-deriving the lesson from raw history on every call.
