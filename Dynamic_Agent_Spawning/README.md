# Dynamic_Agent_Spawning

An agent capable of creating a brand-new sub-agent at runtime — choosing its role, persona, and system prompt on the fly — and assigning it a task, instead of delegating among a fixed roster of specialists someone hardcoded in advance. This directory currently has one template, showing the pattern at its simplest: a meta-agent with zero built-in specialists and exactly one tool for inventing them.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`dynamic_agent_spawning/`](dynamic_agent_spawning/README.md) | A meta-agent that spawns as many uniquely-defined sub-agents as a request needs, capped by `MAX_SUBAGENTS_PER_TURN` |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run the template from the repo root:

```bash
python3 Dynamic_Agent_Spawning/dynamic_agent_spawning/dynamic_agent_spawning.py
```

## When would an agent actually need to spawn one dynamically?

The rule of thumb this template demonstrates: reach for a **fixed** roster (`../Multi_Agent_Systems/orchestrator/`) when you can enumerate the specialists a system will ever need ahead of time — a content pipeline really only ever needs a researcher, a writer, and an editor. Reach for **dynamic spawning** when the specialists needed depend on the specific request in a way you can't enumerate in advance — an assistant fielding arbitrary real-world questions (legal, medical, financial, technical — in any combination, for topics nobody wrote code for) has no fixed list to delegate among; it has to decide what kind of expert it needs *and construct one* for each request.

| | Specialists known at | Roster size | New expertise requires |
|---|---|---|---|
| `../Multi_Agent_Systems/orchestrator/` | write time | fixed (3) | editing the source file |
| `dynamic_agent_spawning/` | request time | unbounded (capped per-turn) | nothing — the parent just calls `spawn_subagent` differently |
