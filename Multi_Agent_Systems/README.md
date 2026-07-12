# Multi_Agent_Systems

Five ways multiple agents can work together, instead of one agent doing everything itself — each pattern differs in who stays in control, whether coordination is sequential or parallel, and whether output gets checked before being trusted.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`orchestrator/`](orchestrator/README.md) | One agent delegates subtasks to different specialists via tool calls, always staying in control |
| 2 | [`worker_agent/`](worker_agent/README.md) | What's actually inside a delegate: a self-contained agent with its own tools and internal tool-calling loop |
| 3 | [`agent_handoff/`](agent_handoff/README.md) | Control transfers permanently from a triage agent to a specialist — no delegation, no returning |
| 4 | [`supervisor_pattern/`](supervisor_pattern/README.md) | An orchestrator that validates a worker's output against deterministic criteria and retries with feedback on failure |
| 5 | [`swarm/`](swarm/README.md) | No coordinator at all — independent agents run genuinely in parallel, merged only afterward |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Multi_Agent_Systems/orchestrator/orchestrator.py
```

## How these relate to each other

| | Who's in control | Coordination | Output checked? |
|---|---|---|---|
| `orchestrator/` | The orchestrator, always | Sequential, deliberate delegation | No — result is trusted as-is |
| `worker_agent/` | Whatever calls `.run()` | N/A — one self-contained unit | N/A |
| `agent_handoff/` | Whichever agent is currently active | Sequential, but control *transfers* rather than returning | No |
| `supervisor_pattern/` | The supervisor, always | Sequential, with retry | **Yes** — validated, retried with feedback on failure |
| `swarm/` | No one — no coordinator | **Parallel**, no ordering at all | No — all results merged, none discarded |

`orchestrator/` and `supervisor_pattern/` both keep one agent permanently in charge; `agent_handoff/` deliberately gives that up. `orchestrator/` and `swarm/` both fan out to multiple specialists, but one is sequential-and-deliberate while the other is parallel-and-uncoordinated. `worker_agent/` is the one template about a single unit rather than a coordination pattern — it's what's inside every `delegate_to_X` call, every handoff target, and every swarm member in the other four.
