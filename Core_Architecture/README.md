# Core_Architecture

The pieces every agent in this repo is built from: a single API call, the model powering it, the instructions shaping it, the tools letting it act, and the autonomy that turns all of that into something more than question-answering. Two general-purpose templates (`basics/`, `token_tracking/`) that every other topic in the repo depends on also live here, since they're the literal foundation this topic's name describes.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`basics/`](basics/README.md) | A single request/response call — API settings, auth, and the core `ask_claude` pattern everything else builds on |
| 2 | [`token_tracking/`](token_tracking/README.md) | Measuring and monitoring token usage and cost — reading `usage`, pre-flight counting, cumulative tracking |
| 3 | [`system_prompt/`](system_prompt/README.md) | The fixed instruction set defining an agent's role and constraints — the same call as `basics/`, with the system prompt varied instead of fixed |
| 4 | [`llm_backbone/`](llm_backbone/README.md) | The model as a swappable reasoning engine — the same harness run against two different backbones |
| 5 | [`tool_use/`](tool_use/README.md) | Giving Claude the ability to act on the world, not just talk — JSON Schema tools, the client-side tool-calling loop |
| 6 | [`agent/`](agent/README.md) | Combining tools with autonomy: one goal in, an unattended perceive-reason-plan-act loop until it's done |
| 7 | [`modular_agent/`](modular_agent/README.md) | The same agent as `agent/`, rebuilt as a GAME-style (Goals, Actions, Memory, Environment) package — one file per concern, so new features never touch the loop |

## Setup

Same as the rest of the repo:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Core_Architecture/agent/agent.py
```

## How these relate to each other

| | What varies | What stays fixed | Number of calls |
|---|---|---|---|
| `basics/` | Nothing — one fixed call | Everything | Exactly 1 |
| `token_tracking/` | Nothing — same as `basics/`, plus reading `usage` | Everything | 1 per measured call |
| `system_prompt/` | The `system` parameter, across 3 prompts | Model, user message, call shape | 1 per prompt compared |
| `llm_backbone/` | The `model` parameter, across 2 backbones | System prompt, tools, task | 1+ per backbone compared |
| `tool_use/` | What the user asks for, turn by turn | The tool catalog and dispatch logic | 1+ per human turn |
| `agent/` | How many steps the goal takes, decided by the model itself | The goal, given once up front | 1+ per autonomous step, no human between them |
| `modular_agent/` | Which component you swap (goal, action, memory, provider) | The loop itself (`loop.py`), across every swap | Same as `agent/` — identical behavior, different structure |

`basics/` is the one fixed call every other template in this table is a variation of. `system_prompt/` and `llm_backbone/` each hold everything constant except one parameter, to make that parameter's effect visible in isolation — `system_prompt/` varies the *instructions*, `llm_backbone/` varies the *reasoning engine* itself. `tool_use/` adds the ability to act, but is still reactive: it waits for a human message before every action. `agent/` is what you get when you combine tool-calling with autonomy — hand it a goal once, and it decides for itself how many perceive-reason-plan-act cycles to run before it's done, which is the actual dividing line between "an agent" and "a chat assistant with tools." `modular_agent/` takes that exact same agent and asks a different question: not "does it work" but "can it grow" — splitting goals, memory, actions, environment, and the LLM wire format into separate, swappable classes around one fixed loop.
