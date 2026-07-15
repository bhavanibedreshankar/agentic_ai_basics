# modular_agent

The same autonomous goal-pursuing agent as `../agent/agent.py`, rebuilt as a
small, GAME-style (**G**oals, **A**ctions, **M**emory, **E**nvironment)
package instead of one script. Where `agent.py` fuses the prompt, the tool
catalog, the dispatch logic, and the loop into a single file, this template
splits each concern into its own class and file, so that adding a feature
later — a new tool, a new memory backend, a second goal, a different model
provider — never requires touching the loop that drives the agent.

## Architecture

| File | Class(es) | Responsibility |
|---|---|---|
| `goals.py` | `Goal` | What the agent is trying to achieve, as data |
| `memory.py` | `Memory`, `MemoryItem` | The growing record of what's happened so far |
| `actions.py` | `Action`, `ActionRegistry` | The catalog of things the agent can do, looked up by name |
| `environment.py` | `Environment`, `ActionResult` | Actually executes an action and turns a crash into a safe result |
| `language.py` | `AgentLanguage`, `AnthropicToolCallingLanguage` | Translates Goals/Memory/Actions into the LLM API's wire format |
| `loop.py` | `run_agent_loop` | The fixed perceive → reason → plan → act cycle |
| `agent.py` | `Agent` | Holds one of each of the above and exposes `run()` |
| `builtin_actions.py` | — | The extension point: concrete actions (`calculate`, `record_finding`, `terminate`) |
| `main.py` | — | Assembles one `Agent` from the pieces and runs it on a goal |

```
Goal(s) ──┐
Memory ───┼──> AgentLanguage ──> LLM call ──> tool_use? ──> Environment ──> ActionResult
Actions ──┘         ▲                              │                            │
                     └──────────── loop.py ─────────┴────────── back into Memory ┘
```

`loop.py` is the only file that ties all the others together at runtime, and
it does so purely through each class's public methods (`agent.language.construct_*`,
`agent.memory.add`, `agent.environment.execute_action`, `agent.actions.get_action`).
It never references a specific goal, tool name, or memory format directly —
see its docstring for why that's the whole point of this package.

## Concepts covered

- **Goals as data, not prose** — `Goal` objects instead of a hand-written system prompt string, so `language.py` can build the prompt from a list that's easy to inspect, extend, or generate at runtime.
- **Memory as its own object** — `Memory.add` / `get_memories` / `as_message_list`, instead of a bare `list[dict]` mutated inline by the loop (contrast `../agent/agent.py`'s `messages` list).
- **Actions looked up by name, not dispatched by `if`/`elif`** — `ActionRegistry.get_action(name)` replaces `../agent/agent.py`'s hardcoded `execute_tool` dispatcher, same idea as `../../Agent_Frameworks_and_Patterns/tool_registry/`.
- **Environment as the execution boundary** — the one place that calls an action's function and catches its exceptions, kept separate from both the action's definition (`actions.py`) and the loop that reacts to the result.
- **AgentLanguage as the provider seam** — the only file that knows what Claude's Messages API's `tools`/`system`/`messages` shape looks like; swapping providers means writing a new `AgentLanguage`, not touching anything else.
- **A terminal action** — `builtin_actions.terminate`, marked `terminal=True`, ends the loop deliberately with a final answer, as an alternative to relying only on `stop_reason != "tool_use"`.
- **The loop as fixed algorithm, everything else as data** — `run_agent_loop` in `loop.py` is intentionally the most boring file here: it should almost never need to change as the agent grows.

## Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Core_Architecture/modular_agent/main.py
```

Try (or just press enter for the built-in default):

```
Goal: I want to buy 3 items priced at $12.50, $7.25, and $19.99. My budget is $50. Do they fit, and how much would I have left over?

[step 1] act: calculate({'expression': '12.50 + 7.25 + 19.99'})
[step 1] perceive: 12.50 + 7.25 + 19.99 = 39.74

[step 2] act: record_finding({'finding': 'total is 39.74, under the $50 budget'})
[step 2] perceive: Noted: total is 39.74, under the $50 budget

[step 3] act: terminate({'final_answer': 'Yes, they fit under your $50 budget, with $10.26 left over.'})
[step 3] perceive: Yes, they fit under your $50 budget, with $10.26 left over.

Final answer: Yes, they fit under your $50 budget, with $10.26 left over.
```

## Extending this without touching the loop

This is the whole point of the package, so each change lists exactly which
file(s) it touches:

- **Add a new tool/capability** — write a function in `builtin_actions.py` (or a new file) shaped `(tool_input: dict) -> str`, wrap it in an `Action`, `register()` it. `loop.py` and `agent.py` are untouched.
- **Add a second goal, or generate goals at runtime** — extend the `goals` list passed into `Agent(...)` in `main.py`. `language.py`'s `construct_system_prompt` already loops over however many goals there are.
- **Swap in a smarter memory strategy** — write a class with the same `add` / `get_memories` / `as_message_list` methods as `Memory` (e.g. one that prunes old turns, like `../../Task_and_State_Management/context_management/`, or persists to disk, like `../../Memory/external_memory.py`) and pass an instance of it to `Agent(memory=...)`.
- **Swap the model or provider** — change `Agent(model=...)` for a same-provider swap (see `../llm_backbone/`), or write a new `AgentLanguage` subclass for a genuinely different API shape (e.g. for `../../Model_Routing/multi_model_fallback.py`-style failover).
- **Add safety limits (approval gates, breakpoints)** — wrap the call to `agent.environment.execute_action(...)` inside `loop.py`'s single loop body — this is the one legitimate reason to touch `loop.py`, since it's about the algorithm itself, not a new capability. See `../../Execution_Loops/human_in_the_loop/` and `../../Execution_Loops/interrupts_breakpoints/` for the patterns.

## Configuration

- `Agent.model`, `Agent.max_tokens`, `Agent.effort` — see `../basics/README.md`
- `Agent.run(user_input, max_iterations=8)` — the hard cap on perceive-reason-act cycles, same role as `../agent/agent.py`'s `MAX_STEPS`
- `builtin_actions.register_builtin_actions` — the three example actions available to the agent; add more here

## See also

- `../agent/README.md` — the identical use case as one monolithic script; read this alongside `main.py` to see exactly what splitting into a package buys you
- `../tool_use/README.md` — the underlying tool-calling loop mechanics this package's `loop.py` builds on
- `../../Agent_Frameworks_and_Patterns/tool_registry/README.md` — the registry-by-name pattern `actions.py` uses
- `../../Memory/README.md` — real alternative `Memory` implementations to swap in
- `../../Execution_Loops/README.md` — approval gates, breakpoints, and iteration caps to layer onto `loop.py`
- `../../Model_Routing/README.md` — routing/fallback logic that would live behind a custom `AgentLanguage` or a swapped `Agent.client`
