# agent

Agent — an AI system that autonomously perceives, reasons, plans, and acts to achieve a goal, beyond simple question-answering.

## agent.py

A budget-checking agent. Give it one goal (e.g. "do these purchases fit under my budget?") and `pursue_goal()` runs unattended — deciding its own steps, tool calls, and when it's done — until it produces a final answer or hits `MAX_STEPS`.

### Concepts covered

- **Autonomy over the whole task, not just one call** — contrast with `../basics/basic.py`'s `ask_claude`: one message in, one message out, no notion of "working on it." Here, a single call to `pursue_goal()` runs an entire perceive → reason → plan → act cycle, repeatedly, deciding on its own how many steps the goal needs.
- **No human between steps** — contrast with `../tool_use/basic_agentic_tools.py`'s `run_turn()`, which waits for a fresh human message before every action. This agent gets ONE goal up front and keeps going without further input until it's satisfied or capped out.
- **The agent decides when it's done** — nothing in this script tells the agent the goal is met. `response.stop_reason != "tool_use"` is checked after every call; the agent itself chooses to stop requesting tools once it judges it has enough.
- **Visible planning via `record_finding`** — a tool with no real computation behind it, whose only purpose is to make the agent's intermediate reasoning show up as tool calls in the transcript rather than staying implicit.
- **`MAX_STEPS` as an unconditional backstop** — same idea as `../../Execution_Loops/max_iterations/max_iterations.py`: autonomy without a hard cap is dangerous, since a confused agent could otherwise loop on tool calls indefinitely.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Core_Architecture/agent/agent.py
```

Try (or just press enter for the built-in default):

```
Goal: I want to buy 3 items priced at $12.50, $7.25, and $19.99. My budget is $50. Do they fit, and how much would I have left over?

[step 1] act: calculate({'expression': '12.50 + 7.25 + 19.99'})
[step 1] perceive: 12.50 + 7.25 + 19.99 = 39.74

[step 2] act: record_finding({'finding': 'total is 39.74, under the $50 budget'})
[step 2] perceive: Noted: total is 39.74, under the $50 budget

[step 3] agent finished on its own

Final answer: Yes, they fit under your $50 budget, with $10.26 left over.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../basics/README.md` for the first three
- `MAX_STEPS` — the hard cap on perceive-reason-act cycles (default: `8`)
- `TOOLS` / `execute_tool` — the `calculate` and `record_finding` tools available to the agent while it works

### See also

- `../basics/README.md` — the single request/response call this template's autonomy is contrasted against
- `../tool_use/README.md` — the underlying tool-calling loop mechanics (`TOOLS`, `execute_tool`, parallel calls), here repurposed for unattended goal pursuit instead of turn-by-turn chat
- `../llm_backbone/README.md` — swapping which model powers this same kind of loop
- `../../Execution_Loops/max_iterations/README.md` — the unconditional-cap pattern `MAX_STEPS` borrows
