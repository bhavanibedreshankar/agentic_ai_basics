# worker_agent

Sub-agent / Worker Agent — a specialized agent that executes a specific subtask, packaged as a reusable, independently-callable unit.

## worker_agent.py

Two concrete workers — `data_analyst` (with a calculator tool) and `researcher` (with a mock fact-lookup tool) — each runnable directly from the CLI. Type `<worker_name>: <task>` to invoke one, or `exit` to quit.

### Concepts covered

- **`WorkerAgent`** — a class wrapping a system prompt, a tool set, and a tool executor into one `.run(task)` method. Internally it runs a complete tool-calling loop (structurally identical to `../../Tools_and_Actions/tool_use/basic_agentic_tools.py`'s `run_turn`) — however many turns that takes is invisible to the caller.
- **Composability** — this is what's actually running inside every `delegate_to_X` call in `../orchestrator/`, every specialist in `../agent_handoff/`, and every member of `../swarm/`: from the outside, a worker is just "call it with a task, get a result back," whether it takes one API call or five with its own tools in between.
- **Two workers sharing one abstraction** — `data_analyst` and `researcher` are both plain `WorkerAgent` instances with different `system_prompt`/`tools`/`tool_executor` arguments, demonstrating the class is a genuine reusable pattern, not a one-off.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Multi_Agent_Systems/worker_agent/worker_agent.py
```

Try:

```
worker_name: task > data_analyst: what's 15% of 340 plus 22

    [data_analyst thinking] I'll calculate 15% of 340 first.
    [data_analyst action] calculate({'expression': '340 * 0.15'})
    [data_analyst observation] 340 * 0.15 = 51.0
    ...

data_analyst final answer: 15% of 340 is 51, plus 22 is 73.
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `WORKERS` — the registry of available workers; add a new one by constructing another `WorkerAgent(...)` and adding it to the dict

### See also

- `../orchestrator/README.md` — the coordination layer that composes several of these workers together
- `../../Tools_and_Actions/tool_use/README.md` — the single-agent tool-calling loop `WorkerAgent.run()` is built on
