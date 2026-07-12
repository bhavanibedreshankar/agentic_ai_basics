# orchestrator

Orchestrator — a high-level agent that breaks a goal into subtasks and delegates each one to a specialized sub-agent, rather than doing the work itself.

## orchestrator.py

A content-team orchestrator managing three specialists: a researcher, a writer, and an editor. Type `exit` to quit.

### Concepts covered

- **Specialists, not interchangeable steps** — contrast with `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`: that template executes every plan step with the same generic executor. Here, each subtask goes to a *different* agent with its own system prompt and persona — `delegate_to_researcher`, `delegate_to_writer`, and `delegate_to_editor` are three distinct calls, not one function reused three times.
- **Sub-agents as tools** — each specialist is declared in `ORCHESTRATOR_TOOLS` and dispatched through `execute_tool`, exactly like any other tool in this repo. Calling a specialist is indistinguishable, mechanically, from calling a calculator or a file editor — the tool's *implementation* just happens to be another full API call under a different system prompt.
- **No fixed sequence** — `ORCHESTRATOR_SYSTEM_PROMPT` tells the model to decide which specialists to use and in what order; nothing in the code hardcodes "always research, then write, then edit."

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Multi_Agent_Systems/orchestrator/orchestrator.py
```

Try:

```
You: Write a short paragraph about the history of coffee, researched and polished.

  [delegating to researcher] {'topic': 'history of coffee'}
  [researcher returned] - Coffee was first discovered in Ethiopia...
  [delegating to writer] {'brief': '...'}
  [writer returned] Coffee's story begins in the Ethiopian highlands...
  [delegating to editor] {'draft': '...'}
  [editor returned] Coffee's story begins...

Orchestrator: Here's your polished paragraph: ...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `ORCHESTRATOR_SYSTEM_PROMPT` — the instruction establishing "delegate, don't do it yourself"
- Each `delegate_to_X` function's own `system` prompt — the specialist's persona

### See also

- `../worker_agent/README.md` — what a delegate looks like when it needs its own tools and internal loop, rather than a single-shot call
- `../supervisor_pattern/README.md` — an orchestrator variant that validates delegated output instead of trusting it
