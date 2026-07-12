# working_memory

Working memory — a scratchpad the agent uses mid-task to track intermediate state, cleared once the task is done.

## working_memory.py

An assistant that solves multi-step problems (e.g. tracking a compound-interest balance year by year) using an explicit `SCRATCHPAD` dict it manages itself via tools. Type `exit` to quit.

### Concepts covered

- **Task-scoped, not persistent** — the opposite design choice from every other template in `../`: `SCRATCHPAD` is never written to disk, and is explicitly cleared (`SCRATCHPAD.clear()`) at the start of every new task in `main()`, so nothing leaks between unrelated tasks.
- **Model-managed, not host-managed** — contrast with `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`, which also tracks intermediate step results, but as a plain Python list the *code* appends to. Here, the *model* decides what's worth recording and when, via `write_scratchpad` and `read_scratchpad` tool calls.
- **Why a scratchpad instead of just conversational context** — a structured dict is easy to inspect, resistant to a model losing track of a number across several steps, and (in a real system) could be dumped or restored to resume a task — none of which plain prose "remembering" in the conversation gives you for free.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Memory/working_memory/working_memory.py
```

Try:

```
Task: I invest $1000 at 5% annual compound interest. Track the balance at the end of each of the next 4 years in your scratchpad, then tell me the final balance.

  [tool] write_scratchpad({'key': 'balance_year_1', 'value': '1050.00'})
  [result] Scratchpad updated: balance_year_1 = 1050.00
...
Claude: After 4 years, the balance is $1215.51.
  [final scratchpad state: {'balance_year_1': '1050.00', 'balance_year_2': '1102.50', ...}]
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`

### See also

- `../../Planning_and_Reasoning/plan_and_execute/README.md` — the host-managed alternative to this template's model-managed scratchpad
- `../in_context_memory/README.md` — why relying purely on conversational context for intermediate state is fragile, which this template's explicit scratchpad avoids
