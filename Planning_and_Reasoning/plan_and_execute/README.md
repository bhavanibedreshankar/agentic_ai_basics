# plan_and_execute

Plan-and-Execute — a two-phase pattern: first generate a high-level plan tailored to the task, then execute each step in sequence using prior results as context.

## plan_and_execute.py

Takes a broad task, plans it into 3–5 concrete steps, then executes each step one at a time. Demonstrated on a compare-and-recommend task (remote vs. office work policy). Type `exit` to quit.

### Concepts covered

- **A dynamically generated plan, not a hardcoded sequence** — contrast with `../../prompt_chaining/basic_prompt_chaining.py`, which always runs the same three fixed steps (outline → draft → edit). Here, `generate_plan` produces a *different* plan depending on the task — a simple request might get 3 steps, a complex one more.
- **Structured output for the plan** — `generate_plan` uses `output_config.format` with a JSON Schema to get a clean, reliably-parseable list of steps instead of regex-ing a free-text numbered list.
- **`execute_step` as a scoped sub-agent** — each step's call receives the overall task, the full plan for context, and every prior step's result, but its job is narrowly focused on producing *this* step's output only. Real Plan-and-Execute systems often give each step's sub-agent its own tools; this template keeps execution tool-free to isolate the plan/execute mechanic on its own.
- **Sequential accumulation** — `plan_and_execute` runs steps in order, passing growing `results` into each subsequent step, so later steps can build on earlier ones.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Planning_and_Reasoning/plan_and_execute/plan_and_execute.py
```

Try:

```
Task: Compare the pros and cons of remote work vs. office work and recommend a policy.

Plan:
  1. List the pros of remote work
  2. List the cons of remote work
  3. List the pros and cons of office work
  4. Synthesize a recommendation based on the above

[1/4] Executing: List the pros of remote work
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../basics/README.md`
- `PLAN_SYSTEM_PROMPT` / `PLAN_SCHEMA` — controls how the plan is generated and its shape
- `execute_step`'s system prompt — controls how narrowly each step focuses on its own job

### See also

- `../../prompt_chaining/README.md` — the fixed-sequence alternative this template's dynamic plan generalizes
- `../../Tools_and_Actions/tool_use/README.md` — how to extend `execute_step` with its own tools per step
