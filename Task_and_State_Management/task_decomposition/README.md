# task_decomposition

Task Decomposition — breaking a complex goal into smaller, manageable subtasks, as its own inspectable artifact rather than fused into an execution pipeline.

## task_decomposition.py

Decomposes a broad goal ("launch a podcast") into a task tree, then flattens it into an execution-ready checklist. Type `exit` to quit.

### Concepts covered

- **Decomposition as a standalone step** — contrast with `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`, which decomposes and immediately executes in one fused pipeline. Here, `decompose()` returns a `TaskNode` tree you can print, inspect, and flatten independently of running anything.
- **A genuine multi-level tree, not a flat list** — a subtask can itself have sub-subtasks if it's still too broad; `plan_and_execute`'s plan is always exactly one level.
- **Structured output, and a real limitation it works around** — the decomposition is returned as JSON Schema output, not parsed from numbered-list text. Claude's structured outputs don't support genuinely recursive schemas, so `TASK_SCHEMA` hardcodes a fixed two-level shape instead of unbounded depth — the module docstring explains the workaround (call `decompose()` again on any subtask still too broad, for real unbounded recursion).
- **`TaskNode.leaves()`** — flattens the tree into only the directly-actionable subtasks; a node with children is a grouping label, not something to execute.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Task_and_State_Management/task_decomposition/task_decomposition.py
```

Try:

```
Goal: Launch a weekly podcast about local history.

=== Task tree ===
- Launch a weekly podcast about local history
  - Plan content
    - Pick 12 episode topics
    - Write a pilot script
  - Set up recording equipment
  - Choose a hosting platform

=== Flattened checklist (4 executable subtasks) ===
  1. Pick 12 episode topics
  2. Write a pilot script
  3. Set up recording equipment
  4. Choose a hosting platform
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `DECOMPOSE_SYSTEM_PROMPT` — the instruction controlling how deep and how broad the breakdown goes

### See also

- `../../Planning_and_Reasoning/plan_and_execute/README.md` — the contrasting fused decompose-and-execute pattern
- `../checkpointing/README.md` — what happens after decomposition, when a flattened step list needs to survive an interruption
