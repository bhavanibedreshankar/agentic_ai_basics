# interrupts_breakpoints

Interrupts / Breakpoints — predefined conditions that pause execution and surface control to a human or supervisor, checked every loop iteration.

## interrupts_breakpoints.py

A research and file-management agent with three independent breakpoint conditions: a token-cost budget, a flagged action (`delete_file`), and a stuck-loop detector. Type `exit` to quit.

### Concepts covered

- **Conditions, not tool names** — contrast with `../human_in_the_loop/human_in_the_loop.py`, whose gate is keyed on which tool was called, every time, regardless of content. `check_breakpoints` here evaluates three independent conditions over the *running state* (`LoopState`) — a `delete_file` call always trips the ACTION breakpoint, but so can a perfectly ordinary `lookup_fact` call once the COST or PROGRESS conditions are met.
- **`LoopState`** — cumulative token usage and a rolling window of recent `(tool, args)` calls, tracked across the whole session, since a budget or a stall pattern is about accumulated behavior, not any single call.
- **Three breakpoint types, each independently verified**: COST (`total_tokens >= COST_BUDGET_TOKENS`), ACTION (`tool_name in ACTION_BREAKPOINTS`), and PROGRESS (the same call repeated `PROGRESS_STALL_THRESHOLD` times in a row) — tested separately and confirmed not to false-positive on distinct calls.
- **Both breakpoint outcomes verified through the real loop** — a full `run_turn` integration test proves that choosing "abort" at a breakpoint means `execute_tool` is *never* called, while choosing "continue" dispatches that exact call once approved.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Execution_Loops/interrupts_breakpoints/interrupts_breakpoints.py
```

Try:

```
You: Delete the file old_report.txt

  >>> BREAKPOINT HIT: ACTION breakpoint: 'delete_file' is flagged as requiring a pause before it runs, regardless of arguments.
      [c]ontinue this one call / [a]bort the task? c
  [tool] delete_file({'path': 'old_report.txt'})
  [result] (simulated) Deleted old_report.txt
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../basics/README.md`
- `COST_BUDGET_TOKENS`, `ACTION_BREAKPOINTS`, `PROGRESS_STALL_THRESHOLD` — the three breakpoint conditions; tune or add new ones in `check_breakpoints`

### See also

- `../human_in_the_loop/README.md` — the contrasting fixed-tool-name gate
- `../max_iterations/README.md` — an unconditional backstop for exactly the runaway-loop case a PROGRESS breakpoint might still miss
