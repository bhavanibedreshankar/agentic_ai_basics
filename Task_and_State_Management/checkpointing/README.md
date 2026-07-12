# checkpointing

Checkpointing — saving intermediate task state so a long-running agent can resume after failure, instead of starting over.

## checkpointing.py

A multi-step report-generation pipeline (outline → 3 sections → conclusion) that checkpoints progress to disk after every step. Interrupt it mid-run (Ctrl-C) and re-run with the same task id to resume exactly where it left off. Type `exit` at the task-id prompt to quit.

### Concepts covered

- **Persisting progress, not just data** — contrast with `../../Memory/episodic_memory/` and `../../Memory/semantic_memory/`, which persist facts or past interactions while the task producing them is expected to finish normally. This template persists an *in-flight, unfinished* task's progress, specifically so an interrupted run can continue rather than restart.
- **Save after every step, not just at the end** — `save_checkpoint` is called immediately after each step completes inside `run_pipeline`'s loop. Verified in testing: a simulated crash after 2 of 5 steps leaves exactly those 2 steps checkpointed on disk — no more, no less.
- **Resume by loading, then skipping what's done** — `run_pipeline` checks `if step in completed: skip` for every step; only steps missing from the checkpoint actually run. Verified that resuming after the simulated crash picks up precisely at the interrupted step (`section_2`), neither re-running the completed steps nor skipping ahead.
- **Contrast with `../../Planning_and_Reasoning/plan_and_execute/plan_and_execute.py`** — that template accumulates step results in a plain Python list, gone the moment the process exits, by design. This template takes the same "plan, then run steps in order accumulating results" shape and adds exactly the one thing plan_and_execute deliberately omits: durability across a crash.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Task_and_State_Management/checkpointing/checkpointing.py
```

Try interrupting a run partway through:

```
Task id (reuse the same id to resume a task): report1
Report topic: the history of the printing press

[outline] running...
...
[checkpointed: outline]

[section_1] running...
^C

[interrupted — progress up to the last completed step is saved under task id 'report1']
Run again with the same task id to resume.
```

Then run it again with the same task id:

```
Task id (reuse the same id to resume a task): report1
Report topic:
  [resuming task 'report1' — 1/5 steps already checkpointed: ['outline']]

[outline] already done (loaded from checkpoint) — skipping

[section_1] running...
...
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `CHECKPOINT_DIR` — where checkpoint files are written (default: `checkpoints/` next to the script)
- `STEPS` / `STEP_PROMPTS` — the fixed pipeline being checkpointed; swap in your own multi-step task

### See also

- `../task_decomposition/README.md` — generating the step list this template's fixed `STEPS` stands in for, for a task whose breakdown isn't known ahead of time
- `../../Planning_and_Reasoning/plan_and_execute/README.md` — the same step-by-step execution shape without durability
