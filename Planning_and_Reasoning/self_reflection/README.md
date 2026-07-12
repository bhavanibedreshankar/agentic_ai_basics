# self_reflection

Self-Reflection / Critique — the agent reviews and critiques its own output before finalizing it, revising iteratively until it's good enough.

## self_reflection.py

Writes a short professional email, then critiques and revises it in a loop (up to `MAX_ROUNDS` times) until an exacting critic approves it or the round limit is hit. Type `exit` to quit.

### Concepts covered

- **Iterative, not fixed** — contrast with `../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py`'s `edit_draft`, which runs exactly one editing pass regardless of draft quality. Here, `self_reflect` loops: critique → revise → critique again, as many times as issues keep turning up (bounded by `MAX_ROUNDS`).
- **A model-judged stopping condition** — contrast with `../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py`'s `validate_outline`, a deterministic Python check with no LLM call. Here, whether to stop is itself an LLM judgment (`critique` deciding "APPROVED" or not) — more flexible for nuanced criteria like tone, but less predictable, which is why `MAX_ROUNDS` exists as a hard backstop.
- **`is_approved`'s strictness** — only an exact `"APPROVED"` counts; a hedge like *"Approved, but the closing feels abrupt"* is correctly treated as **not** approved, since that "but" is still actionable feedback.
- **Contrast with `../tree_of_thought/`** — ToT generates several independent candidates and picks the best; self-reflection takes a *single* candidate and improves it in place.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Planning_and_Reasoning/self_reflection/self_reflection.py
```

Try:

```
Request: Ask my manager for two extra days off next month for a family event, keeping my current project on track.

--- Initial draft ---
...
--- Round 1 critique ---
- Doesn't specify which project or how it stays on track
- Closing is abrupt
--- Round 1 revision ---
...
--- Round 2 critique ---
APPROVED

(approved — no further revision needed)
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `MAX_ROUNDS` — see `../../Core_Architecture/basics/README.md` for the first three; `MAX_ROUNDS` caps revision rounds (default: `3`)
- `DRAFT_SYSTEM_PROMPT` / `CRITIC_SYSTEM_PROMPT` / `REVISE_SYSTEM_PROMPT` — the three focused instructions driving each phase

### See also

- `../../Agent_Frameworks_and_Patterns/prompt_chaining/README.md` — the single-pass alternative this template's loop generalizes
- `../tree_of_thought/README.md` — improving quality by comparing multiple candidates instead of revising one
- `../../Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — a standalone, reusable version of this template's `critique` step, usable outside a revise loop with any task/output/rubric
- `../../Agent_Frameworks_and_Patterns/README.md` — this template is conceptually the "Reflection Loop" pattern in that directory's five-pattern set; it stays here since it's already integrated into `Planning_and_Reasoning/`'s own comparison table
