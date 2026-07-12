# evaluator_agent

Evaluator Agent — an agent (or separate LLM call) that scores or validates another agent's output, built as a reusable utility rather than logic baked into one specific workflow.

## evaluator_agent.py

`evaluate_output(task, output, rubric)` scores two deliberately unrelated kinds of output — a text summary and a SQL query — using the identical function with different rubrics supplied at call time. Type `exit` to quit after the two demos run, or supply your own task/output/rubric to score.

### Concepts covered

- **A general utility, not a one-off** — every other scoring/validation logic in the repo is embedded in the one workflow it serves: `../../Multi_Agent_Systems/supervisor_pattern/supervisor_pattern.py`'s `validate_output` is deterministic and shape-specific; `../../Planning_and_Reasoning/tree_of_thought/tree_of_thought.py`'s `evaluate_branch` and `../../Planning_and_Reasoning/self_reflection/self_reflection.py`'s `critique` are each hardcoded to their one use case. Here, `task`, `output`, and `rubric` are all caller-supplied arguments — verified directly that the same function correctly scores a summarization task and a SQL-safety task without any cross-contamination between the two calls' prompts.
- **Structured output, not free text to parse a verdict from** — the evaluator returns `{score, feedback}` via `EVALUATION_SCHEMA`; there's no "did it say PASS or FAIL somewhere in this paragraph" parsing.
- **Pass/fail is a code decision, not a model one** — `EvaluationResult.passed` is computed in Python (`score >= PASS_THRESHOLD`), not left to the model to declare. Verified at the exact boundary: a score equal to `PASS_THRESHOLD` passes, one point below fails.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py
```

You'll see both built-in demos run automatically:

```
=== Demo 1: evaluating a text summary ===
[summary output]
Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct.
[evaluation] score: 9/10, passed: True
  feedback: Mentions oxygen as required and stays under the word limit.

=== Demo 2: evaluating a SQL query ===
[SQL output]
DELETE FROM users WHERE inactive = true;
[evaluation] score: 10/10, passed: True
  feedback: Correctly scoped to inactive users only, targets the right table.
```

Then you can supply your own `(task, output, rubric)` triple to score.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `PASS_THRESHOLD` — the score cutoff for `passed` (default: `7` out of 10)
- `EVALUATOR_SYSTEM_PROMPT` — the instruction shaping how the evaluator judges anything it's given

### See also

- `../../Multi_Agent_Systems/supervisor_pattern/README.md` — the deterministic-validation alternative for when criteria are checkable in code rather than needing judgment
- `../../Planning_and_Reasoning/self_reflection/README.md` — an evaluator embedded in a generate-critique-revise loop, rather than a standalone reusable scorer
