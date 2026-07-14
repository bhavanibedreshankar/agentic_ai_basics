# llm_judge_benchmarking

Using a separate LLM call to grade a batch of open-ended agent outputs against a rubric and aggregating pass rates and scores across a whole test suite.

## llm_judge_benchmarking.py

Runs 3 open-ended tasks (summarize, write a decline email, explain recursion) through Claude, then grades each output against its own rubric using a second, separate LLM call with structured output. Prints per-case verdicts and an aggregate pass rate plus average score. Type `exit` to quit after the suite runs, or supply your own (task, rubric) pair.

### Concepts covered

- **`evaluate_output(task, output, rubric)`** — the same shape as `../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py`'s `evaluate_output`: task/output/rubric are caller-supplied, scoring is structured output (`EVALUATION_SCHEMA`), and `passed` is a **code** decision (`score >= PASS_THRESHOLD`), never left to the model.
- **`run_benchmark(suite)`** / **`BenchmarkReport.pass_rate` / `.average_score`** — what's new versus `evaluator_agent.py`: this loops the single-item judge over an entire suite and reduces the results to the two numbers you'd track release over release.
- **Why exact-match scoring doesn't work here** — contrast with `../task_accuracy_eval/README.md`: "summarize this in one sentence" has no single correct string, so the judge has to be a model call reading a rubric, not a string comparison.
- **Judge non-determinism** — unlike `../task_accuracy_eval/`'s scoring, running the exact same (task, output, rubric) twice through `evaluate_output` isn't guaranteed to produce the same score, since the judge is itself an LLM call.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Benchmarking/llm_judge_benchmarking/llm_judge_benchmarking.py
```

Try:

```
  [PASS] summary: score=9/10 — Correctly mentions oxygen as a byproduct and stays under 25 words.
  [FAIL] decline_email: score=5/10 — Polite, but implies a specific reason ("timing") that wasn't asked for.
  [PASS] explain_recursion: score=8/10 — Uses a clear mirror-in-a-mirror analogy, stays concise.

Pass rate: 67% (2/3)  Average score: 7.3/10
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `PASS_THRESHOLD` — the score cutoff for `passed` (default: `7` out of 10)
- `BENCHMARK_SUITE` — the (task, rubric) pairs; add a case with any open-ended task and a rubric describing what a good answer looks like

### See also

- `../task_accuracy_eval/README.md` — the deterministic counterpart, for tasks with one objectively correct answer
- `../../Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — the single-call scoring primitive this template loops over a whole suite
- `../regression_testing/README.md` — another way to catch quality drift, by comparing against a stored baseline instead of a fixed rubric
