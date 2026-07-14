# task_accuracy_eval

Running a fixed suite of test cases with known expected answers through an agent and scoring exact/partial match to measure correctness objectively.

## task_accuracy_eval.py

Runs 5 factual/arithmetic/short-answer questions with known-correct answers through Claude, scores each with deterministic string or number matching (no LLM judge involved), and prints a pass rate. Type `exit` to quit after the suite runs, or check your own question against a substring you supply.

### Concepts covered

- **`TASK_SUITE`** — each case declares its own `match` strategy (`exact`, `numeric`, `any_contains`) because "correct" means something different for a capital city than for "17 * 6" or "name a primary color."
- **`score_answer(task, response_text)`** — pure, deterministic Python: no API call, always returns the same verdict for the same input. Contrast with `../llm_judge_benchmarking/README.md`, where the judge is itself a model call and therefore not perfectly reproducible.
- **`run_benchmark(suite)`** / **`BenchmarkReport.pass_rate`** — the harness that runs an entire suite in one pass and reduces it to a single aggregate number, which is what distinguishes *benchmarking* from the single-shot scoring in `../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py`.
- **`_normalize(text)`** — lowercases and strips punctuation so `"Paris."` and `"paris"` compare equal; exact-match scoring shouldn't penalize formatting differences a model reliably introduces.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Benchmarking/task_accuracy_eval/task_accuracy_eval.py
```

Try:

```
  [PASS] capital: expected='paris' got='Paris.'
  [PASS] arithmetic: expected='102' got='102'
  [PASS] color: expected=['red', 'blue', 'yellow'] got='Blue.'
  [PASS] planet_count: expected='8' got='8'
  [PASS] language: expected='portuguese' got='Portuguese'

Pass rate: 100% (5/5)
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT`, `SYSTEM_PROMPT` — see `../../Core_Architecture/basics/README.md`
- `TASK_SUITE` — the test cases; add a case with a `match` strategy of `exact`, `numeric`, or `any_contains`

### See also

- `../llm_judge_benchmarking/README.md` — the subjective counterpart, for tasks with no single correct string
- `../model_prompt_comparison/README.md` — the same scoring idea applied across multiple configurations instead of one
- `../../Agent_Frameworks_and_Patterns/evaluator_agent/README.md` — a single-call LLM-judge utility, contrasted with this template's zero-API-call scoring
