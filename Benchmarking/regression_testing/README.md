# regression_testing

Comparing an agent's current outputs against a stored "golden" baseline from a prior prompt or model version to catch unintended behavior changes.

## regression_testing.py

Captures a golden baseline for 3 test questions using a known-good system prompt, then checks two candidate system prompts against that baseline: the same prompt again (expected to stay stable) and a deliberately "regressed" prompt that responds in French instead of English (expected to be flagged). Baseline data persists in `baseline.json` between runs. Type `exit` to quit.

### Concepts covered

- **`capture_baseline(test_cases, system_prompt)`** — the golden-snapshot step, run once when current behavior is known-good; everything after this is measured *against this snapshot*, not against any absolute standard — the key difference from `../task_accuracy_eval/README.md` and `../llm_judge_benchmarking/README.md`, which both score against a fixed bar.
- **`similarity(a, b)`** — a dependency-free Jaccard word-overlap ratio, honestly a coarse proxy for "did the meaning change," in the same spirit as `../../RAG_and_Knowledge/embedding/embedding_search.py`'s hand-rolled `embed()`. Two outputs can legitimately paraphrase each other and still count as non-regressed; the threshold is tunable.
- **`run_regression_check(test_cases, system_prompt, baseline, threshold)`** — regenerates each test case under a *candidate* prompt and flags a regression when similarity to the stored baseline drops below `SIMILARITY_THRESHOLD`.
- **`baseline.json` persistence** — same pattern as `../../Memory/episodic_memory/episodes.json`: data written by one run is available again on the next, so a baseline captured today can be checked against weeks later.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Benchmarking/regression_testing/regression_testing.py
```

Try:

```
Capturing golden baseline with the current (known-good) system prompt...

Baseline captured for 3 cases and saved to baseline.json.

=== Checking against the SAME system prompt (expect mostly stable) ===
  [stable] capital: similarity=0.83 (threshold=0.3)
  [stable] api: similarity=0.71 (threshold=0.3)
  [stable] exercise: similarity=0.75 (threshold=0.3)

=== Checking against a REGRESSED system prompt (expect flagged regressions) ===
  [REGRESSION] capital: similarity=0.00 (threshold=0.3)
  [REGRESSION] api: similarity=0.00 (threshold=0.3)
  [REGRESSION] exercise: similarity=0.00 (threshold=0.3)
```

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `SIMILARITY_THRESHOLD` — below this, current output is flagged as a regression (default: `0.3`)
- `TEST_CASES` — the fixed prompts checked against the baseline on every run
- `baseline.json` — the persisted golden snapshot; delete it to force a fresh capture on the next run

### See also

- `../llm_judge_benchmarking/README.md` — a rubric-based way to catch quality drift, layerable on top of this same "compare to a baseline" structure
- `../../Memory/episodic_memory/README.md` — another template that persists run data to a JSON file and reloads it across processes
