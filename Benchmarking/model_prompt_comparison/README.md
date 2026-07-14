# model_prompt_comparison

Running the identical test suite through multiple models or prompt variants side by side to decide which configuration performs best.

## model_prompt_comparison.py

Runs the same 4-question test suite through two system-prompt configurations (`terse` and `detailed`, both on the same model), scores each answer by keyword containment, and prints a side-by-side pass-rate and token-usage comparison. Type `exit` to quit after the comparison runs, or press enter to re-run it.

### Concepts covered

- **`CONFIGS`** — each entry is a complete configuration (`model` + `system`), not just a prompt string; swap `model` per entry (e.g. Sonnet vs. Haiku) to compare models exactly the same way this template compares prompts.
- **`score(output, expected_keywords)`** — the same deterministic, zero-API-call scoring idea as `../task_accuracy_eval/task_accuracy_eval.py`'s `score_answer`, applied once per configuration instead of once total.
- **`run_comparison(configs, suite)`** — answers "which configuration is better?", a genuinely different question from `../task_accuracy_eval/README.md`'s "does this one configuration clear the bar?"
- **Config isolation** — `run_config` always uses *that config's own* `model` and `system`, verified directly so configurations never cross-contaminate each other's calls.
- **A quality axis, not a speed axis** — contrast with `../latency_cost_benchmarking/README.md`: that template holds the configuration fixed and sweeps `effort` for a speed/cost curve; this template holds `effort` fixed and sweeps configurations for a correctness curve.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Benchmarking/model_prompt_comparison/model_prompt_comparison.py
```

Try:

```
  [terse] [PASS] capital: 'Canberra.'
  [terse] [FAIL] author: 'A British writer.'
  [terse] [PASS] boiling_point: '100°C.'
  [terse] [PASS] largest_ocean: 'The Pacific.'
  [detailed] [PASS] capital: 'The capital of Australia is Canberra, chosen as a compromise...'
  [detailed] [PASS] author: '1984 was written by George Orwell, published in 1949.'
  [detailed] [PASS] boiling_point: 'Water boils at 100°C (212°F) at standard sea-level pressure.'
  [detailed] [PASS] largest_ocean: 'The Pacific Ocean is the largest, covering about a third of Earth's surface.'

Config      Pass rate     Total output tokens
terse       75%           24
detailed    100%          86
```

Notice the trade-off directly in the table: `detailed` scores higher here, but at roughly 3-4x the output tokens — the kind of concrete trade-off this template exists to surface before picking a configuration to ship.

### Configuration

- `MODEL`, `MAX_TOKENS`, `EFFORT` — see `../../Core_Architecture/basics/README.md`
- `CONFIGS` — the configurations under comparison; add a `model` override per entry to compare models instead of prompts
- `TEST_SUITE` — the shared test cases every configuration is scored against

### See also

- `../task_accuracy_eval/README.md` — the same scoring style, applied to one fixed configuration instead of a comparison
- `../latency_cost_benchmarking/README.md` — the orthogonal speed/cost axis, sweeping `effort` instead of configuration
