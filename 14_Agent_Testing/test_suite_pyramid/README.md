# test_suite_pyramid

Layering unit, integration, adversarial, and LLM-judge/live tests by cost and speed into one CI-style pipeline: fast deterministic tests on every commit, expensive LLM-judge/live tests on a schedule, with pass-rate/latency/cost tracked over time as the suite and agent both grow.

## test_suite_pyramid.py

A pytest-style test module that layers a small representative test registry (`LAYERS`) into a wide, cheap "fast" base (unit/integration/adversarial — miniatures of the other three templates in this directory) and a narrow, expensive "slow" top (one real LLM-judge call, gated behind `RUN_SLOW_TESTS=1`), and runs/reports on them together. Run with `pytest test_suite_pyramid.py` or directly with `python3 test_suite_pyramid.py`.

### Concepts covered

- **`LAYERS`** — the registry: each entry has a `tier` (`"fast"` or `"slow"`) and a list of test callables; a real project populates this by importing its actual fast test modules plus `../../13_Benchmarking/`-style scoring templates as the slow tier.
- **`run_pyramid(layers, run_slow)`** — the gate: slow-tier layers only execute when `run_slow=True`, otherwise they're recorded as skipped at zero cost — the exact mechanism that lets a CI system run fast tiers on every push and slow tiers on a schedule from the same registry.
- **`test_llm_judge_refund_explanation_quality`** — the one genuinely live test in this directory: a real Claude call, scored the same structured-output way as `../../13_Benchmarking/llm_judge_benchmarking/llm_judge_benchmarking.py`'s `evaluate_output`, skipped with a printed reason if `RUN_SLOW_TESTS` or `ANTHROPIC_API_KEY` isn't set.
- **`print_pyramid_report(results)`** — prints the layers bottom (fast/wide) to top (slow/narrow), the shape the pyramid is named for, with pass counts and per-layer timing.
- **`test_slow_tier_skipped_by_default` / `test_slow_tier_runs_when_requested`** — meta-tests proving the gate itself works, using a throwaway layer whose test would fail loudly if it ran when it shouldn't.

### Run

From the repo root:

```bash
pip install -r requirements.txt
python3 14_Agent_Testing/test_suite_pyramid/test_suite_pyramid.py
# or, to also run the live LLM-judge tier:
export ANTHROPIC_API_KEY=your-key-here
RUN_SLOW_TESTS=1 python3 14_Agent_Testing/test_suite_pyramid/test_suite_pyramid.py
```

Try:

```
  ^  slow / expensive / narrow
  |  [slow] llm_judge (live) SKIPPED (tier=slow, 1 test(s) not run — set RUN_SLOW_TESTS=1)
  |  [fast] adversarial      2/2 passed in 0.0ms
  |  [fast] integration      1/1 passed in 0.0ms
  |  [fast] unit             2/2 passed in 0.0ms
  v  fast / cheap / wide
```

### Configuration

- `PASS_THRESHOLD` — the live judge's pass/fail cutoff (default: `7` out of 10), same convention as `../../13_Benchmarking/llm_judge_benchmarking/`
- `RUN_SLOW_TESTS` (env var) — set to `1` to execute the slow tier; unset or any other value skips it
- `LAYERS` — add a new tier or test by appending to this registry

### See also

- `../tool_unit_testing/README.md`, `../agent_integration_testing/README.md`, `../adversarial_safety_testing/README.md` — the full-size fast tiers this file's `LAYERS` entries are miniatures of
- `../../13_Benchmarking/README.md` — the scoring templates (`task_accuracy_eval`, `llm_judge_benchmarking`, `trace_evaluation`) a real project's slow tier would be built from
