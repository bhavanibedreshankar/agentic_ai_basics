# latency_cost_benchmarking

Measuring wall-clock time, token usage, and dollar cost per request across models or configurations to compare performance and efficiency trade-offs.

## latency_cost_benchmarking.py

Runs the same fixed prompt through all three `output_config.effort` levels (`low`, `medium`, `high`), timing each request with `time.perf_counter()` and converting token usage into an estimated dollar cost. Prints a comparison table of mean/median latency, mean cost, and total output tokens per level. Type `exit` to quit after the sweep runs, or benchmark your own prompt.

### Concepts covered

- **`timed_request(prompt, effort)`** — wraps the entire `client.messages.create` call in `time.perf_counter()`, measuring wall-clock latency including network round-trip, the number that actually matters to someone waiting on a response.
- **`estimate_cost(input_tokens, output_tokens)`** — the same pricing-to-cost conversion as `../../Core_Architecture/token_tracking/basic_token_tracking.py`'s `estimate_cost`, here applied per-run across a sweep instead of accumulated over one chat session.
- **A performance axis, not a correctness axis** — contrast with every other template in this directory: there is no scoring here at all. `../task_accuracy_eval/` and `../llm_judge_benchmarking/` ask "is this good?"; this template only asks "how much did it cost, in time and money?" for the identical prompt.
- **`EffortSummary`** — aggregates `RUNS_PER_LEVEL` timed runs per effort level into mean/median latency and mean cost, so the comparison reflects typical behavior rather than one noisy sample.

### Run

From the repo root:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 Benchmarking/latency_cost_benchmarking/latency_cost_benchmarking.py
```

Try:

```
  [low] run 1/2: 1.42s, 118 output tokens, $0.00177
  [low] run 2/2: 1.38s, 122 output tokens, $0.00183
  [medium] run 1/2: 2.51s, 210 output tokens, $0.00315
  [medium] run 2/2: 2.60s, 198 output tokens, $0.00297
  [high] run 1/2: 4.87s, 380 output tokens, $0.00570
  [high] run 2/2: 5.02s, 402 output tokens, $0.00603

Effort    Mean latency    Median latency    Mean cost     Total out tokens
low       1.40            1.40              $0.00180      240
medium    2.55            2.55              $0.00306      408
high      4.94            4.94              $0.00586      782
```

### Configuration

- `MODEL`, `MAX_TOKENS` — see `../../Core_Architecture/basics/README.md`
- `EFFORT_LEVELS` — which effort levels to sweep (default: `["low", "medium", "high"]`)
- `RUNS_PER_LEVEL` — how many timed runs to average per level (default: `2`)
- `PRICE_PER_MILLION_INPUT` / `PRICE_PER_MILLION_OUTPUT` — approximate USD-per-million-token rates; check platform.claude.com/pricing before relying on this for real budgeting

### See also

- `../../Core_Architecture/token_tracking/README.md` — the same usage-to-cost conversion, accumulated per session instead of swept per effort level
- `../model_prompt_comparison/README.md` — the quality-axis counterpart: sweeps configurations for correctness instead of effort for speed/cost
