# Benchmarking

Five ways to measure an agent systematically instead of eyeballing a few transcripts: scoring against known-correct answers, scoring open-ended output with an LLM judge, measuring speed and cost, catching drift against a historical baseline, and comparing configurations against each other.

## Suggested reading order

| Order | Directory | What it covers |
|---|---|---|
| 1 | [`task_accuracy_eval/`](task_accuracy_eval/README.md) | Deterministic scoring — a fixed test suite with known-correct answers, matched by plain string/number comparison, no LLM judge involved |
| 2 | [`llm_judge_benchmarking/`](llm_judge_benchmarking/README.md) | Subjective scoring — a second LLM call grades open-ended outputs against a rubric, aggregated into pass rate and average score |
| 3 | [`latency_cost_benchmarking/`](latency_cost_benchmarking/README.md) | Performance, not correctness — timing requests and converting token usage into dollar cost across effort levels |
| 4 | [`regression_testing/`](regression_testing/README.md) | Comparison across TIME — checking current output against a stored golden baseline to catch drift, rather than against a fixed bar |
| 5 | [`model_prompt_comparison/`](model_prompt_comparison/README.md) | Comparison across CONFIGURATIONS — running the same suite through several models or prompts side by side to pick a winner |

## Setup

Same as the rest of the repo:

```bash
pip install -r ../requirements.txt   # or the root requirements.txt from the repo root
export ANTHROPIC_API_KEY=your-key-here
```

Run any template from the repo root, e.g.:

```bash
python3 Benchmarking/task_accuracy_eval/task_accuracy_eval.py
```

## How these relate to each other

| | Question it answers | Scoring mechanism | Compared against |
|---|---|---|---|
| `task_accuracy_eval/` | Does this configuration get objectively correct answers right? | Deterministic string/number match | A fixed answer key |
| `llm_judge_benchmarking/` | Does this configuration produce good open-ended output? | A second LLM call against a rubric | A fixed rubric |
| `latency_cost_benchmarking/` | How fast and how expensive is this configuration? | Wall-clock timing + token-to-cost conversion | No scoring — a speed/cost curve across effort levels |
| `regression_testing/` | Did this configuration's behavior change from before? | Word-overlap similarity | A stored historical snapshot |
| `model_prompt_comparison/` | Which of several configurations is best? | Deterministic match, run per configuration | Other configurations, not an absolute bar |

`task_accuracy_eval/` and `llm_judge_benchmarking/` are the two foundational scoring mechanisms everything else either reuses or deliberately avoids: exact-match for tasks with one correct answer, LLM-as-judge for tasks that don't have one. `latency_cost_benchmarking/` ignores scoring entirely and measures a different resource (time and money). `regression_testing/` and `model_prompt_comparison/` are both *comparisons* rather than absolute checks — the former across time (now vs. a saved snapshot), the latter across configurations (this prompt vs. that one) evaluated at the same moment.
