"""
CONCEPT: Latency and Cost Benchmarking — measuring wall-clock time, token
usage, and dollar cost per request across configurations, to compare
performance and efficiency trade-offs rather than output quality.

This is a different axis from every other template in this directory.
../task_accuracy_eval/ and ../llm_judge_benchmarking/ both ask "is the
output CORRECT/GOOD?" — this template asks "how much does getting an
answer actually COST, in time and money?" for the exact same prompt. It
sweeps `output_config.effort` (low / medium / high) — which trades
thinking depth for speed and token spend — and shows the trade-off
directly, using the same usage-to-cost conversion as
../../Core_Architecture/token_tracking/basic_token_tracking.py's
`estimate_cost`, just aggregated across many timed runs instead of one
running session total.

Also contrast with ../model_prompt_comparison/model_prompt_comparison.py:
that template sweeps DIFFERENT prompts or models to see which one
produces BETTER answers on a scored test suite. This template holds the
prompt and model FIXED and sweeps only `effort`, purely to characterize
the speed/cost curve — there's no scoring here at all, because
correctness isn't the question being asked.

Timing measures wall-clock latency with `time.perf_counter()` around
each `client.messages.create` call — this includes network round-trip
time, not just server-side generation time, which is the number that
actually matters to a user waiting on a response.

Use case: benchmarking how a single fixed prompt performs across the
three effort levels, with several runs per level to see typical latency
variance, not just one noisy sample. Type 'exit' to quit after the sweep
runs, or benchmark your own prompt.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT_LEVELS = ["low", "medium", "high"]
RUNS_PER_LEVEL = 2  # more runs = a more reliable latency picture, but more cost/time to benchmark

# CONCEPT: same pricing-to-cost conversion as
# ../../Core_Architecture/token_tracking/basic_token_tracking.py's
# PRICE_PER_MILLION_INPUT/OUTPUT — approximate standard rates in USD per
# million tokens; check platform.claude.com/pricing for current numbers
# before relying on this for real budgeting.
PRICE_PER_MILLION_INPUT = 3.00
PRICE_PER_MILLION_OUTPUT = 15.00

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

BENCHMARK_PROMPT = (
    "Explain the difference between a list and a tuple in Python, and "
    "give one example of when you'd choose each."
)


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT + (output_tokens / 1_000_000) * PRICE_PER_MILLION_OUTPUT


@dataclass
class RunResult:
    effort: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class EffortSummary:
    effort: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def mean_latency(self) -> float:
        return statistics.mean(r.latency_s for r in self.runs)

    @property
    def median_latency(self) -> float:
        return statistics.median(r.latency_s for r in self.runs)

    @property
    def mean_cost(self) -> float:
        return statistics.mean(r.cost for r in self.runs)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.runs)


def timed_request(prompt: str, effort: str) -> RunResult:
    """CONCEPT: the timer wraps the ENTIRE API call, not just a server-
    reported duration — perf_counter() measures wall-clock time from this
    process's perspective, which includes network latency, exactly the
    number a real caller waiting on a response would experience.
    """
    start = time.perf_counter()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = time.perf_counter() - start
    usage = response.usage
    cost = estimate_cost(usage.input_tokens, usage.output_tokens)
    return RunResult(effort=effort, latency_s=latency_s, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cost=cost)


def run_benchmark(prompt: str, effort_levels: list[str] = EFFORT_LEVELS, runs_per_level: int = RUNS_PER_LEVEL) -> list[EffortSummary]:
    summaries = []
    for effort in effort_levels:
        summary = EffortSummary(effort=effort)
        for i in range(runs_per_level):
            result = timed_request(prompt, effort)
            summary.runs.append(result)
            print(f"  [{effort}] run {i + 1}/{runs_per_level}: {result.latency_s:.2f}s, {result.output_tokens} output tokens, ${result.cost:.5f}")
        summaries.append(summary)
    return summaries


def print_comparison_table(summaries: list[EffortSummary]) -> None:
    print(f"\n{'Effort':<10}{'Mean latency':<16}{'Median latency':<18}{'Mean cost':<14}{'Total out tokens'}")
    for s in summaries:
        print(f"{s.effort:<10}{s.mean_latency:<16.2f}{s.median_latency:<18.2f}${s.mean_cost:<13.5f}{s.total_output_tokens}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking latency and cost across effort levels {EFFORT_LEVELS} ({RUNS_PER_LEVEL} runs each).\n")
    print(f"Prompt: {BENCHMARK_PROMPT!r}\n")
    summaries = run_benchmark(BENCHMARK_PROMPT)
    print_comparison_table(summaries)

    print("\nNow benchmark your own prompt across the same effort levels (type 'exit' to quit):")
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        if not prompt:
            continue
        summaries = run_benchmark(prompt)
        print_comparison_table(summaries)


if __name__ == "__main__":
    main()
