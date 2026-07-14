"""
CONCEPT: Model and Prompt Comparison — running the IDENTICAL test suite
through multiple models or prompt variants side by side, to decide which
configuration to actually ship, rather than evaluating just one.

Contrast with ../task_accuracy_eval/task_accuracy_eval.py: that template
runs ONE fixed system prompt and model once, to answer "does this
configuration clear the bar?" This template runs the SAME suite through
SEVERAL configurations to answer a different question — "which of these
configurations is better?" — a comparison, not a pass/fail check against
an absolute standard. The scoring mechanic itself (keyword-containment
against each task's expected answer) is the same deterministic idea as
../task_accuracy_eval/'s `score_answer`, just applied once per config
instead of once total.

Also contrast with ../latency_cost_benchmarking/latency_cost_benchmarking.py:
that template holds the prompt and model FIXED and sweeps `effort` to
characterize a SPEED/COST curve — no scoring involved. This template
holds `effort` fixed and sweeps CONFIGURATIONS (different system prompts
here; swap in a different `model` per entry to compare models the same
way) to characterize a QUALITY curve instead. They're deliberately
orthogonal axes of "benchmarking": one is about how fast/cheap an answer
is, the other is about how good it is.

The default comparison uses two system-prompt variants on the SAME model
— guaranteed to run for anyone with basic API access, unlike a multi-
model comparison, which can 404 if the caller's key doesn't have access
to every model in the list. `CONFIGS` shows exactly where a real `model`
swap would go if you wanted to compare Sonnet vs. Haiku, for example,
instead of (or in addition to) prompt variants.

Use case: comparing a terse system prompt against a detailed one over a
small suite of short-answer questions, to see which one gets more
answers objectively right. Type 'exit' to quit after the comparison
runs.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: each entry is a full configuration to compare — its own model
# AND system prompt. To compare models instead of prompts, change `model`
# per entry and keep `system` fixed; both axes vary the same way.
CONFIGS = [
    {"name": "terse", "model": MODEL, "system": "Answer in one short sentence. No explanation, no hedging."},
    {"name": "detailed", "model": MODEL, "system": "Answer thoroughly, with brief reasoning and context."},
]

# CONCEPT: deliberately the same style of deterministic scoring as
# ../task_accuracy_eval/task_accuracy_eval.py — expected_keywords are
# words that MUST appear (case-insensitive) somewhere in a correct
# answer, regardless of how verbose or terse the surrounding text is.
TEST_SUITE = [
    {"id": "capital", "prompt": "What is the capital of Australia?", "expected_keywords": ["canberra"]},
    {"id": "author", "prompt": "Who wrote the novel '1984'?", "expected_keywords": ["orwell"]},
    {"id": "boiling_point", "prompt": "At what temperature (Celsius) does water boil at sea level?", "expected_keywords": ["100"]},
    {"id": "largest_ocean", "prompt": "What is the largest ocean on Earth?", "expected_keywords": ["pacific"]},
]


@dataclass
class TaskOutcome:
    task_id: str
    output: str
    output_tokens: int
    passed: bool


@dataclass
class ConfigReport:
    name: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.passed for o in self.outcomes) / len(self.outcomes)

    @property
    def total_output_tokens(self) -> int:
        return sum(o.output_tokens for o in self.outcomes)


def score(output: str, expected_keywords: list[str]) -> bool:
    normalized = re.sub(r"[^\w\s]", "", output.lower())
    return any(keyword.lower() in normalized for keyword in expected_keywords)


def generate_answer(prompt: str, model: str, system: str) -> tuple[str, int]:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.output_tokens


def run_config(config: dict, suite: list[dict]) -> ConfigReport:
    report = ConfigReport(name=config["name"])
    for task in suite:
        output, output_tokens = generate_answer(task["prompt"], config["model"], config["system"])
        passed = score(output, task["expected_keywords"])
        report.outcomes.append(TaskOutcome(task_id=task["id"], output=output, output_tokens=output_tokens, passed=passed))
        status = "PASS" if passed else "FAIL"
        print(f"  [{config['name']}] [{status}] {task['id']}: {output[:80]!r}")
    return report


def run_comparison(configs: list[dict], suite: list[dict]) -> list[ConfigReport]:
    return [run_config(config, suite) for config in configs]


def print_comparison_table(reports: list[ConfigReport]) -> None:
    print(f"\n{'Config':<12}{'Pass rate':<14}{'Total output tokens'}")
    for r in reports:
        print(f"{r.name:<12}{r.pass_rate:<14.0%}{r.total_output_tokens}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing {len(CONFIGS)} configurations over {len(TEST_SUITE)} test cases.\n")
    reports = run_comparison(CONFIGS, TEST_SUITE)
    print_comparison_table(reports)

    print("\nType 'exit' to quit, or press enter to re-run the comparison.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        reports = run_comparison(CONFIGS, TEST_SUITE)
        print_comparison_table(reports)


if __name__ == "__main__":
    main()
