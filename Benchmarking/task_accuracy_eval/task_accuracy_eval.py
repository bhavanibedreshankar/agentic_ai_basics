"""
CONCEPT: Task Accuracy Evaluation — running a fixed suite of test cases
with KNOWN, OBJECTIVELY CORRECT answers through an agent and scoring each
one by exact or partial match, rather than by anyone's subjective
judgment (human or model).

This is the DETERMINISTIC half of benchmarking. Contrast with
../llm_judge_benchmarking/llm_judge_benchmarking.py, which handles the
opposite case — tasks like "summarize this" or "write an apology email"
that have no single correct string to match, so scoring needs an LLM's
judgment against a rubric instead. Here, `score_answer()` is plain Python
string/number comparison; it never calls the API, and given the same
(task, response) pair it always returns the same verdict — that
determinism is exactly what makes exact-match scoring trustworthy for
the tasks it's suited to (facts, arithmetic, short structured answers),
and exactly why it CAN'T be used for open-ended writing tasks, where two
correct answers can share zero words.

Also contrast with ../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py:
that template scores ONE (task, output, rubric) triple per call, as a
reusable utility invoked from wherever it's needed. This template is the
other axis — it's a HARNESS that runs an entire SUITE of cases in one
pass and aggregates the results into a single pass-rate number, which is
what "benchmarking" (as opposed to one-off evaluation) actually means.

Use case: a small suite of factual, arithmetic, and short-answer
questions with known expected answers — the kind of test set you'd run
after every prompt or model change to catch a correctness regression
before it ships. Results are printed per-task and as an aggregate
summary; type 'exit' to quit after the suite runs, or ask an ad hoc
question to see the same scoring machinery applied live.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 256  # these tasks all expect short answers
EFFORT = "low"
SYSTEM_PROMPT = (
    "Answer the question as briefly as possible — a single word, number, "
    "or short phrase. No explanation, no extra sentences."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: the test suite itself is the benchmark's real content — the
# code below is just the harness that runs it. Each case declares its own
# `match` strategy because "correct" means something different for a
# capital city (exact string) than for "name a primary color" (any of
# several acceptable answers) or "17 * 6" (a number, however it's phrased).
# ---------------------------------------------------------------------------
TASK_SUITE = [
    {"id": "capital", "prompt": "What is the capital of France?", "expected": "paris", "match": "exact"},
    {"id": "arithmetic", "prompt": "What is 17 * 6?", "expected": "102", "match": "numeric"},
    {"id": "color", "prompt": "Name one primary color.", "expected": ["red", "blue", "yellow"], "match": "any_contains"},
    {"id": "planet_count", "prompt": "How many planets are in our solar system?", "expected": "8", "match": "numeric"},
    {"id": "language", "prompt": "What language is primarily spoken in Brazil?", "expected": "portuguese", "match": "exact"},
]


@dataclass
class TaskResult:
    task_id: str
    prompt: str
    expected: object
    actual: str
    passed: bool


@dataclass
class BenchmarkReport:
    results: list[TaskResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation/whitespace so 'Paris.' and 'paris'
    compare equal — models reliably vary formatting even when the
    underlying answer is identical, and exact-match scoring shouldn't
    penalize that.
    """
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def score_answer(task: dict, response_text: str) -> bool:
    """CONCEPT: the entire scoring step is deterministic Python — no API
    call, no model judgment. Given the same task and response_text this
    always returns the same bool, which is what makes it safe to run in
    an automated CI-style check without worrying about run-to-run
    variance in the JUDGE itself (unlike ../llm_judge_benchmarking/,
    where the judge is itself a model call and therefore not perfectly
    deterministic).
    """
    normalized = _normalize(response_text)
    match = task["match"]

    if match == "exact":
        return task["expected"] in normalized
    if match == "numeric":
        found = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        return task["expected"] in found
    if match == "any_contains":
        return any(_normalize(option) in normalized for option in task["expected"])
    raise ValueError(f"Unknown match strategy: {match}")


def generate_answer(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def run_benchmark(suite: list[dict]) -> BenchmarkReport:
    report = BenchmarkReport()
    for task in suite:
        actual = generate_answer(task["prompt"])
        passed = score_answer(task, actual)
        report.results.append(
            TaskResult(task_id=task["id"], prompt=task["prompt"], expected=task["expected"], actual=actual, passed=passed)
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {task['id']}: expected={task['expected']!r} got={actual!r}")
    return report


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Running task accuracy benchmark — {len(TASK_SUITE)} cases with known-correct answers.\n")
    report = run_benchmark(TASK_SUITE)
    print(f"\nPass rate: {report.pass_rate:.0%} ({sum(r.passed for r in report.results)}/{len(report.results)})")

    print("\nNow try your own question with a known expected answer (type 'exit' to quit):")
    while True:
        prompt = input("\nQuestion: ").strip()
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        if not prompt:
            continue
        expected = input("Expected answer (substring to look for): ").strip()
        if not expected:
            continue
        actual = generate_answer(prompt)
        passed = score_answer({"expected": expected.lower(), "match": "exact"}, actual)
        print(f"[{'PASS' if passed else 'FAIL'}] got={actual!r}")


if __name__ == "__main__":
    main()
