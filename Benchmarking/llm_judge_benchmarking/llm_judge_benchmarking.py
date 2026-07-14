"""
CONCEPT: LLM-as-Judge Benchmarking — using a separate LLM call to grade a
BATCH of open-ended agent outputs against a rubric, then aggregating the
per-item scores into whole-suite statistics (pass rate, average score).

This is the SUBJECTIVE half of benchmarking. Contrast with
../task_accuracy_eval/task_accuracy_eval.py: that template's tasks each
have one objectively correct short answer, scored with plain string/number
comparison and zero API calls for scoring. The tasks here — "summarize
this paragraph", "write a decline email", "explain recursion simply" —
have no single correct string; two genuinely good answers can share
almost no words. There's no deterministic function that can score them,
so the judge itself has to be another LLM call, reading a rubric and
returning a structured verdict.

The single-item judging mechanic (`evaluate_output`) is the same idea as
../../Agent_Frameworks_and_Patterns/evaluator_agent/evaluator_agent.py's
`evaluate_output` — task, output, and rubric as caller-supplied arguments,
scored via structured output so there's no free text to parse a verdict
out of. What's new here is everything AROUND that one call: `run_benchmark`
loops it over an entire SUITE and reduces the individual scores into
`pass_rate` and `average_score` — the two numbers you'd actually track
release over release to know whether a prompt or model change made things
better or worse.

One real limitation worth being explicit about: the judge is ITSELF a
model call, so unlike ../task_accuracy_eval/'s scoring, running the exact
same (task, output, rubric) twice is not guaranteed to produce the exact
same score — LLM-judge benchmarks are noisier than exact-match benchmarks,
which is a real tradeoff for gaining the ability to grade open-ended work
at all.

Use case: a small suite of open-ended writing/explanation tasks, each
judged against its own rubric. Type 'exit' to quit after the suite runs,
or supply your own (task, rubric) pair to benchmark live.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, field

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"
PASS_THRESHOLD = 7  # out of 10 — same convention as evaluator_agent.py

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. Given a task, an output "
    "produced for that task, and a rubric to judge it against, score the "
    "output from 0 to 10 on how well it meets the rubric, and explain "
    "your score in one or two sentences of specific, actionable "
    "feedback — not vague praise or criticism."
)

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "0 to 10"},
        "feedback": {"type": "string"},
    },
    "required": ["score", "feedback"],
    "additionalProperties": False,
}

# CONCEPT: the benchmark suite — each case is an open-ended task paired
# with its OWN rubric, since "good" means something different for a
# summary than for an email. There is no "expected" field like in
# ../task_accuracy_eval/'s TASK_SUITE, because there's no single string
# that would count as correct here.
BENCHMARK_SUITE = [
    {
        "id": "summary",
        "task": "Summarize in one sentence: Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose, using carbon dioxide and water as inputs and releasing oxygen as a byproduct.",
        "rubric": "Must be a single sentence, must mention oxygen as a byproduct, and must not exceed 25 words.",
    },
    {
        "id": "decline_email",
        "task": "Write a short, polite email declining a job offer, without giving a specific reason.",
        "rubric": "Must be polite and professional, must clearly decline, must not fabricate or imply a specific reason for declining.",
    },
    {
        "id": "explain_recursion",
        "task": "Explain what recursion is to a 10-year-old, using an analogy.",
        "rubric": "Must use a concrete, age-appropriate analogy (not just technical jargon), must be understandable without programming background, must stay under 80 words.",
    },
]


@dataclass
class EvaluationResult:
    score: int
    passed: bool
    feedback: str


@dataclass
class JudgedCase:
    case_id: str
    output: str
    result: EvaluationResult


@dataclass
class BenchmarkReport:
    cases: list[JudgedCase] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.result.passed for c in self.cases) / len(self.cases)

    @property
    def average_score(self) -> float:
        if not self.cases:
            return 0.0
        return statistics.mean(c.result.score for c in self.cases)


def generate_output(task: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": task}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def evaluate_output(task: str, output: str, rubric: str) -> EvaluationResult:
    """CONCEPT: same shape as evaluator_agent.py's evaluate_output — task,
    output, and rubric are all caller-supplied, and `passed` is a CODE
    decision against PASS_THRESHOLD, never left to the model to declare.
    """
    prompt = f"Task: {task}\n\nOutput to evaluate:\n{output}\n\nRubric:\n{rubric}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=JUDGE_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": EVALUATION_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    score = result["score"]
    return EvaluationResult(score=score, passed=score >= PASS_THRESHOLD, feedback=result["feedback"])


def run_benchmark(suite: list[dict]) -> BenchmarkReport:
    report = BenchmarkReport()
    for case in suite:
        output = generate_output(case["task"])
        result = evaluate_output(case["task"], output, case["rubric"])
        report.cases.append(JudgedCase(case_id=case["id"], output=output, result=result))
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {case['id']}: score={result.score}/10 — {result.feedback}")
    return report


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print(f"Running LLM-as-judge benchmark — {len(BENCHMARK_SUITE)} open-ended cases, each with its own rubric.\n")
    report = run_benchmark(BENCHMARK_SUITE)
    print(
        f"\nPass rate: {report.pass_rate:.0%} "
        f"({sum(c.result.passed for c in report.cases)}/{len(report.cases)})  "
        f"Average score: {report.average_score:.1f}/10"
    )

    print("\nNow try your own (task, rubric) pair (type 'exit' to quit):")
    while True:
        task = input("\nTask: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue
        rubric = input("Rubric: ").strip()
        if not rubric:
            continue
        output = generate_output(task)
        print(f"\n[output]\n{output}")
        result = evaluate_output(task, output, rubric)
        print(f"[{'PASS' if result.passed else 'FAIL'}] score={result.score}/10 — {result.feedback}")


if __name__ == "__main__":
    main()
