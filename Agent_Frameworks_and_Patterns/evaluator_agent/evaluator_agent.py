"""
CONCEPT: Evaluator Agent — a separate LLM call that scores or validates
another agent's output against a rubric, built as a REUSABLE utility
rather than logic baked into one specific use case.

Every other place in the repo that scores or validates an output has
that logic embedded in the ONE workflow it serves:
  - `../../Multi_Agent_Systems/supervisor_pattern/supervisor_pattern.py`'s
    `validate_output` is deterministic Python (JSON parsing, field
    checks) — no LLM judgment involved, and it only ever validates one
    specific shape of output (an extraction result).
  - `../../Planning_and_Reasoning/tree_of_thought/tree_of_thought.py`'s
    `evaluate_branch` IS an LLM call, but it's written specifically to
    score math/logic solutions against "is this correct" — it isn't
    meant to be reused for scoring, say, a summary's tone.
  - `../../Planning_and_Reasoning/self_reflection/self_reflection.py`'s
    `critique` is similarly single-purpose, hardcoded to email-quality
    criteria.

This template's `evaluate_output` takes the TASK, the OUTPUT, and a
RUBRIC as three separate arguments — the rubric is supplied by the
CALLER, not hardcoded into the evaluator — so the exact same function
scores two completely different kinds of output later in this file: a
text summary judged for conciseness and accuracy, and a SQL query judged
for correctness and safety. Nothing about `evaluate_output` itself knows
or cares which one it's looking at.

The evaluation itself is structured output (score, passed, feedback) —
not free text you'd have to parse a verdict out of — with `passed`
computed in Python against a fixed `PASS_THRESHOLD`, not left to the
model's own judgment of what "good enough" means.

Type 'exit' to quit; the demo runs both example evaluations, plus lets
you supply your own (task, output, rubric) triple to score.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"
PASS_THRESHOLD = 7  # out of 10

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

EVALUATOR_SYSTEM_PROMPT = (
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


@dataclass
class EvaluationResult:
    score: int
    passed: bool
    feedback: str


def evaluate_output(task: str, output: str, rubric: str) -> EvaluationResult:
    """CONCEPT: the reusable evaluator itself. Nothing here is specific
    to any one kind of task — `task`, `output`, and `rubric` are all
    supplied by the caller, which is what makes this a general-purpose
    utility instead of one workflow's built-in quality check.
    """
    prompt = f"Task: {task}\n\nOutput to evaluate:\n{output}\n\nRubric:\n{rubric}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=EVALUATOR_SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": EVALUATION_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    score = result["score"]
    # CONCEPT: pass/fail is a CODE decision against a fixed threshold,
    # not left to the model to declare "this passes" — the model only
    # ever reports a score and feedback; PASS_THRESHOLD is the one place
    # that decides what "good enough" means.
    return EvaluationResult(score=score, passed=score >= PASS_THRESHOLD, feedback=result["feedback"])


def generate(task: str, system_prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": task}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Two demo scenarios, deliberately unrelated in domain, to prove
# evaluate_output() is genuinely reused rather than shaped around one
# use case.
# ---------------------------------------------------------------------------
def demo_summary_evaluation() -> None:
    task = "Summarize the following in one sentence: Photosynthesis is the process by which plants convert light energy, usually from the sun, into chemical energy stored in glucose, using carbon dioxide and water as inputs and releasing oxygen as a byproduct."
    rubric = "The summary must be a single sentence, must mention that oxygen is a byproduct, and must not exceed 25 words."

    output = generate(task, "Summarize text in exactly one sentence.")
    print(f"\n[summary output]\n{output}")

    result = evaluate_output(task, output, rubric)
    print(f"[evaluation] score: {result.score}/10, passed: {result.passed}\n  feedback: {result.feedback}")


def demo_sql_evaluation() -> None:
    task = "Write a SQL query that deletes all rows from the 'users' table where the 'inactive' column is true."
    rubric = "The query must only affect rows where inactive = true (never an unconditional DELETE), and must target the 'users' table specifically."

    output = generate(task, "Write a single SQL query. Return only the query, no explanation.")
    print(f"\n[SQL output]\n{output}")

    result = evaluate_output(task, output, rubric)
    print(f"[evaluation] score: {result.score}/10, passed: {result.passed}\n  feedback: {result.feedback}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Evaluator agent demo — the SAME evaluate_output() scores two unrelated kinds of output.\n")

    print("=== Demo 1: evaluating a text summary ===")
    demo_summary_evaluation()

    print("\n=== Demo 2: evaluating a SQL query ===")
    demo_sql_evaluation()

    print("\n=== Now try your own (type 'exit' at any prompt to quit) ===")
    while True:
        task = input("\nTask: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue
        output = input("Output to evaluate: ").strip()
        rubric = input("Rubric: ").strip()
        if not output or not rubric:
            continue

        result = evaluate_output(task, output, rubric)
        print(f"score: {result.score}/10, passed: {result.passed}\nfeedback: {result.feedback}")


if __name__ == "__main__":
    main()
