"""
CONCEPT: Regression Testing — comparing an agent's CURRENT outputs against
a stored "golden" baseline captured from a prior, known-good prompt or
model version, to catch unintended behavior changes before they ship.

This is a different question from every other template in this
directory. ../task_accuracy_eval/ and ../llm_judge_benchmarking/ both
ask "is this output good, measured against a fixed standard?" — this
template asks "did this output change from what it USED to be?", which
is a comparison across TIME (a stored snapshot vs. right now), not
against an absolute bar. A prompt tweak that makes answers slightly
worse but still individually "passable" on an accuracy or rubric check
can still be a real regression this template is built to catch.

The comparison itself can't be exact string equality — the SAME prompt
and system prompt, run twice, won't produce byte-identical text even
with nothing actually wrong (models paraphrase). `similarity()` is a
deliberately simple, dependency-free word-overlap ratio (Jaccard
similarity over lowercased word sets) — honestly a coarse proxy for "did
the meaning change," in the same spirit as
../../RAG_and_Knowledge/embedding/embedding_search.py's hand-rolled
`embed()`: good enough to demonstrate the MECHANIC (flag when current
output has drifted too far from the baseline), not a production-grade
semantic diff. A real regression suite would likely use embedding
similarity or a human/LLM-judge review pass — see
../llm_judge_benchmarking/llm_judge_benchmarking.py for that half of
the toolkit — layered on top of this same "compare to a stored
baseline" structure.

Demonstrated with a system-prompt regression that's easy to see clearly:
a baseline captured with a normal English-answering system prompt, then
a "regressed" version that responds in French — word overlap against
the English baseline collapses to near zero, exactly the kind of
regression this mechanic is built to surface. Baseline data persists in
baseline.json between runs, the same way ../../Memory/episodic_memory/'s
episodes.json does. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 256
EFFORT = "low"

# CONCEPT: below this similarity, current output is flagged as a
# regression against the baseline. Tunable — a stricter suite (e.g.
# scoring exact factual answers) would set this much higher than a suite
# of open-ended writing, where paraphrasing alone can legitimately drop
# word overlap.
SIMILARITY_THRESHOLD = 0.3

BASELINE_FILE = Path(__file__).parent / "baseline.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM_PROMPT_BASELINE = "Answer questions concisely, in English, in one or two sentences."
# CONCEPT: a deliberately broken "new" system prompt, standing in for an
# accidental prompt change that silently altered behavior — the kind of
# thing a regression suite exists to catch before it reaches production.
SYSTEM_PROMPT_REGRESSED = "Always respond only in French, regardless of what language the question is asked in."

TEST_CASES = [
    {"id": "capital", "prompt": "What is the capital of Japan?"},
    {"id": "api", "prompt": "What is an API, in one sentence?"},
    {"id": "exercise", "prompt": "Name one benefit of regular exercise."},
]


def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def similarity(a: str, b: str) -> float:
    """Jaccard similarity over word sets: |intersection| / |union|. Two
    empty strings are treated as identical (1.0); one empty and one not
    is treated as maximally different (0.0).
    """
    words_a, words_b = _tokenize(a), _tokenize(b)
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def generate(prompt: str, system_prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def capture_baseline(test_cases: list[dict], system_prompt: str) -> dict[str, str]:
    """CONCEPT: the golden snapshot step — run once, when the current
    behavior is known-good, and persist it. Everything after this point
    is measured AGAINST this snapshot, not against any absolute standard.
    """
    baseline = {case["id"]: generate(case["prompt"], system_prompt) for case in test_cases}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
    return baseline


def load_baseline() -> dict[str, str] | None:
    if not BASELINE_FILE.exists():
        return None
    return json.loads(BASELINE_FILE.read_text())


@dataclass
class RegressionResult:
    case_id: str
    baseline_output: str
    current_output: str
    similarity: float
    regressed: bool


def run_regression_check(
    test_cases: list[dict], system_prompt: str, baseline: dict[str, str], threshold: float = SIMILARITY_THRESHOLD
) -> list[RegressionResult]:
    results = []
    for case in test_cases:
        current_output = generate(case["prompt"], system_prompt)
        baseline_output = baseline.get(case["id"], "")
        sim = similarity(baseline_output, current_output)
        regressed = sim < threshold
        results.append(
            RegressionResult(
                case_id=case["id"], baseline_output=baseline_output, current_output=current_output, similarity=sim, regressed=regressed
            )
        )
        status = "REGRESSION" if regressed else "stable"
        print(f"  [{status}] {case['id']}: similarity={sim:.2f} (threshold={threshold})")
    return results


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Capturing golden baseline with the current (known-good) system prompt...\n")
    baseline = capture_baseline(TEST_CASES, SYSTEM_PROMPT_BASELINE)
    print(f"Baseline captured for {len(baseline)} cases and saved to {BASELINE_FILE.name}.\n")

    print("=== Checking against the SAME system prompt (expect mostly stable) ===")
    run_regression_check(TEST_CASES, SYSTEM_PROMPT_BASELINE, baseline)

    print("\n=== Checking against a REGRESSED system prompt (expect flagged regressions) ===")
    run_regression_check(TEST_CASES, SYSTEM_PROMPT_REGRESSED, baseline)

    print("\nNow try your own system prompt against the same baseline (type 'exit' to quit):")
    while True:
        candidate_prompt = input("\nSystem prompt to check: ").strip()
        if candidate_prompt.lower() == "exit":
            print("Goodbye!")
            break
        if not candidate_prompt:
            continue
        run_regression_check(TEST_CASES, candidate_prompt, baseline)


if __name__ == "__main__":
    main()
