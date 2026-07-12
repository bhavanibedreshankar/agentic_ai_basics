"""
CONCEPT: Tree of Thought (ToT) — extending chain-of-thought from a single
reasoning path into several, explored in parallel, with the best one
selected at the end instead of committing to the first line of reasoning
that comes out.

../chain_of_thought/chain_of_thought.py's prompted CoT produces ONE
reasoning path per call — better than answering directly, but if that one
path takes a wrong turn early, the final answer inherits the error. Tree
of Thought's insight: generate SEVERAL independent reasoning paths (the
"branches"), evaluate each one, and pick the best — a wrong turn in one
branch doesn't sink the whole attempt if another branch got it right.

The full ToT technique (Yao et al., 2023) recurses this at EVERY
reasoning step — branch, evaluate, prune, expand the survivors, branch
again — searched with BFS or DFS, many levels deep. This template
implements a simplified, SINGLE-LEVEL version to demonstrate the core
mechanic clearly: generate N candidate approaches to a problem, evaluate
each one independently, and select the best — one branch/evaluate/select
round, not a multi-level search tree.

Rather than relying on randomness for the branches to actually be
different from each other (harder now that sampling parameters like
`temperature` are no longer configurable on current models — see
../../Core_Architecture/basics/README.md), each branch is generated with an explicit,
different STRATEGY instruction, guaranteeing the diversity ToT depends on
instead of hoping for it.

Demonstrated on a deliberately tricky problem — the classic "bat and
ball" puzzle, where a fast, intuitive answer is a well-known wrong answer
and a careful one gets it right. Generating multiple approaches and
checking them against each other is a good way to catch exactly that kind
of mistake. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: branch generation via explicit strategy diversity
# ---------------------------------------------------------------------------
# Each strategy nudges the model toward a genuinely different way of
# attacking the same problem, rather than three near-identical attempts.
BRANCH_STRATEGIES = [
    "Solve this using algebra — set up an equation and solve it.",
    "Solve this using careful logical reasoning, without writing any algebraic equations.",
    "Solve this by proposing a candidate answer and checking whether it satisfies every condition in the problem.",
]


def generate_branch(problem: str, strategy: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=(
            f"Solve the given problem. Approach: {strategy} Show your "
            "reasoning, then state your final answer on its own line as "
            "'Answer: ...'."
        ),
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": problem}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_branches(problem: str) -> list[str]:
    """Generate one branch per strategy — the "explore multiple reasoning
    branches in parallel" half of ToT. (Sequential API calls here for
    simplicity; a production version could run these concurrently.)
    """
    return [generate_branch(problem, strategy) for strategy in BRANCH_STRATEGIES]


# ---------------------------------------------------------------------------
# CONCEPT: evaluating each branch — a separate, focused call whose only
# job is judging a candidate solution, same idea as
# ../self_reflection/self_reflection.py's critique step, applied here to
# rank several candidates against each other instead of revising one.
# ---------------------------------------------------------------------------
EVALUATOR_SYSTEM_PROMPT = (
    "You are a careful math and logic checker. Given a problem and one "
    "proposed solution, verify the reasoning step by step and check "
    "whether the final answer actually satisfies every condition in the "
    "problem. Respond with ONLY a number from 0 to 10 (10 = correct and "
    "well-reasoned, 0 = wrong), followed by a one-sentence justification. "
    "Format: '<score>: <justification>'."
)


def evaluate_branch(problem: str, branch_text: str) -> tuple[int, str]:
    prompt = f"Problem: {problem}\n\nProposed solution:\n{branch_text}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=EVALUATOR_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    score_str, _, justification = text.partition(":")
    try:
        score = int("".join(ch for ch in score_str if ch.isdigit()))
    except ValueError:
        score = 0
    return score, justification.strip()


def select_best_branch(problem: str) -> tuple[str, list[tuple[str, int, str]]]:
    """Run the full round: branch, evaluate every branch, select the
    highest-scored one. Returns the winning branch's text, plus every
    branch with its score and justification for display.
    """
    branches = generate_branches(problem)
    scored = []
    for strategy, branch_text in zip(BRANCH_STRATEGIES, branches):
        score, justification = evaluate_branch(problem, branch_text)
        scored.append((branch_text, score, justification))

    best_text, best_score, _ = max(scored, key=lambda item: item[1])
    return best_text, scored


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Tree of Thought demo. Type 'exit' to quit.\n")
    print(
        "Try: \"A bat and a ball cost $1.10 in total. The bat costs $1.00 "
        "more than the ball. How much does the ball cost?\"\n"
    )

    while True:
        problem = input("Problem: ").strip()
        if problem.lower() == "exit":
            print("Goodbye!")
            break
        if not problem:
            continue

        best_text, scored = select_best_branch(problem)

        print("\n=== Branches ===")
        for i, (branch_text, score, justification) in enumerate(scored):
            print(f"\n[Branch {i + 1}] strategy: {BRANCH_STRATEGIES[i]}")
            print(branch_text)
            print(f"  -> score: {score}/10 ({justification})")

        print(f"\n=== Selected answer (highest-scored branch) ===\n{best_text}\n")


if __name__ == "__main__":
    main()
