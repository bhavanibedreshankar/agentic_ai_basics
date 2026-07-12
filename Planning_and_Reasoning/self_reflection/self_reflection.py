"""
CONCEPT: Self-Reflection / Critique — the agent reviews and critiques its
OWN output before finalizing it, revising based on that critique, and
repeating until the output is good enough (or a round limit is reached).

Compare this with ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's
edit_draft step: that template runs exactly ONE fixed editing pass, no
matter how good or bad the draft was. Self-reflection is ITERATIVE —
critique and revise repeat in a loop, and the number of rounds isn't
fixed in advance; it depends on how many issues keep turning up. It also
differs from ../tree_of_thought/tree_of_thought.py: ToT evaluates several
independent candidate answers and picks the best one; self-reflection
takes a SINGLE answer and improves it in place through repeated critique.

The stopping condition here is a MODEL judgment, not deterministic code —
contrast with ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's
validate_outline, which is a plain Python check with no LLM call
involved. Asking the model "is this good enough now?" is more flexible
(it can judge nuanced things like tone or clarity that code can't check)
but less predictable than a hard rule — this template caps it with
MAX_ROUNDS so a critic that never quite says "approved" can't loop
forever.

Demonstrated on writing a short, professional email — a task where a
critique pass genuinely tends to catch real issues (missing information,
unclear asks, tone). Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
MAX_ROUNDS = 3

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

DRAFT_SYSTEM_PROMPT = "Write a short, professional email for the given request. Return only the email."

CRITIC_SYSTEM_PROMPT = (
    "You are an exacting editor reviewing an email draft against four "
    "criteria: clarity, correctness (does it actually accomplish what "
    "was asked), tone (professional and appropriate), and conciseness. "
    "If the draft is genuinely strong on all four, respond with EXACTLY "
    "the single word: APPROVED. Otherwise, list the specific issues you "
    "found, one per line, each concrete enough that someone could fix it "
    "without guessing what you meant."
)

REVISE_SYSTEM_PROMPT = (
    "Revise the email draft to address every issue in the feedback. Keep "
    "everything that already works well. Return only the revised email."
)


def generate_draft(task: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=DRAFT_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": task}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def critique(task: str, draft: str) -> str:
    """CONCEPT: a focused, separate call whose only job is judging the
    current draft — same "narrow, single-purpose step" idea as
    ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's steps, but this one's
    output feeds back into ANOTHER call (revise) rather than forward into
    the next stage of a fixed pipeline.
    """
    prompt = f"Original request: {task}\n\nCurrent draft:\n{draft}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=CRITIC_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def revise(task: str, draft: str, feedback: str) -> str:
    prompt = f"Original request: {task}\n\nCurrent draft:\n{draft}\n\nFeedback to address:\n{feedback}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=REVISE_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def is_approved(feedback: str) -> bool:
    """Deliberately strict — only an exact "APPROVED" (ignoring case and
    surrounding whitespace/punctuation) counts. A critic that hedges
    ("Approved, but...") should NOT short-circuit the loop; that "but" is
    feedback that deserves another revision round.
    """
    return feedback.strip().strip(".").upper() == "APPROVED"


def self_reflect(task: str, max_rounds: int = MAX_ROUNDS) -> str:
    draft = generate_draft(task)
    print(f"\n--- Initial draft ---\n{draft}")

    for round_num in range(1, max_rounds + 1):
        feedback = critique(task, draft)
        print(f"\n--- Round {round_num} critique ---\n{feedback}")

        if is_approved(feedback):
            print("\n(approved — no further revision needed)")
            break

        draft = revise(task, draft, feedback)
        print(f"\n--- Round {round_num} revision ---\n{draft}")
    else:
        print(f"\n(reached MAX_ROUNDS={max_rounds} without explicit approval — returning latest draft)")

    return draft


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Self-reflection / critique demo. Type 'exit' to quit.\n")
    print(
        "Try: \"Ask my manager for two extra days off next month for a "
        "family event, keeping my current project on track.\"\n"
    )

    while True:
        task = input("Request: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue

        final_draft = self_reflect(task)
        print(f"\n=== Final draft ===\n{final_draft}\n")


if __name__ == "__main__":
    main()
