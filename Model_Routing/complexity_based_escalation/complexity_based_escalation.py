"""
CONCEPT: Complexity-based escalation -- instead of deciding a model tier
BEFORE the first call (as both ../task_classifier_router/task_classifier_router.py
and ../cost_aware_model_selection/cost_aware_model_selection.py do), always
try the CHEAPEST model first, look at whether it says it was actually
confident in its own answer, and only pay for a stronger model when the
cheap one admits it wasn't sure.

This is an ADAPTIVE, retry-based strategy rather than an upfront one:
  - task_classifier_router.py spends a real (if cheap) extra API call to
    judge complexity before ever answering, so it never wastes a full
    Opus-tier answer on a misjudged-simple question -- but it also never
    gets a "second opinion" if its one-shot classification turns out to
    be wrong.
  - cost_aware_model_selection.py judges complexity for free from surface
    text features, with the same one-shot limitation.
  - This template judges nothing about the QUESTION up front. It commits
    to trying the cheapest tier's actual answer, and lets THAT model's own
    self-reported confidence decide whether to escalate. On the common
    case (the cheap model nails it), this costs exactly one call, same as
    just always using the cheap model -- the expensive tiers are ONLY
    ever paid for on the harder slice of requests that genuinely need them.

Self-reported confidence comes back as structured output (a JSON schema
with a boolean field), not a string marker inside the answer text to
search for -- the same "never free text to parse" principle as
../../Agent_Frameworks_and_Patterns/router_agent/router_agent.py's
CLASSIFY_SCHEMA, applied here to a confidence flag instead of a category.

Use case: a general assistant that always tries Haiku first, escalates
to Sonnet if Haiku isn't confident, and escalates once more to Opus if
even Sonnet isn't confident. Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# CONCEPT: the escalation ladder, cheapest first. Walked in order until a
# tier reports confidence or the ladder runs out.
ESCALATION_LADDER = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-4-8",
]

SYSTEM_PROMPT = (
    "Answer the user's question as best you can. Set confident to false "
    "if you are genuinely unsure you can answer correctly, or if the "
    "question needs deeper reasoning than you can reliably do in one "
    "pass -- still give your best attempt in 'answer' either way, but "
    "don't claim confidence you don't have."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
    },
    "required": ["answer", "confident"],
    "additionalProperties": False,
}


def ask_tier(model: str, question: str) -> tuple[str, bool]:
    """One tier's attempt at the question. Returns (answer, confident)."""
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    return result["answer"], result["confident"]


def ask_with_escalation(question: str) -> tuple[str, str, int]:
    """Walk the ladder from cheapest to most capable, stopping at the
    first tier that's confident (or the last tier, whatever it says).
    Returns (answer, model_used, tiers_tried).
    """
    for i, model in enumerate(ESCALATION_LADDER):
        answer, confident = ask_tier(model, question)
        is_last_tier = i == len(ESCALATION_LADDER) - 1
        if confident or is_last_tier:
            if i > 0:
                print(f"  [escalation] settled on tier {i + 1}/{len(ESCALATION_LADDER)}: {model}")
            return answer, model, i + 1
        print(f"  [escalation] {model} not confident -- escalating to next tier")
    raise AssertionError("unreachable: loop always returns on the last tier")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    print("Escalating assistant (Haiku -> Sonnet -> Opus). Type 'exit' to quit.")
    print('Try: "What color is the sky?" (should stop at Haiku) then')
    print('"Prove there are infinitely many prime numbers" (may escalate)\n')

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        answer, model, tiers_tried = ask_with_escalation(question)
        print(f"\nClaude ({model}, {tiers_tried} tier(s) tried): {answer}\n")


if __name__ == "__main__":
    main()
