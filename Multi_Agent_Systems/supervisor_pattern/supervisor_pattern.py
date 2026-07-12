"""
CONCEPT: Supervisor Pattern — an orchestrator that doesn't just delegate
and trust the result: it MONITORS the sub-agent's output, VALIDATES it
against explicit criteria, and RETRIES — with specific feedback about
what was wrong — if validation fails, up to a limit.

Contrast with ../orchestrator/: that template delegates and takes
whatever comes back at face value, with no check at all. Contrast with
../../Planning_and_Reasoning/self_reflection/: that template has ONE
agent critique and revise ITS OWN output, and the critic is another LLM
call judging quality/tone/clarity. Here, a SEPARATE supervisor validates
a DIFFERENT agent's (the worker's) output against DETERMINISTIC, coded
criteria — not a model's opinion — and retries the SAME worker with
corrective feedback rather than revising anything itself.

Use case: extracting structured data (name, email, amount owed) from a
support ticket. The worker is asked to return JSON; the supervisor
validates that output is actually valid JSON with the right fields and a
plausible email, and retries with specific feedback if not.

Type 'exit' to quit.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 512
EFFORT = "medium"
MAX_RETRIES = 3

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

WORKER_SYSTEM_PROMPT = (
    "Extract the customer name, email, and amount owed (as a number, no "
    "currency symbol) from the ticket text. Respond with ONLY a JSON "
    "object: {\"name\": ..., \"email\": ..., \"amount\": ...}. No other "
    "text, no markdown formatting."
)


def run_worker(ticket_text: str, feedback: str | None = None) -> str:
    """The delegated sub-agent — a single focused call, same shape as
    ../orchestrator/'s delegate_to_X functions. `feedback`, when present,
    is the supervisor's correction from a previous failed attempt.
    """
    prompt = ticket_text
    if feedback:
        prompt += f"\n\n(Your previous attempt had a problem: {feedback} Please try again.)"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=WORKER_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def validate_output(raw_output: str) -> tuple[bool, str]:
    """CONCEPT: deterministic validation, not another model call. Plain
    Python — parse the JSON, check required fields exist, do a trivial
    sanity check on the email. This is cheap, fast, and gives specific,
    actionable feedback the worker can act on, exactly the way
    ../../Agent_Frameworks_and_Patterns/prompt_chaining/basic_prompt_chaining.py's validate_outline
    gates a pipeline stage — applied here to a delegated agent's output
    instead of one step in a fixed chain.
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return False, "Output is not valid JSON."

    if not isinstance(data, dict):
        return False, "Output is not a JSON object."

    required = {"name", "email", "amount"}
    missing = required - data.keys()
    if missing:
        return False, f"Missing required field(s): {', '.join(sorted(missing))}."

    if "@" not in str(data.get("email", "")):
        return False, "The email field doesn't look like a valid email address."

    try:
        float(data["amount"])
    except (TypeError, ValueError):
        return False, "The amount field is not a number."

    return True, "Valid."


def supervise(ticket_text: str, max_retries: int = MAX_RETRIES) -> dict | None:
    """CONCEPT: the supervise/retry loop. Delegate, validate, and — on
    failure — delegate again with feedback about specifically what was
    wrong. Gives up and reports failure after max_retries rather than
    looping forever.
    """
    feedback = None
    for attempt in range(1, max_retries + 1):
        print(f"\n[attempt {attempt}/{max_retries}] delegating to worker...")
        raw_output = run_worker(ticket_text, feedback)
        print(f"  worker output: {raw_output}")

        is_valid, message = validate_output(raw_output)
        print(f"  [supervisor validation] {message}")

        if is_valid:
            return json.loads(raw_output)

        feedback = message

    print(f"  [supervisor] worker failed validation after {max_retries} attempts — giving up.")
    return None


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Supervisor pattern demo — structured extraction with validation and retry.")
    print("Type 'exit' to quit.\n")
    print(
        "Try: \"Hi, this is Jane Smith, jane.smith@example.com. I was billed "
        "$149.99 for a plan I cancelled last month.\"\n"
    )

    while True:
        ticket_text = input("Ticket text: ").strip()
        if ticket_text.lower() == "exit":
            print("Goodbye!")
            break
        if not ticket_text:
            continue

        result = supervise(ticket_text)
        if result is not None:
            print(f"\n=== Validated extraction ===\n{json.dumps(result, indent=2)}\n")
        else:
            print("\n=== Extraction failed validation — no result returned ===\n")


if __name__ == "__main__":
    main()
