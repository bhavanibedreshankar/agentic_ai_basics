"""
CONCEPT: Guardrails — rules or classifiers that prevent the agent from
taking harmful or out-of-scope actions, enforced automatically by CODE,
with no human in the loop for the common case.

This is a different mechanism from two things elsewhere in the repo that
can look similar at a glance:
  - `../../Execution_Loops/interrupts_breakpoints/interrupts_breakpoints.py`
    PAUSES and asks a human when a condition is met — the action might
    still happen, just after approval. Guardrails here REJECT outright,
    automatically, no human involved (a guardrail *could* escalate to a
    human for edge cases, but the default posture is "block and explain
    why," not "pause and ask").
  - `../../Task_and_State_Management/state_machine/state_machine.py`
    checks whether a transition is LEGAL given the current state.
    Guardrails check something orthogonal: whether the actual CONTENT of
    an input or the CONTENT of a requested action is safe/in-scope,
    independent of any state machine — an order in a perfectly legal
    state can still request a refund amount that violates policy.

Two guardrail layers, checked in TWO different places:
  1. INPUT guardrail — `check_input_guardrail` screens the user's raw
     message BEFORE it's ever sent to the model, for patterns that look
     like prompt injection or attempts to extract the system prompt.
     Rejected input never reaches the API at all — no tokens spent, no
     model turn happens.
  2. ACTION guardrail — `check_action_guardrail` screens a requested tool
     call's ARGUMENTS against a policy BEFORE dispatch (a refund above
     a hard cap), the same "check before mutation" placement used by
     `state_machine.py`, just checking a business rule instead of state
     legality.

Use case: a customer support agent that can look up accounts and issue
refunds, guarded against both prompt-injection-shaped input and
policy-violating refund amounts. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import re
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a customer support agent. Use lookup_account to check "
    "account details and issue_refund to process refunds. Only issue "
    "refunds for legitimate, verified account issues."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

ACCOUNTS = {
    "A100": {"name": "Jane Doe", "balance_owed": 45.00},
    "A200": {"name": "Sam Lee", "balance_owed": 12.50},
}

MAX_REFUND_AMOUNT = 100.00

# ---------------------------------------------------------------------------
# CONCEPT: input guardrail — a classifier (here, a small set of regex
# patterns; a production system would likely use a real classifier
# model) checked against RAW USER TEXT before it's ever sent to the main
# agent. This is the only guardrail layer in this file that can prevent
# a model call from happening at all.
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous|prior|above) instructions",
    r"reveal (your |the )?system prompt",
    r"you are now (in )?(dan|developer|jailbreak) mode",
    r"disregard (your |all )?(guidelines|rules|policy)",
]


def check_input_guardrail(user_text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Checked BEFORE any API call — a
    rejected input costs nothing and never reaches the model.
    """
    lowered = user_text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Input blocked: matched a prompt-injection pattern ({pattern!r})."
    return True, "ok"


# ---------------------------------------------------------------------------
# CONCEPT: action guardrail — a policy check on a requested tool call's
# ARGUMENTS, run right before that tool would execute. The tool itself
# (issue_refund) is fully available to the model; this guardrail only
# blocks specific calls that violate the policy, and explains why so
# the model can propose something that would pass instead.
# ---------------------------------------------------------------------------
def check_action_guardrail(tool_name: str, tool_input: dict) -> tuple[bool, str]:
    if tool_name == "issue_refund":
        amount = tool_input.get("amount", 0)
        if amount > MAX_REFUND_AMOUNT:
            return False, f"Refund of ${amount:.2f} exceeds the maximum allowed refund of ${MAX_REFUND_AMOUNT:.2f}."
    return True, "ok"


def lookup_account(account_id: str) -> str:
    account = ACCOUNTS.get(account_id)
    if not account:
        return f"No account found for ID {account_id}."
    return f"Account {account_id}: {account['name']}, balance owed: ${account['balance_owed']:.2f}"


def issue_refund(account_id: str, amount: float) -> str:
    if account_id not in ACCOUNTS:
        return f"Error: no account found for ID {account_id}."
    return f"Refund of ${amount:.2f} issued to account {account_id}."


TOOLS = [
    {
        "name": "lookup_account",
        "description": "Look up an account's details by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund to an account.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "amount": {"type": "number", "description": "Refund amount in dollars"},
            },
            "required": ["account_id", "amount"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "lookup_account":
        return lookup_account(**tool_input), False
    if name == "issue_refund":
        return issue_refund(**tool_input), False
    return f"Unknown tool: {name}", True


def run_turn(messages: list[dict]) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # CONCEPT: the action guardrail check, placed before
                # dispatch — same placement pattern as
                # ../../Task_and_State_Management/state_machine.py's
                # legality check, enforcing a policy rule instead.
                allowed, reason = check_action_guardrail(block.name, block.input)
                if not allowed:
                    print(f"  [GUARDRAIL BLOCKED] {block.name}({block.input}) — {reason}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Action blocked by guardrail: {reason}",
                            "is_error": True,
                        }
                    )
                    continue

                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                print(f"  [result] {result_text}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Customer support agent (guardrails demo). Type 'exit' to quit.\n")
    print("Try: \"Refund account A100 $500\" (blocked by the action guardrail)\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        # CONCEPT: the input guardrail check — before this message is
        # even appended to `messages`, let alone sent to the API.
        is_safe, reason = check_input_guardrail(user_input)
        if not is_safe:
            print(f"  [INPUT GUARDRAIL BLOCKED] {reason}\n")
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
