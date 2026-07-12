"""
CONCEPT: Human-in-the-Loop (HITL) — a design where a human approves or
corrects an agent's action at key checkpoints, BEFORE that action
actually executes, rather than the agent acting fully autonomously.

Every tool-using template elsewhere in this repo executes a tool the
moment Claude requests it — `../../Core_Architecture/tool_use/basic_agentic_tools.py`'s
`run_turn` goes straight from "Claude called a tool" to "run the tool
and send back the result," with no pause in between. HITL inserts one
new step into that exact same loop: between "Claude requested this
action" and "the action actually runs," a human is asked to approve,
reject, or edit the request. This is a small change in shape — one
`if` before the existing dispatch — but it's the difference between an
agent that can (for example) send a real email or delete a real file
autonomously, and one that can only ever draft the action for a human
to confirm.

The line between "safe enough to auto-run" and "needs approval" is a
design decision, not a technical one — this template draws it by
REVERSIBILITY (reading data is auto-approved; anything that changes
state — sending, deleting — requires approval), which is a common and
sensible default, but the actual criteria always depend on the
application's real stakes.

Use case: an email assistant that can freely search a mock inbox, but
must get human approval before actually sending anything — and the human
can approve, reject with feedback, or edit the message before it goes
out. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are an email assistant. Use search_inbox freely to look things "
    "up. Use send_email to send messages — you'll be asked to wait for "
    "human approval before it actually sends, so don't be surprised by a "
    "delay or a request to revise."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# A tiny mock inbox — enough to make search_inbox return something real
# without needing an actual email account.
INBOX = [
    {"from": "sara@example.com", "subject": "Q3 budget review", "body": "Can we push the budget review to next Tuesday?"},
    {"from": "devteam@example.com", "subject": "Deploy window", "body": "Deploying to prod Friday 2-4pm, expect brief downtime."},
]

# CONCEPT: which tools require human approval before they run. This is
# the ENTIRE policy — everything else in the file just enforces it.
# `read`-shaped actions (search) are auto-approved; anything that sends
# or changes state requires a human to sign off first.
REQUIRES_APPROVAL = {"send_email"}


def search_inbox(query: str) -> str:
    matches = [e for e in INBOX if query.lower() in (e["subject"] + e["body"]).lower()]
    if not matches:
        return "No matching emails found."
    return "\n\n".join(f"From: {e['from']}\nSubject: {e['subject']}\n{e['body']}" for e in matches)


def send_email(to: str, subject: str, body: str) -> str:
    # A real implementation would call an email API here. Standing in
    # for that isn't the point of this template — the approval gate
    # around this call is.
    return f"Email sent to {to} — subject: '{subject}'"


TOOLS = [
    {
        "name": "search_inbox",
        "description": "Search the inbox for emails matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Text to search for in subject or body"}},
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email. Requires human approval before it actually sends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "search_inbox":
        return search_inbox(**tool_input), False
    if name == "send_email":
        return send_email(**tool_input), False
    return f"Unknown tool: {name}", True


def request_human_approval(tool_name: str, tool_input: dict) -> tuple[bool, dict]:
    """CONCEPT: the actual HITL checkpoint. Blocks on real terminal
    input — nothing about the loop continues until a human responds.
    Three outcomes: approve as-is, reject with feedback (the tool never
    runs; the feedback becomes the tool_result so Claude can react to
    it), or edit specific fields before it runs. Returns
    (approved, possibly-modified tool_input).
    """
    print(f"\n  >>> APPROVAL NEEDED: {tool_name}({tool_input})")
    decision = input("      [a]pprove / [r]eject / [e]dit? ").strip().lower()

    if decision == "a":
        return True, tool_input

    if decision == "e":
        edited = dict(tool_input)
        for key in edited:
            new_value = input(f"      {key} [{edited[key]}]: ").strip()
            if new_value:
                edited[key] = new_value
        return True, edited

    # Anything else (including "r") is treated as reject.
    feedback = input("      Reason for rejecting (sent back to the agent): ").strip()
    return False, {"feedback": feedback or "Rejected by human reviewer, no reason given."}


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
                tool_input = block.input

                # CONCEPT: the gate itself — one check, before dispatch,
                # for exactly the tools this policy flags. Everything
                # else in this loop is identical to a non-HITL tool loop.
                if block.name in REQUIRES_APPROVAL:
                    approved, tool_input = request_human_approval(block.name, tool_input)
                    if not approved:
                        print(f"  [rejected] {tool_input['feedback']}")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Action rejected by human reviewer: {tool_input['feedback']}",
                                "is_error": True,
                            }
                        )
                        continue

                print(f"  [tool] {block.name}({tool_input})")
                result_text, is_error = execute_tool(block.name, tool_input)
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

    print("Email assistant (human-in-the-loop demo). Type 'exit' to quit.\n")
    print("Try: \"Reply to Sara agreeing to move the budget review to Tuesday.\"\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
