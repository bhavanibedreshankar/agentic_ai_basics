"""
CONCEPT: Permission Scoping — limiting which tools or resources an agent
can access based on the task context (here, the role of whoever is
using it), enforced by never TELLING the model those tools exist in the
first place, rather than telling it about everything and hoping it
behaves.

This is a different enforcement point from
`../guardrails/guardrails.py`: guardrails check a specific call's
CONTENT after the model has already decided to make it — the tool is
fully declared in `tools=`, and a call to it can still be rejected based
on its arguments. Permission scoping acts one step earlier: the tool
declaration itself is CONSTRUCTED per-session from a role, so a
low-privilege role's `tools=` list never contains `delete_ticket` at
all — Claude cannot request an action it was never told about, the same
way it can't call a Python function that was never imported. There's no
"the model asked and got rejected" moment for an out-of-scope tool here;
it never had the option.

Three roles, three tool sets, same underlying tool IMPLEMENTATIONS:
  - `read_only`  — search_kb, view_ticket
  - `agent`      — read_only's tools + update_ticket, escalate_ticket
  - `admin`      — agent's tools + delete_ticket

Each role's set is a NESTED superset of the one below it — built with
plain dict/list composition (`ROLE_TOOLS`), not duplicated tool
definitions, so a tool implementation is written once and its
availability is controlled purely by which roles' lists include it.

Use case: a support-ticket system, run three times in a row under
different roles so you can see the exact same request behave differently
depending on what that session's `tools=` list even contains. Type
'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

TICKETS = {
    "T1": {"subject": "Login issue", "status": "open"},
    "T2": {"subject": "Billing question", "status": "escalated"},
}


def search_kb(query: str) -> str:
    return f"Top KB article for '{query}': 'How to reset your password' (article #482)"


def view_ticket(ticket_id: str) -> str:
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        return f"No ticket found for ID {ticket_id}."
    return f"Ticket {ticket_id}: {ticket['subject']} (status: {ticket['status']})"


def update_ticket(ticket_id: str, status: str) -> str:
    if ticket_id not in TICKETS:
        return f"Error: no ticket found for ID {ticket_id}."
    TICKETS[ticket_id]["status"] = status
    return f"Ticket {ticket_id} updated to status '{status}'."


def escalate_ticket(ticket_id: str) -> str:
    return update_ticket(ticket_id, "escalated")


def delete_ticket(ticket_id: str) -> str:
    if ticket_id not in TICKETS:
        return f"Error: no ticket found for ID {ticket_id}."
    del TICKETS[ticket_id]
    return f"Ticket {ticket_id} permanently deleted."


# Every tool the SYSTEM knows how to run, regardless of role. This is
# NOT what gets shown to the model — ROLE_TOOLS below decides that.
ALL_TOOL_DEFS = {
    "search_kb": {
        "name": "search_kb",
        "description": "Search the knowledge base for an article matching a query.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    "view_ticket": {
        "name": "view_ticket",
        "description": "View a support ticket's details.",
        "input_schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
    },
    "update_ticket": {
        "name": "update_ticket",
        "description": "Update a ticket's status.",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"}},
            "required": ["ticket_id", "status"],
        },
    },
    "escalate_ticket": {
        "name": "escalate_ticket",
        "description": "Escalate a ticket to senior support.",
        "input_schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
    },
    "delete_ticket": {
        "name": "delete_ticket",
        "description": "Permanently delete a ticket.",
        "input_schema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
    },
}

TOOL_IMPLEMENTATIONS = {
    "search_kb": search_kb,
    "view_ticket": view_ticket,
    "update_ticket": update_ticket,
    "escalate_ticket": escalate_ticket,
    "delete_ticket": delete_ticket,
}

# ---------------------------------------------------------------------------
# CONCEPT: the scoping table itself — one dict mapping role -> the tool
# NAMES available to it. Everything else in this file just reads from
# here. `admin`'s list is built by extending `agent`'s at import time, so
# adding a tool to `agent` automatically makes it available to `admin`
# too, without editing `admin`'s definition — the nesting is structural,
# not duplicated by hand.
# ---------------------------------------------------------------------------
ROLE_TOOLS: dict[str, list[str]] = {
    "read_only": ["search_kb", "view_ticket"],
    "agent": ["search_kb", "view_ticket", "update_ticket", "escalate_ticket"],
}
ROLE_TOOLS["admin"] = ROLE_TOOLS["agent"] + ["delete_ticket"]

SYSTEM_PROMPTS: dict[str, str] = {
    "read_only": "You are a support assistant with read-only access. You can search the knowledge base and view tickets, but cannot modify anything.",
    "agent": "You are a support agent. You can search, view, update, and escalate tickets.",
    "admin": "You are a support admin with full access, including deleting tickets.",
}


def build_tools_for_role(role: str) -> list[dict]:
    """CONCEPT: this is the actual scoping mechanism — a `tools=` list
    built from ROLE_TOOLS[role], containing ONLY the schemas for tools
    that role is allowed to see. Nothing downstream needs to check
    permissions at call time, because an out-of-scope tool is never in
    this list to begin with.
    """
    return [ALL_TOOL_DEFS[name] for name in ROLE_TOOLS[role]]


def execute_tool(role: str, name: str, tool_input: dict) -> tuple[str, bool]:
    # Defense in depth: even though an out-of-scope tool should never be
    # requested (it was never declared to the model), double-check here
    # too, in case something else in a larger system calls execute_tool
    # directly without going through the scoped `tools=` list.
    if name not in ROLE_TOOLS[role]:
        return f"Error: role '{role}' is not permitted to use '{name}'.", True
    return TOOL_IMPLEMENTATIONS[name](**tool_input), False


def run_turn(role: str, messages: list[dict]) -> None:
    tools = build_tools_for_role(role)
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPTS[role],
            tools=tools,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\n[{role}] Claude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(role, block.name, block.input)
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

    print(f"Permission scoping demo. Roles: {list(ROLE_TOOLS)}. Type 'exit' to quit.\n")

    role = ""
    while role not in ROLE_TOOLS:
        role = input(f"Choose a role ({'/'.join(ROLE_TOOLS)}): ").strip()

    print(f"\nRunning as '{role}' — available tools: {ROLE_TOOLS[role]}\n")

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(role, messages)


if __name__ == "__main__":
    main()
