"""
CONCEPT: Agent Handoff — passing control of a conversation from one agent
to another, permanently, rather than delegating a subtask and getting a
result back.

Contrast with ../orchestrator/: there, the orchestrator calls a
specialist, receives ITS result as a tool_result, and immediately
regains control to decide what happens next — the orchestrator is always
in charge, for the whole conversation. Handoff is the opposite: once
agent A hands off to agent B, B takes over completely — A doesn't get
control back, and every subsequent turn goes straight to B under B's own
system prompt and tools. This is the "triage" pattern: a first agent's
only job is figuring out WHO should handle a request, then it steps
aside for good.

The mechanic: a triage agent has transfer_to_X tools, one per specialist
it can route to. Calling one doesn't run a sub-task and return a result
the way ../orchestrator/'s delegate_to_X tools do — it just flips which
agent is "active" for the rest of the session. The very next model call,
even within the same turn, uses the new agent's system prompt and tools
— giving the user a response from the specialist directly, with no
further involvement from triage.

Use case: a customer support system with a triage agent routing to
billing, technical, or general support — whichever is chosen then
handles the rest of the conversation directly. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
EFFORT = "medium"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

TRANSFER_TOOLS = [
    {
        "name": "transfer_to_billing",
        "description": "Transfer the conversation to the billing specialist — for invoices, payments, refunds, subscriptions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "transfer_to_technical",
        "description": "Transfer the conversation to the technical support specialist — for errors, bugs, troubleshooting.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "transfer_to_general",
        "description": "Transfer the conversation to general support — for anything not specifically billing or technical.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# CONCEPT: each agent's full definition — a persona and, optionally,
# tools of its own. "triage" is the only one with transfer tools; once
# control leaves it, nothing hands the conversation back in this
# template (a natural extension: give specialists a transfer_to_triage
# tool too, for genuinely bidirectional handoff).
AGENTS = {
    "triage": {
        "system": (
            "You are a triage agent for customer support. Read the user's "
            "message and transfer to the right specialist using the "
            "transfer tools available. Don't try to solve the issue "
            "yourself — your only job is routing."
        ),
        "tools": TRANSFER_TOOLS,
    },
    "billing": {
        "system": "You are a billing support specialist. Help with invoices, payments, refunds, and subscription questions.",
        "tools": [],
    },
    "technical": {
        "system": "You are a technical support specialist. Help debug errors, troubleshoot issues, and give technical guidance.",
        "tools": [],
    },
    "general": {
        "system": "You are a general support specialist. Handle anything not specifically about billing or technical issues.",
        "tools": [],
    },
}

# CONCEPT: the currently active agent — this is the state a handoff
# mutates. Everything downstream (which system prompt, which tools) reads
# from here, not from a fixed choice made when the script started.
current_agent_name = "triage"


def run_turn(messages: list[dict]) -> None:
    """Same tool-calling loop shape as every other template in this repo
    — the thing that's different here is what happens when a
    transfer_to_X tool is called: instead of running a task and returning
    a result, it swaps `current_agent_name`, and the NEXT iteration of
    this very loop picks up the new agent's system prompt and tools.
    """
    global current_agent_name

    while True:
        agent = AGENTS[current_agent_name]
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=agent["system"],
            tools=agent["tools"],
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\n[{current_agent_name}] {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name.startswith("transfer_to_"):
                target = block.name.removeprefix("transfer_to_")
                print(f"  [handoff: {current_agent_name} -> {target}]")
                current_agent_name = target
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Transferred to {target}. You are now speaking with {target} support.",
                    }
                )

        messages.append({"role": "user", "content": tool_results})
        # Loop back to the top: `agent = AGENTS[current_agent_name]` now
        # resolves to the NEW agent, so the next API call in this same
        # turn already speaks as the specialist, not triage.


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Customer support (agent handoff demo). Type 'exit' to end the conversation.\n")
    print("Try: \"I was charged twice for my subscription this month.\"\n")

    messages: list[dict] = []

    while True:
        prompt = f"[{current_agent_name}] You: "
        user_input = input(prompt).strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
