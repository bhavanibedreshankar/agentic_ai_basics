"""
CONCEPT: Minimal Footprint Principle — an agent should request only the
permissions it actually needs for THIS SPECIFIC task, and avoid side
effects beyond that task, rather than being handed broad access "just in
case" it turns out to be useful.

This is the same underlying idea as
`../permission_scoping/permission_scoping.py` taken one level further.
Permission scoping grants a FIXED set of tools per ROLE — coarse-grained,
decided once per session, the same for every task that role ever
performs. Minimal footprint asks a narrower question for EVERY
individual request: "does *this* task actually need write access, or
would read-only cover it?" — and grants a set that can be narrower than
even the role's full allowance, re-derived fresh for each task rather
than fixed for the whole session.

The mechanic: every tool is tagged with the CAPABILITIES it requires
(`{"read"}`, `{"write"}`, `{"external_communication"}`). A small,
deterministic, LOCAL classifier — `infer_required_capabilities` — looks
at the task text for capability-indicating keywords and returns the
minimal capability set that task appears to need, defaulting to `{"read"}`
alone if nothing suggests otherwise. `select_tools_for_task` then grants
only the tools whose capabilities are a subset of what was inferred.
This classifier is deliberately simple keyword matching, NOT another LLM
call — asking a second model "what permissions does this need" would add
its own latency/cost/failure-mode on top of the very call it's supposed
to be minimizing exposure for.

Use case: a workspace assistant (search notes, edit a note, send a
digest email) where the tool grant for each request is computed fresh,
and visibly different, depending on what that specific request actually
asks for. Type 'exit' to quit.
"""

from __future__ import annotations

import os
import sys

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
EFFORT = "medium"
SYSTEM_PROMPT = "You are a workspace assistant. Only use the tools you've been given for this request."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

NOTES = {"todo": "Buy milk. Finish report.", "ideas": "Podcast about local history."}


def search_notes(query: str) -> str:
    matches = [f"{title}: {body}" for title, body in NOTES.items() if query.lower() in body.lower()]
    return "\n".join(matches) or "No matching notes."


def edit_note(title: str, new_content: str) -> str:
    NOTES[title] = new_content
    return f"Updated note '{title}'."


def send_digest_email(to: str, summary: str) -> str:
    return f"Digest email sent to {to}: {summary[:60]}..."


# CONCEPT: every tool tagged with what it actually requires. A read-only
# lookup needs no more than {"read"}; anything that mutates state needs
# "write"; anything that leaves the workspace entirely (an email out to
# someone) needs "external_communication" on top of that.
TOOL_REGISTRY = {
    "search_notes": {
        "capabilities": {"read"},
        "definition": {
            "name": "search_notes",
            "description": "Search notes for a query.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
        "implementation": search_notes,
    },
    "edit_note": {
        "capabilities": {"read", "write"},
        "definition": {
            "name": "edit_note",
            "description": "Edit a note's content.",
            "input_schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "new_content": {"type": "string"}},
                "required": ["title", "new_content"],
            },
        },
        "implementation": edit_note,
    },
    "send_digest_email": {
        "capabilities": {"read", "external_communication"},
        "definition": {
            "name": "send_digest_email",
            "description": "Send a summary email.",
            "input_schema": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["to", "summary"],
            },
        },
        "implementation": send_digest_email,
    },
}

# CONCEPT: the classifier — deliberately local, deterministic keyword
# matching, not a second LLM call. Each capability beyond the always-on
# baseline ("read") is only added if the task text contains one of its
# trigger words; nothing is granted "just in case."
CAPABILITY_TRIGGERS = {
    "write": {"edit", "update", "change", "rewrite", "modify"},
    "external_communication": {"email", "send", "notify", "message"},
}


def infer_required_capabilities(task: str) -> set[str]:
    words = set(task.lower().split())
    capabilities = {"read"}  # baseline: every task can at least look things up
    for capability, triggers in CAPABILITY_TRIGGERS.items():
        if words & triggers:
            capabilities.add(capability)
    return capabilities


def select_tools_for_task(task: str) -> tuple[list[dict], set[str], list[str]]:
    """Returns (tool definitions to grant, inferred capabilities, tool
    names granted) — the last two purely so the caller can print exactly
    what was decided and why, making the minimal-footprint computation
    visible rather than implicit.
    """
    required = infer_required_capabilities(task)
    granted_names = [
        name for name, entry in TOOL_REGISTRY.items()
        if entry["capabilities"] <= required   # tool needs nothing beyond what the task was inferred to need
    ]
    granted_defs = [TOOL_REGISTRY[name]["definition"] for name in granted_names]
    return granted_defs, required, granted_names


def execute_tool(granted_names: list[str], name: str, tool_input: dict) -> tuple[str, bool]:
    if name not in granted_names:
        return f"Error: '{name}' was not granted for this task.", True
    return TOOL_REGISTRY[name]["implementation"](**tool_input), False


def run_task(task: str) -> None:
    tools, required_capabilities, granted_names = select_tools_for_task(task)
    print(f"  [inferred capabilities: {sorted(required_capabilities)}]")
    print(f"  [granted tools: {granted_names} (of {len(TOOL_REGISTRY)} total available)]")

    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
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
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(granted_names, block.name, block.input)
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

    print(f"Workspace assistant (minimal footprint demo) — {len(TOOL_REGISTRY)} tools in the registry. Type 'exit' to quit.\n")
    print("Try: \"What's on my todo list?\" then \"Update my todo list to add 'call the bank'\" then \"Email a digest to sam@example.com\"\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() == "exit":
            print("Goodbye!")
            break
        if not task:
            continue

        run_task(task)
        print()


if __name__ == "__main__":
    main()
