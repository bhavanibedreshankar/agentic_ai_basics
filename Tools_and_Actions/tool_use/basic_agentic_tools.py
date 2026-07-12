"""
CONCEPT: Tool use (a.k.a. "function calling") — giving Claude the ability to
take actions, not just talk.

A tool-using agent has TWO loops working together:
  1. The OUTER loop (in main()) — same pattern as ../../Execution_Loops/agentic_loop/basic_agentic_loop.py: keep
     prompting the user and growing the conversation history until "exit".
  2. The INNER loop (in run_turn()) — new here: within a single user turn,
     Claude may ask to call one or more tools before it can give a final
     answer. We keep calling the API, running whatever tools it asks for,
     and feeding the results back, until Claude stops asking for tools.

This file's task manager (add_task / list_tasks / complete_task) is a stand-in
for any real tool — a database query, a web search, an email send, etc. The
mechanics are identical no matter what the tool actually does.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import anthropic

# --- API settings (see ../../basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a task management assistant. Use the available tools to add, list, "
    "and complete the user's tasks. After using a tool, briefly confirm what happened."
)

# Tasks are stored as plain JSON on disk next to this script, so state
# persists between runs of the program (not just between turns of one chat).
TASKS_FILE = Path(__file__).parent / "tasks.json"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# ---------------------------------------------------------------------------
# CONCEPT: Tool definitions (JSON Schema)
# ---------------------------------------------------------------------------
# Each tool is a plain dict with three parts:
#   - "name"        : the identifier Claude uses when it wants to call it
#   - "description" : tells Claude WHEN and WHY to use this tool — Claude
#                      relies entirely on this text to decide when a tool is
#                      relevant, so be explicit here
#   - "input_schema": a JSON Schema describing the arguments the tool takes.
#                      Claude uses this to construct valid, well-typed inputs.
#                      "required" lists which fields must be present; fields
#                      not listed are optional. "enum" restricts a field to a
#                      fixed set of values.
#
# Claude never executes these tools itself — it only requests a call (name +
# arguments). YOUR code is responsible for actually running the function and
# returning a result (see `execute_tool` below).
TOOLS = [
    {
        "name": "add_task",
        "description": "Add a new task to the task list. Call this when the user wants to create or add a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short description of the task"},
                "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format, if given"},
            },
            "required": ["title"],  # due_date is optional; title is not
        },
    },
    {
        "name": "list_tasks",
        "description": "List existing tasks, optionally filtered by status. Call this when the user asks what tasks they have.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "completed", "all"],  # Claude can only pick one of these three
                    "description": "Filter tasks by status. Defaults to 'all'.",
                },
            },
            # No "required" key at all — every field in this tool is optional,
            # so Claude can call list_tasks() with no arguments.
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as completed by its ID. Call this when the user says they finished a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The ID of the task to complete"},
            },
            "required": ["task_id"],
        },
    },
]
# NOTE on tool_choice: we don't pass a `tool_choice` param below, so it
# defaults to {"type": "auto"} — Claude decides on its own whether a tool is
# needed at all, and which one(s) to call. You could instead force a specific
# tool with {"type": "tool", "name": "add_task"}, force ANY tool with
# {"type": "any"}, or disable tools entirely with {"type": "none"}.


# ---------------------------------------------------------------------------
# Tool implementations — ordinary Python functions with no knowledge of the
# Claude API. These are the "real" actions a tool name maps to.
# ---------------------------------------------------------------------------

def _load_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        return []
    return json.loads(TASKS_FILE.read_text())


def _save_tasks(tasks: list[dict]) -> None:
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))


def add_task(title: str, due_date: str | None = None) -> dict:
    # CONCEPT: stateful tool — this doesn't just compute and return a value,
    # it has a real side effect (writing to tasks.json) that outlives the
    # conversation. Most useful agent tools work this way.
    tasks = _load_tasks()
    task = {"id": uuid.uuid4().hex[:8], "title": title, "due_date": due_date, "status": "open"}
    tasks.append(task)
    _save_tasks(tasks)
    return task


def list_tasks(status: str = "all") -> list[dict]:
    tasks = _load_tasks()
    if status == "all":
        return tasks
    return [t for t in tasks if t["status"] == status]


def complete_task(task_id: str) -> dict:
    tasks = _load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = "completed"
            _save_tasks(tasks)
            return task
    # Raising here is intentional — execute_tool below catches it and turns
    # it into an error result Claude can see and react to.
    raise ValueError(f"No task found with id '{task_id}'")


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Dispatch a tool call by name and return (result_text, is_error).

    CONCEPT: tool results must always be strings, and errors are reported
    via a boolean flag (`is_error`) rather than raising all the way up —
    this lets Claude see that something went wrong and decide how to
    recover (retry with different arguments, apologize to the user, etc.)
    instead of crashing the whole program.
    """
    try:
        if name == "add_task":
            task = add_task(**tool_input)
            return f"Added task {task['id']}: {task['title']}", False
        if name == "list_tasks":
            return json.dumps(list_tasks(**tool_input)), False
        if name == "complete_task":
            task = complete_task(**tool_input)
            return f"Completed task {task['id']}: {task['title']}", False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to Claude
        return f"Error: {exc}", True


# ---------------------------------------------------------------------------
# CONCEPT: The (inner) agentic tool-calling loop
# ---------------------------------------------------------------------------

def run_turn(messages: list[dict]) -> None:
    """Handle one user turn end-to-end, including any tool calls.

    Claude's response to a single user message might not be the final
    answer — it might be a request to call one or more tools first. So we
    loop: call the API, check what Claude wants, and either (a) print its
    final text and stop, or (b) run the requested tool(s) and call the API
    again with the results, repeating until Claude is done.

    Mutates `messages` in place so the outer chat loop in main() keeps the
    full history, including every tool call and result.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,  # <-- this is what makes tool use possible at all
            output_config={"effort": EFFORT},
            messages=messages,
        )
        # Claude's response (which may include tool_use blocks) becomes part
        # of the conversation history, exactly like a plain text reply would.
        messages.append({"role": "assistant", "content": response.content})

        # A single response can mix text and tool calls — print any text
        # Claude produced before/alongside its tool request.
        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        # response.stop_reason tells us WHY Claude stopped generating.
        # "tool_use" means: "I need a tool result before I can continue."
        # Anything else (e.g. "end_turn") means Claude is done — no more
        # tool calls needed, so we can return to the outer loop.
        if response.stop_reason != "tool_use":
            return

        # CONCEPT: parallel tool calls — Claude can request MULTIPLE tools
        # in one response (e.g. "add a task AND show me my list"). We must
        # execute every one of them and send ALL the results back together
        # in a single message — splitting them across multiple messages is
        # not supported and will confuse the conversation.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        # tool_use_id links this result back to the specific
                        # tool_use block that requested it — required so
                        # Claude can match results to calls when there are
                        # multiple in flight at once.
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        # Tool results are sent back as a "user" turn — from the API's
        # perspective, this is just more input for Claude to respond to.
        messages.append({"role": "user", "content": tool_results})
        # Loop back to the top: call the API again now that Claude has the
        # tool results, and see if it's ready to give a final answer.


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Set ANTHROPIC_API_KEY in your environment before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Task manager agent. Type 'exit' to end the conversation.\n")

    messages: list[dict] = []

    # ---- OUTER LOOP: same shape as ../../Execution_Loops/agentic_loop/basic_agentic_loop.py ----
    # Keeps the conversation going across turns; each iteration hands off to
    # run_turn(), which may itself loop several times internally to resolve
    # tool calls before returning.
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
